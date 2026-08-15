# Lab3 GDN Prefill 优化记忆

> 本文档记录 Lab3 GDN prefill forward kernel 的完整优化历程、关键决策、实测数据和踩坑经验。
> 供后续继续优化或写实验报告时参考。所有性能数据基于 **H800 MIG 1g.10gb (14 SM, 9.75 GiB, CC 9.0)**。

---

## 1. 项目背景与约束

### 任务
用 TileLang 实现 Gated DeltaNet 的 prefill 前向 kernel (`gdn_prefill_forward`)，只实现核心的 U/W/S/O 计算
（g_cumsum 和 A 已由 preprocessing 给出，不在计时区）。基线是 FlashQLA。

### 计算语义（每 chunk, C=64, d=128）
```
γ = exp(g_cumsum), γr = γ[ℓ-1]  (尾块取最后有效 token)
W = A @ (β⊙γ⊙K),  U = A @ (β⊙V),  ds = Lower(QKᵀ) ⊙ (γ_i/γ_j),  sK = K ⊙ (γr/γ)
V_new = U − W @ S_old
S_new = γr·S_old + sKᵀ @ V_new
O = scale·[ γ⊙(Q@S_old) + ds @ V_new ]      scale = 128**-0.5
```

### 接口与约束
- 输入: q/k [B,T,Hq,128] BF16; v [B,T,Hv,128] BF16; g_cumsum/beta [B,T,Hv] FP32; A [B,T,Hv,64] BF16; initial_state [B,Hv,128,128] FP32 或 None
- 输出: output [B,T,Hv,128] BF16; final_state [B,Hv,128,128] FP32
- 容差: RTOL = ATOL = 5e-3
- 只能修改 `student/tilelang_fwd.py`
- 只能用 TileLang，禁止调 reference/FlashQLA/FLA/FlashInfer 完成被测计算

### 8 个公开 case (evaluation/cases.csv)
| case | B | T | Hq | Hv | state | gate_mode |
|------|---|---|----|----|-------|-----------|
| short_tail_state | 1 | 1025 | 2 | 8 | yes | random_decay |
| chain_equal | 1 | 8192 | 4 | 4 | no | random_decay |
| parallel_equal | 1 | 2048 | 16 | 16 | no | random_decay |
| parallel_gva | 1 | 2048 | 4 | 16 | no | mixed |
| long_low_gva | 1 | 32768 | 2 | 8 | no | random_decay |
| batch_split_gva | 4 | 8192 | 2 | 8 | no | random_decay |
| wide_gva_state | 1 | 8192 | 16 | 64 | yes | random_decay |
| deep_gva_state | 1 | 16384 | 8 | 32 | yes | random_decay |

### 集群使用
- DevPod: x86-5418Y 预设，**无 GPU**，只做编辑
- GPU 在 `lab3` 分区计算节点: `hpc submit -p lab3 "cd ~/HPC101_homework/lab3 && python run.py --case <name>"`
- lab3 限制: 8 CPU, 5min walltime, **maxJobs=1**（同时只能跑 1 个 job）
- python 在 `/opt/lab3-venv/bin/python`（PATH 已含，直接 `python` 即可）
- DevPod 上 `python3` 是系统 python，**无 torch/tilelang**，只能 `python3 -m py_compile` 做语法检查
- 代码同步: 用 `scp -P 443 本地文件 h3250105245+lab2+hpc101@clusters.zju.edu.cn:~/HPC101_homework/lab3/student/` 直接传，**不要**用 git push/pull 污染提交记录

### 设备实测属性 (H800 MIG 1g.10gb)
- SMs = 14
- sharedMemPerBlock = 49152 (48 KB, 静态)
- sharedMemPerBlockOptin = 232448 (227 KB, 动态 optin 上限)
- **但 TileLang 动态 shared 申请门槛更低**: 实测 ~238KB 就会被拒 ("Failed to set the allowed dynamic shared memory size to 238592")

---

## 2. 开发循环与调试经验

### 提交流程
```
本地编辑 tilelang_fwd.py
→ scp -P 443 传到 DevPod ~/HPC101_homework/lab3/student/
→ ssh -p 443 ... 'cd ~/HPC101_homework/lab3 && hpc submit -p lab3 "cd ~/HPC101_homework/lab3 && python run.py --case <name>"'
→ 看输出 (PASS/FAIL + median ms)
```

### 常见错误与修复

1. **`local_buf W_shared must be a fragment, but got shared.dyn`**
   - 原因: `T.gemm` 的输出（accumulator）必须是 fragment，不能是 shared
   - 修复: GEMM 结果先进 fragment，再 `T.copy(fragment, shared)`

2. **`A and B must have the same dtype`**
   - 原因: `T.gemm(A, B, C)` 要求 A/B 同 dtype（BF16×BF16）
   - 修复: 把 FP32 fragment 结果 `T.copy` 成 BF16 shared 再当 GEMM 操作数

3. **`kernel.initial_state is expected to have non-NULL pointer`**
   - 原因: TileLang kernel 参数不能传 None
   - 修复: wrapper 里 `if initial_state is None: initial_state = torch.zeros(...)`

4. **`Failed to set the allowed dynamic shared memory size to 238592`**
   - 原因: shared buffer 总和超过 MIG 动态 shared 上限
   - 修复: 复用 shared buffer（时序错开），或把大 buffer 改 fragment

5. **`T.gemm N shape check failed: N_B = 128, N_C = 64`**
   - 原因: GEMM 输出 fragment 的 N 维与操作数 B 的 N 维不匹配（复用 fragment 时尺寸不对）
   - 修复: 用不同尺寸的 fragment（tmp_dv [64,128] 给 W，tmp_dv2 [64,block_DV] 给 V_new）

6. **正确性回归（结合律版 + exp 复用）**
   - 现象: chain_equal/parallel_equal 17% mismatch，greatest abs diff 0.08
   - 原因: 结合律 `W@S = A@(βγK@S)` 中间结果 `βγK@S` 若用 BF16 截断两次（βγK→BF16, βγK@S→BF16）累积误差
   - 修复: `βγK@S` 保 FP32 fragment，只在 `A@它` 前拷成 BF16 shared（一次截断）。但实测仍有小误差——最终通过把 O 的 scale 合并到最后统一乘（不在中间乘 scale）解决

### TileLang 语法要点（实测确认）
- `T.gemm(A, B, C, transpose_A=True/False, transpose_B=True/False, clear_accum=True/False)`: A/B 须同 dtype（BF16），C 是 fragment 累加器（FP32）
- `clear_accum=False`: 累加进现有 C（用于 O = O_st + O_in）
- `T.copy(src, dst)`: src/dst 可以 global/shared/fragment 任意组合；FP32 fragment → BF16 shared 会截断
- `T.alloc_shared((shape), dtype)`: block 内共享
- `T.alloc_fragment((shape), dtype)`: 寄存器（Tensor Core 累加器）
- `T.alloc_local((shape), dtype)`: 标量寄存器
- `T.Parallel(d1, d2)`: 把循环映射到 thread 并行
- `T.Pipelined(num_iters, num_stages=N)`: 软流水，**仅重叠 load↔compute**，不重叠 compute↔compute
- `T.min(a, b)`: 内置 min
- `T.exp2(x)`: 2^x（比 T.exp 快，配合 LOG2E 用）
- `T.cast(x, dtype)`: 类型转换
- `T.if_then_else(cond, a, b)`: 条件
- `T.use_swizzle(10)`: 启用 swizzle 布局（减少 bank conflict）
- `T.disable_warp_group_reg_alloc()`: 让编译器自由分配寄存器（降 reg/thread）
- `T.annotate_layout({buf: make_swizzled_layout(buf)})`: 手动指定 swizzle
- `@tilelang.jit(out_idx=[-2, -1])`: 末尾两个 tensor 参数为输出
- `pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}`: 启用快速数学

### ncu profiling 命令
```bash
hpc submit -p lab3 "cd ~/HPC101_homework/lab3 && ncu --clock-control none -k regex:kernel_kernel --csv --metrics <metrics> python run.py --case <name>"
```
- `--clock-control none`: MIG 不能锁频，必须加这个
- 关键 metrics:
  - `launch__registers_per_thread`: 寄存器数
  - `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed`: TC 利用率
  - `sm__sass_thread_inst_executed_op_fmul_pred_on.sum`: fmul 指令数（element-wise 开销）
  - `launch__waves_per_multiprocessor`: SM 占用（block 数 / SM 数）
  - `launch__shared_mem_per_block_static`: 静态 shared

---

## 3. 优化历程与关键决策

### 版本演进（git log 顺序）

#### v1: 朴素 V_new 形式 (初始正确版)
- 单 fused kernel, state 跨 chunk 驻留 shared/fragment
- `T.Pipelined` 软流水, block_DV=64, threads=128, num_stages=2
- 8 case 全 PASS
- 性能: short_tail 0.195, long_low 4.125, wide 5.439

#### v2: WY 仿射形式 (尝试, **失败**)
- 思路: 消 V_new, `S_new = T·S + b`, `O = P·S + dsU`，P2 单胖 GEMM 关键路径最短
- **致命问题**: T=[128,128] 需物化为 shared 做 T·S GEMM，这块 32KB(BF16) 加其他 buffer 总 shared 达 238KB，**超过 MIG optin 上限 232KB**
- 放 fragment 更糟（255 regs 已满，spill 严重）
- split-dv 不行——T·S 需要完整 T，不能沿 dv 切
- **结论**: WY 在 shared 受限的 MIG 上走不通。基线 FlashQLA 用朴素 V_new 形式，说明 V_new 实践上未必差。WY 是未验证的赌注，需实测。

#### v3: 降寄存器 (结合律消 W) (**部分成功**)
- 思路: `W@S = A@(βγK@S)`，不物化 W [64,128]，省 8192 floats 寄存器 + 16KB shared
- fragment 复用: U/V_new/O 共用 tmp_dv，算完即拷走
- 寄存器 255 → 234 → (加 disable_warp_group_reg_alloc) 215
- **短序列受益**: short_tail 0.195 → 0.143（快 27%）
- **长序列退化**: long_low 4.125 → 5.108（多一次 GEMM + fragment→shared 拷贝累积开销）
- 原因: 结合律多一次 GEMM（βγK@S 再 A@它），短序列 launch 开销占比大所以受益，长序列 chunk 多累积开销大

#### v4: per-case 分发 (**成功**)
- 短序列 (T<=2048): 结合律版（寄存器少，launch 占比大受益）
- 长序列 (T>2048): 物化W版（GEMM 数少，chunk 多累积开销小）
- 各取最优，8 case 全 PASS

#### v5: block_DV 调参 (**32 失败, 64 最优**)
- 试 block_DV=32 提 SM 占用 → 反而变慢（long_low 5.49 vs 4.31）
- 原因: 32 tile 太细，mma 效率下降抵消 SM 占用收益
- 试 block_DV=128 → 寄存器溢出（parallel_equal 0.44→0.94，慢 2x）
- **结论**: block_DV=64 对所有 case 最稳

#### v6: num_stages 调参 (**2 最优**)
- num_stages=3 → 反而变慢且超时
- num_stages=1 → 略慢
- **结论**: num_stages=2

#### v7: threads 调参 (**128 最优**)
- threads=256 → 无收益（寄存器分摊但调度开销增加）

#### v8: exp 复用 (**长序列成功, 短序列需回退**)
- 预算 `g_exp[64]`/`g_inv[64]`/`beta_g[64]` 三个 shared 数组，复用到 βγK/ds/O_st/gate_V_new
- ncu: fmul 383M → 339M（-12%），TC 13.5% → 13.9%
- 长序列受益: long_low 4.30 → 3.95，wide 5.59 → 5.07，deep 5.61 → 5.00
- **短序列回归**: short_tail 0.143 → 0.188（chunk 少，exp 预算循环开销 > 复用收益）
- 修复: 短序列（结合律版）回退 inline exp，长序列（物化W版）保留 exp 复用

#### v9: @autotune 自动搜参 (**长序列大幅成功**)
- 用 `@tilelang.autotune` 搜 7 个配置 (block_DV × threads × num_stages)
- **关键发现**: block_DV=128 + threads=256 + stages=1 对长序列最优
  - 之前手动测 block_DV=128 寄存器溢出，但配合 threads=256（寄存器分摊到 2 warp group）+ stages=1（减流水 buffer）反而最优
- 实测提升: long_low 4.00→3.22 (-20%), wide 5.14→4.11 (-20%), deep 4.97→4.86, batch_split 2.54→2.45, parallel_gva 0.47→0.42 (-11%)
- 短序列 autotune 选 block_DV=64 stages=1，与手动一致

#### v10: 扩展 swizzle (**收益微小**)
- 结合律版: 加 bkg_shared swizzle（Q/K/W 仍因 pipeline 冲突不加）
- 物化W版: 加 Q/K/bkg/W swizzle（该版无 pipeline 冲突）
- 实测: 几乎持平（long_low 3.22→3.21, wide 4.11→4.14 略退, parallel_gva 0.42→0.43）
- **结论**: bank conflict 非 TC 13.9% 的主瓶颈，寄存器限制才是

#### v11: @autotune 自动搜参 (**性能成功, 但 OJ 超时致命**)
- 用 `@tilelang.autotune` 搜 7 个配置 (block_DV × threads × num_stages)
- **性能**: long_low 4.00→3.22 (-20%), wide 5.14→4.11 (-20%), parallel_gva 0.47→0.42 (-11%)
- autotune 选出: 短序列 DV=64/th=128/st=1~2; 长序列 **DV=128/th=256/st=1**
  - 之前手动测 DV=128 寄存器溢出，但配合 th=256（寄存器分摊到 2 warp group）+ st=1（减流水 buffer）反而最优
- **★ OJ 灾难**: 提交后 wide_gva_state / deep_gva_state / 全部 4 个 hidden case **超时 285s 拿 0 分**
  - 根因: `@autotune` 每个配置要 warmup=3 + rep=10 次实测择优，长序列 case (T=8192/16384) × 7 配置 × 13 次 = 大量编译+运行，OJ 5min walltime 超限
  - 对比: 5d25f18（手动固定 DV=64）OJ 92/120 全 PASS; 2633345（autotune）OJ 43/120，6 个 case 超时 0 分
- **★ 教训**: **OJ 5min walltime 下绝不能用 @autotune**。autotune 只在本地离线搜参用，搜完必须把最优配置硬编码回代码
- **修复 v12**: 移除 `@autotune` 装饰器和 `_AUTOTUNE_CONFIGS`，把搜出的最优配置作为默认参数硬编码:
  - `_gdn_naive_kernel` (短序列): `block_DV=64, threads=128, num_stages=2`
  - `_gdn_naive_kernel_matw` (长序列): `block_DV=128, threads=256, num_stages=1`
  - `gdn_prefill_forward` 分发时显式传这三个参数

### 最终架构 (commit c8cd8b9 + v12 autotune 回退)

**分发逻辑** (`gdn_prefill_forward`):
```python
if num_tokens <= 2048:
    block_DV = 64
    kernel = _gdn_naive_kernel(...)        # 结合律版, inline exp
else:
    block_DV = 64
    kernel = _gdn_naive_kernel_matw(...)   # 物化W版, exp 复用
```

**结合律版 `_gdn_naive_kernel`** (短序列):
- `W@S = A@(βγK@S)`，不物化 W
- fragment: tmp_dv[64,block_DV], ds_tmp[64,64], O_fragment[64,block_DV]
- inline exp2（每处直接算）
- `T.disable_warp_group_reg_alloc()`

**物化W版 `_gdn_naive_kernel_matw`** (长序列):
- W=A@βγK 驻 shared，直接 W@S
- fragment: tmp_dv[64,128](W产出), tmp_dv2[64,block_DV](V_new等), ds_tmp[64,64], O_fragment[64,block_DV]
- exp 复用: g_exp/g_inv/beta_g 预算一次复用
- `T.disable_warp_group_reg_alloc()`

### 正确性要点 (最终版已全部处理)
1. `scale = 128**-0.5`（不是 1/128）
2. `γr = g_cumsum[start + ℓ - 1]`（尾块取最后**有效** token，不是 index 63）
3. 尾块 Q/K/V/β 补零；A padding 列已是单位行（kkt_solve 保证，写 `1 if t==d else 0`）
4. gate 用差值形式 `exp2((γr-g)*LOG2E)` 或 `gl * g_inv`，**不用比值** `γr/γ`（padding 位 x/0→nan）
5. GVA: `bhg = bh // (Hv//Hq)`，index 映射不展开
6. initial_state=None 时传零张量
7. **O 用 S_old**（更新 S 之前算 O）：P3 必须在 P2 更新 S 之前
8. 输出尾块只写前 ℓ 行

---

## 4. 最终性能数据 (core forward, H800 MIG 10G)

| case | student(ms) | FlashQLA(ms) | vs FlashQLA | 策略 (autotune 选) |
|------|------------|-------------|-------------|------|
| short_tail_state | 0.146 | 0.271 | **1.86x 快** | 结合律, DV=64,th=128,st=1 |
| parallel_equal | 0.45 | 0.467 | **1.04x 快** | 结合律, DV=64 |
| parallel_gva | 0.43 | 0.434 | **1.01x 快** | 结合律, DV=64 |
| chain_equal | 0.53 | 0.450 | 0.85x | 结合律, DV=64 |
| long_low_gva | 3.21 | 1.824 | 0.57x | 物化W, DV=128,th=256,st=1 |
| batch_split_gva | 2.44 | 1.803 | 0.74x | 物化W, DV=128,th=256,st=1 |
| wide_gva_state | 4.14 | 3.363 | 0.81x | 物化W, DV=128,th=256,st=1 |
| deep_gva_state | 4.85 | 3.636 | 0.75x | 物化W, DV=128,th=256,st=1 |

**3 个 case 超过 FlashQLA**（short_tail 1.86x, parallel_equal 1.04x, parallel_gva 1.01x）

### 优化进展 (v1 → 最终)
- short_tail: 0.195 → 0.146（快 25%）
- long_low: 4.125 → 3.21（快 22%）
- wide_gva: 5.439 → 4.14（快 24%）
- deep_gva: 5.157 → 4.85（快 6%）
- chain_equal: 0.561 → 0.53（快 5%）

---

## 5. ncu 分析数据 (long_low_gva, 物化W版+exp复用+autotune DV=128)

| 指标 | v1 朴素 | v8 (DV=64) | 最终 (DV=128,th=256) | 目标 |
|------|---------|--------|--------|------|
| reg/thread | 255 | 220 | ~160 (th=256 分摊) | <128 (让 2 block/SM) |
| TC 利用率 | 13.8% | 13.9% | 待测 (预期升) | >50% |
| fmul 指令数 | 383M | 339M | 待测 | 更低 |
| waves/SM | 1.14 | 1.14 | 待测 | >2 |
| `__launch_bounds__` | (128, 1) | (128, 1) | (256, 1) | (128, 2) |

**核心瓶颈仍存**: 即便 threads=256 分摊寄存器到 2 warp group，TC 利用率仍受限于 element-wise 与 GEMM 串行（单 warp 组下 TC 等 element-wise 跑完才轮到）。
autotune 的 DV=128+th=256+st=1 突破点在于: 大 tile 提升 mma 效率 + 寄存器分摊降压力 + 单级流水减 buffer，三者协同。

---

## 6. 尝试过但未采用的方案

### WY 仿射形式
- 失败原因: T=[128,128] 物化超 shared 上限 (238KB > 232KB)
- 教训: WY 消 V_new 的代价是物化 T/b/P/dsU，shared 压力反增。在 shared 受限的 MIG 上，WY 的"关键路径短"优势被 shared 压力抵消
- 基线 FlashQLA 用朴素 V_new 形式，佐证 V_new 实践上未必差

### block_DV=32
- 失败原因: tile 太细，mma 效率下降抵消 SM 占用收益
- 教训: MIG 上 TC 效率比 SM 占用更重要

### block_DV=128
- 失败原因: s_fragment [128,128] 寄存器溢出
- 教训: 大 fragment 会 spill，反而变慢

### num_stages=3
- 失败原因: 反而变慢且超时
- 教训: 更多流水级 = 更多 shared buffer + 更高寄存器压力

### threads=256
- 失败原因: 无收益（寄存器分摊但调度开销增加）
- **修正 (v11)**: 配合 block_DV=128 + stages=1 时，threads=256 反而最优（autotune 发现）。单独调 threads=256 无益，但三者协同突破

### exp 复用用于短序列
- 失败原因: chunk 少时 exp 预算循环开销 > 复用收益
- 教训: 优化要分 case，短序列和长序列瓶颈不同

### @autotune 用于 OJ 提交 (**最严重踩坑**)
- 失败原因: OJ 5min walltime 下，autotune 对长序列 case × 7 配置 × 13 次(warmup3+rep10) 编译运行超时，6 个 case 拿 0 分（43/120）
- 教训: **autotune 只能本地离线搜参，搜完必须硬编码最优配置回代码再提交 OJ**。OJ 的 285s 超时远比单 case 编译+运行严格

### Warp specialization 探针 (Phase A, 进行中)
- 目标: 用 2-WG producer/consumer 验证 TileLang warp-spec API 在 MIG 上可用，为 4-WG 铺路
- 调试开关: `GDN_WS_PROBE=1` 环境变量，`gdn_prefill_forward` 分发到 `_gdn_ws_probe_kernel`；默认走老 kernel 保 92 分基线
- **★ 核心教训 1: T.Parallel element-wise store 对 wgmma async proxy 不可见 → NaN**
  - producer 用 `T.Parallel` 写 shared，consumer 用 `T.gemm`(wgmma) 读 → 读到脏数据/未定义值 → NaN
  - `T.fence_proxy_async()` **不够**（wgmma 异步代理仍看不到 T.Parallel 的常规 store）
  - 修复: load 改用 `T.copy`（走 TMA 路径，TMA 的 shared 写天然对 async proxy 可见）
  - g/beta 标量例外: consumer 用 `T.Parallel`(非 wgmma) 读，故 producer `T.Parallel` store + `fence_proxy_async` 即可
- **★ 核心教训 2: mbarrier parity 语义 (照抄官方 example_warp_specialize_gemm_barrierpipe_stage2)**
  - `wait_parity(p)` 阻塞直到 phase parity != p（不是 ==）。init phase=0，arrive 满 arrive_count 次后翻转 0→1→0
  - 双 buffer (num_stages=2) barrier 布局: `mbars = T.alloc_barrier([128,128]*2)` → 4 个实例
    - `[0,1]`=data_is_ready (producer→consumer), `[2,3]`=data_is_free (consumer→producer)
  - 公式 (buf = i_c % num_stages):
    - producer wait `mbars[buf + num_stages], ((i_c//num_stages)%num_stages)^1`
    - consumer wait `mbars[buf], (i_c//num_stages)%num_stages`
  - arrive_count=128 (一个 warp group 的 128 threads 都 arrive)
- **★ 核心教训 3: T.use_swizzle(10) 在 warp-spec 下不要用**
  - 官方 warp_specialize 示例均不用 `T.use_swizzle`。它干扰 mbarrier 代码生成
  - bank-conflict 优化改用 `T.annotate_layout({buf: make_swizzled_layout(buf)})`
- **★ 核心教训 4: threads 必须 ≥ 256 才有 2 个 warp group**
  - `T.ws(0)` = [0,128), `T.ws(1)` = [128,256)。threads=128 只有 1 个 wg，ws(1) 无线程执行 → 死锁
- **★ 核心教训 5: 尾块 T.copy 动态切片问题**
  - `T.copy(Q[bb, left:last, ...], Q_shared[buf, 0:last-left, :])` 用运行时动态长度切片
  - 无尾块 case (chain_equal/parallel_equal/long_low) 全 PASS
  - 有尾块 case (short_tail T=1025, 尾块1 token) FAIL: NaN @ index 64 (尾块起点)
  - 推测: TileLang T.copy 不支持运行时动态切片长度，或切片后 shared 未覆盖部分读到垃圾
  - 待修: 尾块改用固定 block_S 切片 + 越界 mask，或对最后一个 chunk 特殊处理
- **探针性能 (2-WG, 未优化, 仅验证)**:
  - chain_equal 1.16ms (基线 0.53ms, 慢 2x — 预期, 2-WG 没藏 element-wise, 仅验证 API)
  - long_low 4.56ms (基线 3.21ms, 慢 — 同理)
  - 性能不是探针目标，Phase B 4-WG 才追求性能

### Phase B: 2-WG element-wise 搬移 (**无收益, 结构局限**)
- 尝试: 把 βγK/βV/g_exp/g_inv/beta_g 从 consumer 搬到 producer (ws1), 期望藏进 consumer GEMM 周期
- 实测: chain_equal 1.16→1.22ms (持平), long_low 4.56→4.90ms (略慢)
- **★ 根因: 2-WG 在 GDN 上做不到 element-wise 藏进 GEMM**
  - GDN 的 element-wise 产物 (βγK/βV) 恰是 consumer 第一个 GEMM (W=A@bkg, U=A@bv) 的操作数
  - 时序上 element-wise 卡在 consumer 启动前, 无法重叠
  - 双 ready barrier (input_ready + ew_ready) 只能让 QKᵀ(1/7 GEMM, 不依赖 ew) 与 producer ew 重叠, 收益太小
- **★ FlashQLA 能藏 element-wise 的真正机制: 4-WG 让 3 个 consumer 并行跑不同 GEMM**
  - S/V/O 各一个 consumer 并行 GEMM, producer 的 ew 藏进"其他 consumer 正在跑 GEMM"的周期
  - 不是藏进"同一个 consumer 的下一个 GEMM" (2-WG 的错误假设)
- **结论**: 2-WG 止步于此 (已验证 API + 数学正确), 性能没提升是 2-WG 结构局限而非实现错误
- **回退**: 代码回退到 Phase A 探针版 (f7b5cdb), Phase B 失败改动丢弃; 默认走 matw 版 (OJ 92 分基线安全)

### Phase B: 4-WG warp specialization (**死锁, 未通过**)
- 尝试: 4-WG (threads=512), FlashQLA 数学形式
  - CONSUMER_S [0,128): h*=γr; h+=Kᵀ@V'; 发布 h_shared(S_old)
  - CONSUMER_V [128,256): U=K@S; W=V−g⊙U; Vd=A@W; V'=ratio⊙Vd; 发布 vn_shared
  - CONSUMER_O [256,384): P=Q@Kᵀ; G; O=Q@S; O+=(scale·G⊙P)@V'
  - PRODUCER [384,512): load + ew g_exp/g_inv
- barrier: data_is_ready[2](128), data_is_free[2](384), h_ready[2](128), vn_ready[2](128)
- **结果**: chain_equal 编译通过但运行死锁 (walltime timeout, 无 PASS/FAIL)
- **★ 踩坑 1: T.gemm 要求操作数同 dtype**
  - `T.gemm(p_fragment, vn_shared, ...)` 中 p_fragment 是 FP32, vn_shared 是 BF16 → "A and B must have the same dtype"
  - 修复: 加 `p_shared = T.alloc_shared(BF16)`, `T.copy(p_fragment, p_shared)` 再 gemm
- **★ 踩坑 2: mbarrier parity 公式必须用 (i_c//num_stages)%num_stages, 不是 i_c%2**
  - 错误 `parity = i_c % 2`: chunk1 buf=1 wait(ready[1], 1), init0, 0!=1 提前返回读脏数据
  - 正确 `parity = (i_c//2)%2`: chunk0=0, chunk1=0, chunk2=1, 每个 barrier 实例服务 2 chunk 才翻转
- **★ 踩坑 3: data_is_free 必须双 buffer, 不能单实例**
  - 单实例 data_is_free[0]: chunk1 producer wait(free[0]) 等 chunk1 consumer arrive, 但 consumer 等 ready[1] (producer 还没 arrive) → 循环死锁
  - 修复: data_is_free[2] 双 buffer, chunk1 buf=1 用 data_is_free[1]
- **★ 踩坑 4 (致命, 未解决): T.ws(3) 疑似在 TileLang 不支持或行为异常**
  - 官方示例只用 ws(0)/ws(1); FlashMLA 用 `if tx<128 else` 手写
  - ws(2)/ws(3) 下 kernel 编译通过但运行死锁, 无法确定是 ws(3) 问题还是 barrier 时序
  - 修复方向: 改用 `tx = T.get_thread_binding()` + `if tx<128/elif/<256/elif/<384/else` 手写 4 分支 (flashmla 风格)
- **结论**: 4-WG 在 TileLang 上的 barrier handshake 极易死锁, 调试成本高 (远程 5min walltime)
  - 已验证: barrier 语义、parity 公式、双 buffer 必要性、dtype 要求
  - 未解决: ws(3) 支持性 / 完整 4 分支死锁定位
  - **回退**: 代码回退到 Phase A 探针版 (f7b5cdb), 4-WG 代码丢弃; 默认走 matw 版 (OJ 92 分基线安全)
  - 下次若重试 4-WG: 用 `tx = T.get_thread_binding()` 手写 4 分支替代 T.ws(3)

### 4-WG tx 手写分支 (**barrier 调通, 卡 2.7% 精度**)
- 用 `tx = T.get_thread_binding()` + `if tx<128/elif/<256/elif/<384/else` 手写 4 分支 (flashmla 风格)
- **调通过程** (8 轮集群迭代):
  1. `U=K@S` 错用 `transpose_B=True` → 去掉 (59.6% mismatch)
  2. `W` 缺 `β` 因子 → `W=β(V−γU)` (56.4%)
  3. O 用 `V'`(γr/γ·v_new) 应改用 `Vd`(=v_new) → 分离 vd/vn 双 buffer (2.7%)
  4. matw 数学 (A@βγK 物化) 替代 FlashQLA 数学 → 仍 2.7% (非数学形式问题)
  5. `TL_DISABLE_WGMMA=True` pass_config → 仍 2.7% (非 wgmma 问题)
- **★ 2.7% 根因定位**: matw+T.serial 单 kernel **PASS** (0.886ms), 4-WG 同数学 FAIL
  - 误差来自多 WG 同步, 非 T.serial / wgmma / 数学形式
  - 推测: 跨 WG `T.copy(fragment, shared)` + `fence_proxy_async` 对 threads=512 的 async proxy 可见性不足
  - 与 Phase A 教训 1 一致 (T.Parallel store 对 wgmma 不可见→NaN), 4-WG 下是部分可见→2.7% 精度差
- **结论**: 4-WG 在 TileLang + MIG 上精度无法通过 RTOL=5e-3 (当前 TileLang 版本限制)
- 回退到 f7b5cdb 基线

### 长序列调参矩阵 (**DV=128/th=256/st=1 全局最优, 无提升空间**)
| case | DV=128/th=256/st=1 | DV=64/th=256/st=1 | DV=64/th=128/st=2 | DV=128/th=512 |
|------|---------------------|--------------------|--------------------|---------------|
| long_low | 3.21ms (基线) | 4.98ms | 3.93ms | 精度回归 |
| wide | 4.14ms (基线) | 6.37ms | - | FAIL |
| deep | 4.85ms (基线) | 6.30ms | - | FAIL |
| batch_split | 2.44ms (基线) | 3.08ms | - | - |
- threads=512 → wide/deep FAIL (abs 0.025, wgmma 精度回归)
- split-DV (DV=64) 全线更慢: grid 收益 < TC 效率损失 (v10 教训一致)
- per-case 离线 autotune 硬编码 → 无提升 (DV=128 已最优)

### chunk 间 state 并行 (**死路**)
- GDN state 递推: `S[c] = (γr·I - sK^T@W)@S[c-1] + sK^T@U`, 需 [128,128] 矩阵 scan
- WY v2 已证明 T=[128,128] 超 shared 上限, 走不通
- FLA `chunk_h_parallel` 两阶段 (partial + scan) 不适用 GDN delta rule

---

## 7. 下一步优化方向

### 目标: OJ 110 分

当前 ~92-95 分 (95ba1d4 回退 autotune 后)。110 分需要长序列 4 case 平均 ~113 分 (p≈1.4, 即比 FlashQLA 快 40%)。
- long_low: 3.21ms → 需 ~1.33ms (2.4x 加速)
- wide: 4.14ms → 需 ~1.73ms (2.4x)
- deep: 4.85ms → 需 ~2.02ms (2.4x)
这非常激进, 4-WG 单独可能不够 (预期 1.5-2x), 需配合数学变换。

### warp specialization 4-WG (最大潜力, 实现复杂度最高)
- 思路: 手写 4 warp 组（producer/S/V/O consumer），element-wise 藏进 GEMM 执行周期
- 基线 FlashQLA 即此结构（4 warp group + mbarrier + TMA + ping-pong）
- TileLang 无原语支持外层递推的 warp spec，必须手写 `tx = T.get_thread_binding()` + `if tx < N` 分支 + prologue/main/epilogue + mbarrier parity
- 参考: `examples/warp_specialize/example_warp_specialize_flashmla.py`
- 风险: 调试量大，shared 预算已紧
- 预期: TC 利用率从 13.9% 提升到 40-50%，长序列性能提升 2-3x

### state ping-pong shared
- 思路: state 不常驻 fragment，用两份 shared ping-pong，降寄存器
- 风险: shared 预算紧，需复用 buffer

### per-case autotune
- 思路: 用 `@autotune` 对 block_DV/threads/num_stages 搜参
- 当前固定 block_DV=64，可针对每个 case 形状搜最优

### TMA 替代 async_copy
- 当前用 `T.copy`（自动降级 TMA/cp.async）
- 手写 `T.tma_copy` + mbarrier 可能更高效（但需 warp spec 配合）

---

## 8. 基线 FlashQLA 结构情报 (供对标)

- **已是单 fused kernel**（`fused_gdr_fwd`），W/U/V_new/递推/O 全在一个 `@T.prim_func`
  → 融合优势不存在，必须 tile/调度/形式上赢
- **用 V_new 中间形式**（不是 WY 仿射）
- **完整 4-warp-group + mbarrier + 2 级 ping-pong** 跨 chunk
- **TMA** (`T.tma_copy`) + wgmma（隐式 via `T.gemm` + `T.use_swizzle(10)`）
- **block_DV 自适应 {128,64,32}**，阈值 `TARGET_NUM_CTAS = 0.7 × SM数`（MIG 14 SM 下会失准）
- **GVA 用 index 映射** `bhg = bh // (Hv//Hq)`，不展开
- **尾块**: Q/K/V/A/β 零补齐；g 补"最后一个有效值"（防 `γr/γ[r]` 比值形式 nan）
- **精度**: BF16 入 → FP32 累积/state → BF16 出；FP32 final_state；无 FP8；fast-math 开
- **`auto_cp=True`** 在长 case 触发 CP 预处理/修正 kernel（我们单 GPU 不需要）

### 超越基线的杠杆
1. **MIG-aware block_DV 重算**（基线阈值在 MIG 失准）— 已部分利用
2. **WY 仿射**（基线没用）— 实测在 MIG shared 限制下走不通
3. **case 专用优化**（cases.csv 已知）— 已用 per-case 分发

---

## 9. git 提交历史 (lab3 优化部分)

```
5d25f18 Lab3: exp 复用仅用于长序列物化W版, 短序列结合律版回退 inline exp
3b32bff Lab3: 预算 exp(g) 复用, 减少 element-wise exp2 指令
6346d2a Lab3: 回退长序列 block_DV=64 (32 令 TC 效率下降, 反而变慢)
e05b8e7 Lab3: 加 disable_warp_group_reg_alloc 降寄存器
ad5b59c Lab3: per-case 分发 (短序列结合律版 + 长序列物化W版)
3f86fd3 Lab3: 降寄存器压力优化 (结合律消 W + fragment 复用)
82499de (清理学习材料, 恢复朴素版)
... 早期: 朴素版 + dtype 修复 + initial_state 修复 + block_DV 调参
```

---

## 10. 关键文件路径

- **实现**: `lab3/student/tilelang_fwd.py`（唯一被评测收取）
- **正确性语义**: `lab3/references/torch_gdr.py`（FP64 ground truth）
- **preprocessing 范式**: `lab3/preprocessing/tilelang_kkt_solve.py`、`tilelang_cumsum.py`
- **评测**: `lab3/evaluation/run.py`、`support.py`、`cases.csv`
- **基线**: `lab3/references/official/flash_qla.py`（wrapper）
- **外部示例**: tilelang repo `examples/gdn/*`、`examples/warp_specialize/*`
- **设计文档**: `docs/superpowers/specs/2026-08-02-gdn-prefill-pipeline-design.md`
