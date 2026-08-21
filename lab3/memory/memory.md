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

★ FlashQLA 参考时间用作业页面 t100 (2026-08-17 从 hpc101.zjusct.io 确认):
  short_tail=0.346, chain=0.498, parallel_equal=0.511, parallel_gva=0.492,
  long_low=1.859, batch_split=1.532, wide=2.427, deep=2.831

★ FLA / FlashInfer 实测时间 (2026-08-21, --reference-benchmarks full_forward):
  | case | FlashQLA | FLA | FlashInfer |
  |------|----------|-----|------------|
  | short_tail_state | 0.270 | 0.459 | 0.304 |
  | chain_equal | 0.464 | 0.951 | 0.793 |
  | parallel_equal | 0.462 | 0.945 | 0.473 |
  | parallel_gva | 0.440 | 0.843 | 0.447 |
  | long_low_gva | 1.829 | 6.447 | 1.804 |
  | batch_split_gva | 1.812 | 6.161 | 1.445 |
  | wide_gva_state | 3.346 | 12.555 | 2.405 |
  | deep_gva_state | 3.617 | 12.823 | 2.658 |
  ★ FLA 在所有 case 都是最慢的 (长序列慢 3-4x FlashQLA).
  ★ 作业页面 t60 (60分 checkpoint) = 0.574/4.000/2.017/... 与 FLA 时间不符 (FLA chain 0.951 vs t60 4.000).
    故 t60 不是 FLA 时间, 是另一个 checkpoint (可能是某个固定分数对应的实现, 非 FLA).
  ★ FlashInfer 长序列最快 (long_low 1.804 vs FlashQLA 1.829), 短序列 FLA 最慢.

★ num_stages=2 微优化 (2026-08-19): 大 grid 长序列 matw 从 st=1 改 st=2,
  T.Pipelined 自动 multi-version shared buffer, 跨 chunk load 重叠:
  | case | st=1 (旧) | st=2 (新) | t100 | p (新) |
  |------|------|------|------|------|
  | long_low | 3.16 | 3.04 | 1.859 | 0.61 |
  | wide | 4.14 | 3.77 | 2.427 | 0.64 |
  | deep | 4.85 | 4.46 | 2.831 | 0.63 |
  | batch_split | 2.40 | 2.26 | 1.532 | 0.68 |
  | short_tail | 0.146 | 0.150 | 0.346 | 2.31 |
  | chain | 0.532 | 0.537 | 0.498 | 0.93 |
  长序列全线提升 ~6-10%, 短序列持平/微退 (chunk 少, pipeline 开销 > 收益).
  chain/short_tail 保持原配置 (chain DV=64 st=2, short_tail 结合律 st=2).

| case | student(ms) | t100(ms) | p=t100/t | 估算分 |
|------|------------|----------|----------|--------|
| short_tail_state | 0.150 | 0.346 | 2.31 | 120(封顶) |
| parallel_equal | 0.45 | 0.511 | 1.14 | ~103 |
| parallel_gva | 0.43 | 0.492 | 1.14 | ~103 |
| chain_equal | 0.537 | 0.498 | 0.93 | ~96 |
| long_low_gva | 3.04 | 1.859 | 0.61 | ~82 |
| batch_split_gva | 2.26 | 1.532 | 0.68 | ~85 |
| wide_gva_state | 3.77 | 2.427 | 0.64 | ~83 |
| deep_gva_state | 4.46 | 2.831 | 0.63 | ~83 |

**3 个 case 超过 FlashQLA**（short_tail 2.37x, parallel_equal 1.14x, parallel_gva 1.14x）
★ 评分公式 p=t100/t, p>1 进 100-120 奖励区. 长序列 4 case p≈0.58-0.64(~81分) 是提分瓶颈.
★ 冲 110 需长序列 p≈1.0 (追平 FlashQLA), 即 long_low 3.16→1.86ms (1.7x加速). 4-WG 是唯一路径但不可行.

### 优化进展 (v1 → 最终)
- short_tail: 0.195 → 0.146（快 25%）
- long_low: 4.125 → 3.21（快 22%）
- wide_gva: 5.439 → 4.14（快 24%）
- deep_gva: 5.157 → 4.85（快 6%）
- chain_equal: 0.561 → 0.53（快 5%）

---

## 5. ncu 分析数据 (long_low_gva, 物化W版+exp复用+autotune DV=128)

| 指标 | v1 朴素 | v8 (DV=64) | 最终 (DV=128,th=256) | DV=64 th=128 | DV=64 th=256 | set_max_nreg(160) | 目标 |
|------|---------|--------|--------|--------|--------|--------|------|
| reg/thread | 255 | 220 | **250** | 220 | 186 | 160 | <128 (让 2 block/SM) |
| TC 利用率 | 13.8% | 13.9% | **15.6%** | 13.9% | 13.0% | 11.5% | >50% |
| waves/SM | 1.14 | 1.14 | **0.57** | 1.14 | 1.14 | 0.57 | >2 |
| shared (dynamic) | — | — | 222976 B | — | — | — | <232KB |

**★ ncu 完整分析 (2026-08-21)**:
- long_low DV=128: reg=250, waves=0.57 (grid=8 < 14 SM, 不足 1 波), TC=15.6%.
  ★ TC 15.6% = 真正瓶颈: 84% 时间 TC idle, 等 ew/copy/stall.
  ★ waves=0.57 不变是 grid=8 < 14 SM 导致, 非 reg 限制 (set_max_nreg(160) reg 降但 waves 不变, TC 反降).
  ★ DV=64 翻 grid=16 (waves 1.14) 但 TC 降 (13.9% vs 15.6%), 净退步 (long_low 3.04→4.00ms).
  ★ async wgmma 无收益 (单 WG issue 串行, wait 增开销).
  ★ set_max_nreg 无收益 (grid 限制非 reg 限制, spill 反降 TC).
  ★ RS GEMM (fragment A) 不可行 (layout infer conflict: FP32→BF16 fragment cast 在 T.Parallel 内 layout 不匹配).
- 核心瓶颈: TC 15.6% idle 84%, ew/copy stall 主导. 单 WG 无法藏 ew (需 4-WG, 但不可行).

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

### 4-WG 第二轮 (FlashMLA 模板照搬, **仍未通**)
- 子 agent 读 FlashMLA 完整源码 (github.com/tile-ai/tilelang examples/warp_specialize/)
  - ★ FlashMLA 实为 **2-WG** (threads=256, tx<128/else), 非 4-WG! tilelang 仓库无 4 分支范本
  - 唯一 `T.fence_proxy_async` 在 line 231: `T.copy(frag, shared) → fence → barrier_arrive`
  - 注释明证: "InjectFenceProxy cannot infer from warp specialization code that this is a RAW dependency"
  - 8 处 `T.wait_wgmma(0)` 在每个 wgmma 组后读 accumulator 前
  - gemm_barrierpipe (2-WG 严格 P/C) 的 barrier Pattern A: `[128,128]*num_stages`, parity `(i//ns)%ns`
- 照搬模板重写 4-WG (朴素 V_new, num_stages=2):
  - 跨 WG 写 (h_shared/vn_shared) 后都 `fence → arrive` (照搬 FlashMLA line 231)
  - 每个 `T.gemm` 后 `T.wait_wgmma(0)` 再读 accumulator (照搬 FlashMLA 8 处)
  - barrier `[128,384,128,128]*2`, parity `(i//2)%2`
- **实测**: chain_equal 仍 FAIL — 69.1% mismatch (abs 0.10), 加 intra-WG bkg/bv fence 后变 79% nan
- **★ 二分诊断** (S+V 空转, 只验证 producer→S→V 同步链): **仍 nan**
  - 即使 S 只预算 gate + 发布 h (不算 S_new), V 只算 W (不算 V_new), O/V 全空转 → nan
  - 说明 nan 不来自计算, 来自 **最基础的 producer→consumer data_ready 同步** (element-wise load + fence + arrive → consumer wait + 读)
- 回退到 466c000 基线

### 4-WG 第三轮 (T.ws 调研反转 + FlashQLA 配置 + WY-O, **仍未通**)- **子 agent 源码调研 (关键反转)**:
  - `T.ws(N)` 语法不限 N，但 `producer_consumer_ws.cc` **只支持 2 分区**! 4 分区代码生成零验证
  - ★ FlashQLA hopper 版是 **TileLang** (非 CUDA), 4-WG threads=512, 用**手写 tx if/elif 4 分支**(和前两轮一样)
  - tilelang 作者写 4-WG 时都绕过 T.ws 用手写 tx → 手写 tx 本身不是 bug 根源
  - FlashQLA 完整配置: `data_ready` arrive_count=**96**(3 producer sub-warp×32), `data_free`=384, per-WG `set_max_nreg`(S=160/V=128/O=128/P=32), parity=`i_s%2`(内部) vs `(i_s//2)%2`(外部 buffer), **无 wait_wgmma**(靠 barrier 隐式 commit), 无 prologue
  - 我 Round 2 差异: arrive_count=128(错), 无 set_max_nreg(关键漏), 每处加 wait_wgmma(反 nan), parity 统一(错)
- **第三轮 a (WY-O + Q' 复用 K_shared_2)**: chunk0 **PASS(0.0003)**, chunk1+ 全错(~0.05, 恒定不累积)
  - 数学 WY-O: `O=scale*((γ⊙Q-ds@W)@S_old + ds@U)`, 消除 O→V_new 跨 WG 依赖
  - 诊断: Q'=γ⊙Q-ds@W 写到 K_shared_2[buf] 覆盖 K, 但 CONSUMER_S 需 K 算 Kᵀ@V_new → 竞争
  - chunk0 对因 S_old=0 特殊性 (V_new=U−W@0=U, Kᵀ@V_new 错误被掩盖)
- **第三轮 b (原始 O 形式, 不覆盖 K)**: kernel 调用本身抛异常(卡在 gdn_prefill_forward 内部, sync 前挂)
  - 根因: 原始 O 依赖 V_new(等 vn_ready), 与 4-WG 并行目标冲突 → barrier 环形依赖死锁
  - ★ 证明 WY-O 是必须的(消 O→V 依赖), 不能回退原始 O
- **第三轮 c (WY-O + Qp 独立 buffer, num_stages=2)**: nan(99.5%) — 比第三轮 a 退步
- **第三轮 c (num_stages=1, phase=i%2)**: 仍 nan(89.2%)
  - 独立 Qp 反而 nan, 说明复用 K 不是根因 → chunk1+ 错来自别处
- **★ 第三轮最终根因推测**: chunk0 PASS 证明结构和数学正确, chunk1+ 恒定不累积错 → **跨 chunk state 传递竞争**
  - s_fragment 单份跨 chunk 更新; h_shared 双 buffer 但 S 在 vn_ready 后更新, 下一 chunk 发布 h_shared[buf] 用更新后 S, 而 V/O 下一 chunk 在 buf 翻转后读 h_shared[buf] 可能读到未更新值
  - 4-WG 下 3 consumer 的 wait/arrive 交错比 2-WG 复杂得多, barrier 时序在跨 chunk 边界有无法定位的竞争
- **结论**: 4-WG 经过三轮(a/b/c)、多变体(num_stages 1/2、phase 公式、buffer 复用/独立、WY-O/原始O), **chunk0 PASS 但 chunk1+ 精度始终无法解决**. tilelang 仓库无 4-WG 范本(FlashQLA 是唯一 4-WG 但其 barrier 配置极复杂: 13 个 barrier + 4 producer sub-warp, 远超我能手写调通的复杂度). 4-WG 在当前 TileLang + MIG + 远程调试(5min walltime)下不可行.
- 回退到 466c000 (OJ 93 分) 基线, 4-WG 方向最终暂停

### 4-WG 第四轮: WY-O + async + GEMM 融合 (单 WG, **精度全过性能全退, 证伪**)
- ★ 子agent调研关键反转: tilelang 有官方 async API (T.wgmma_gemm + T.wait_wgmma, gemm_op.py:142 docstring +
  wgmma_macro_generator.py:340). 默认 T.gemm 隐式 wait(0) 阻塞; T.wgmma_gemm 不 emit wait.
  flashmla example (example_warp_specialize_flashmla.py:160) 是唯一 async 先例 (重叠 GEMM↔TMA load, 非GEMM↔ew).
  T.Pipelined 只重叠 load↔compute (pipeline_planning.cc:602 ClassifyCopyLikeStage 只认 CopyNode).
  → 此前"WY-O 单 WG 慢需 4-WG 兑现"是基于同步 T.gemm 的误判, 重测三路:
- **4a (async, T.wgmma_gemm+wait_wgmma(0))**: 8 case 全 PASS, 但慢 40-80%
  - chain 0.954 (vs 0.53), long_low 4.529 (vs 3.16), wide 5.974, deep 6.894, batch 3.375
  - 根因: HW OoO 重叠有限 + fragment 复用 copy (U 驻留需 shared 中转) + 寄存器压力
- **4b (同步对照, T.gemm)**: 慢 2.3x (chain 1.230, long_low 4.604)
  - 证明 async 给了微小但不够的收益; WY-O 的 +1 GEMM (ds@W) 串行开销仍主导
- **4c (F1 融合, A@[βγK|βV]→[W|U], N=256)**: 慢 2.6x (chain 1.363, long_low 5.468)
  - ★ T.gemm M=128/N=256 codegen 支持 (子agent1确认 gemm.cc:96 op.m_>=64, wgmma.h:220 N枚举到256)
  - ★ fragment 偏移索引 (frag[t,DK+d]) 零先例 → 改 shared 切片中转 (T.copy frag→bkbv_shared 再切片读) 成功
  - 根因: 拼接 ew (block_S×(DK+DV) 写) + frag→shared 中转 copy > 省 1 GEMM; N=256 大 tile TC 效率未兑现
- **4d (F2/F3 未测)**: F1 已证净负, F2 fragment [128,DV] 寄存器更紧, F3 需重写时序, 趋势单调下降, 停测
- ★ 数学修正: WY-O 的 O = scale*(Qp@S + ds@U), Qp=γ⊙Q-ds@W 已含 -ds@W, **不应再减 ds@W@S**
  (初版多减一次 → 0.049 误差, 修正后全 PASS). 见 [[4wg-wyo-math]].
- **结论**: WY-O 三路 (async/同步/融合) 在单 WG 下性能全部证伪. 根因统一:
  (1) tilelang 无 GEMM↔ew 重叠机制 (T.Pipelined 只重叠 load↔compute, HW OoO 不可控)
  (2) fragment 复用 + U 驻留拉长生命周期, 寄存器压力激增
  (3) 融合的拼接/copy 开销 > 省 GEMM 收益
  (4) async wait(0) 用太狠变同步, wait(N) 保留多 GEMM 在飞会 fragment 冲突
- 回退到 466c000 (OJ 93 分) 基线, WY-O 方向最终暂停

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

### 2-WG producer 做 state-free GEMM (**PASS 但慢, 无收益**)
- 思路: producer(ws1) 做 W=A@βγK, U=A@βV, ds=QKᵀ⊙gate (state-free GEMM), consumer(ws0) 做 state GEMM
- threads=256 精度好 (不触发 wgmma), 全 PASS
- **单 buffer 版** (DV=128): chain 1.32ms, long_low 5.22ms (比 matw 0.53/3.21 慢 1.6-2x)
  - 根因: W/U/ds 单 buffer, producer/consumer 同 chunk 串行, 无重叠
- **双 buffer 版** (DV=64, W/U/ds 双 buffer 真正重叠): chain 0.89ms, long_low 6.66ms
  - 比 matw 仍慢: DV=64 TC 效率降 + warp-spec barrier 开销 > 重叠收益
  - long_low 更慢 (512 chunks × barrier 开销累积)
- **结论**: 2-WG GEMM 在 MIG 上性能不升反降, DV=64 TC 效率损失是主因
- **回退**: 代码回退到 f7b5cdb 基线

### TL_DISABLE_WGMMA pass_config (**无效**)
- `tilelang.PassConfigKey.TL_DISABLE_WGMMA = "tl.disable_wgmma"` 存在
- 4-WG threads=512 + TL_DISABLE_WGMMA=True: 仍 2.7% 精度差 (abs 0.069 不变)
- 说明误差不是 wgmma vs mma 问题, 是多 WG 同步可见性
- matw+T.serial 单 kernel PASS 证明: T.serial 本身不引入误差

### TMA load 实验 (单 WG, T.copy 替代 T.Parallel, **精度全过性能仍退**)
- ★ 动机: 子agent2 调研 T.Pipelined 的 ClassifyCopyLikeStage (pipeline_planning.cc:602) 只认
  T.copy(global→shared) 为 producer, T.Parallel ew load 进 consumer 和 GEMM 串行. 把 6 个 load 换 T.copy
  期望 load↔compute 重叠. 子agent3 确认 state 用单 shared (chunk_delta_h.py 标准模式, 不需 ping-pong).
- ★ 实测 (GDN_TMA=1, ns=1, 同 matw 数学):
  | case | matw (93基线) | TMA ns=1 | 变化 |
  |------|------|------|------|
  | chain_equal (DV=64,th=128,ns=2) | 0.539 | 0.594 | +10% |
  | long_low_gva (DV=128,th=256,ns=1) | 3.16 | 4.008 | +27% |
  | wide_gva_state (DV=128) | 4.14 | 5.113 | +23% |
  | deep_gva_state (DV=128) | 4.85 | 6.077 | +25% |
  | batch_split_gva (DV=128) | 2.40 | 3.013 | +25% |
  | parallel_equal (DV=64) | 0.45 | 0.469 | +4% |
  | parallel_gva (DV=64) | 0.43 | 0.476 | +11% |
- ★ 编译警告暴露根因:
  - "src range must have last dim multiple of 16 for tma bulk load g_cumsum range 1*4 % 16 != 0"
    g/beta 是 [block_S] 1D FP32, TMA bulk load 要求末维 16 倍数, block_S=64 FP32=256B 满足但 range shape "1*4"
    (B*Hq?) 不满足 → **g/beta 的 T.copy fallback 到 normal copy (element-wise)**, 没走 TMA!
  - "src and dst must have the same dtype for tma load initial_state vs s_shared dtype float32 vs bfloat16
    will be fallback to normal copy" → initial_state T.copy 也 fallback.
  - 即只有 Q/K/V/A 的 T.copy 真走 TMA, g/beta/initial_state 仍 element-wise.
- ★ ns=2 (显式 _2 ping-pong buffer): shared 273KB > 232KB 超限. 因 T.Pipelined 已对 T.copy producer
  自动 multi-version (MultiVersionBufferRewriter), 显式 _2 维 + 自动 multi-version 双重计算爆 shared.
  正确做法是单 buffer + ns=2 让 T.Pipelined 自动管 (但 DV=128 th=256 寄存器紧, ns=2 可能仍超).
- **结论**: TMA bulk load 在 GDN 上未兑现 load↔compute 重叠. 原因推测:
  (1) g/beta fallback 到 element-wise, 这两个 ew 仍在 consumer 串行
  (2) T.Pipelined 的自动 multi-version 对 6 个 producer T.copy 生成复杂 pipeline, 可能与 consumer 的
      state 递推依赖 (s_fragment 跨 chunk) 冲突, 实际重叠有限
  (3) H800 MIG 14 SM, load 本身不是瓶颈 (L2 cache 命中率高), GEMM/ew 才是
- **回退**: GDN_TMA 默认关, 走 matw 保 93 分基线. 代码保留 _gdn_tma_kernel 供报告引用.

### 三 kernel 分解 (K1 并行 W/U/ds + K2 串行 S + K3 并行 O, **精度全过性能全退**)
- ★ 数学验证 (torch FP64): state ref vs serial-scan diff 2.78e-16, O decomp vs O_ref 0.00e+00.
  M[c]=γr·I−sKᵀ@W, b[c]=sKᵀ@U, S[c]=M[c]@S[c-1]+b[c]. W/U/ds 三 GEMM state-free.
  7 GEMM 里只有 W@S→V_new→Kᵀ@V_new→S_new (2 GEMM) 真串行.
- ★ 动机: 单 kernel grid=B×Hv (chain=4, long_low=8) 严重欠占 14 SM. 拆成 K1/K3 grid=chunks×B×Hv
  (long_low 512×8=4096 blocks) 填满 SM. state 精确串行 (无 scan 近似, 误差不叠加).
- ★ 实测 (GDN_DECOMP=1, DV 整块不切, threads 自适应):
  | case | matw (93基线) | decomp | 变化 |
  |------|------|------|------|
  | chain_equal (th=128) | 0.539 | 1.211 | +125% |
  | long_low_gva (th=256) | 3.16 | 6.584 | +108% |
  | wide_gva_state (th=256) | 4.14 | 12.077 | +192% |
  | deep_gva_state (th=256) | 4.85 | 12.495 | +158% |
  | batch_split_gva (th=256) | 2.40 | 6.162 | +157% |
  | parallel_equal (th=256) | 0.45 | 0.999 | +122% |
  | parallel_gva (th=256) | 0.43 | 0.915 | +113% |
- ★ 退步根因 (统一):
  (1) **K2 串行 kernel 瓶颈**: K2 grid=B×Hv (long_low=8), chunk 串行 T.serial.
      单 kernel 里 chunk 串行但有 T.Pipelined 重叠 load; K2 用 T.serial 无重叠, 每 chunk
      6 次 global→shared T.copy (K/W/U/g + S存/读) 全串行. long_low 512 chunk × 6 copy 累积.
  (2) **global 读写开销**: K2 读 W/U/K/g (40KB/chunk), 写 S_all (64KB/chunk); K3 读 S_all+W/U/ds.
      long_low 共 ~50MB 额外 global 读写, 无 L2 命中 (S_all 首次写, K3 首次读).
  (3) **DV 整块不切**: s_shared [128,128] BF16 = 32KB, s_fragment [128,128] FP32 = 64KB.
      K2 threads=256 下 s_fragment 64KB 寄存器 → spill. 无法像 matw 那样 DV=64 降一半.
  (4) **3 kernel launch + JIT**: 3 个 kernel 各编译一次, OJ walltime 紧张.
  (5) **占用率提升没兑现**: K1/K3 grid 虽大, 但每 block 只 3/2 GEMM (比 matw 7 少),
      per-block 工作量太轻, launch 开销 + global 往返吃掉占用率收益.
- ★ 与 matw 的根本差异: matw 单 kernel 把 W/U 复用 (tmp_dv 临时, 算完即丢), state 驻 fragment.
  decomp 把 W/U 物化到 global (K1→K2→K3 传递), 每 chunk 多 3 次 [64,128] 写 + 3 次读.
  物化开销 >> 占用率收益. 这是 "拆 kernel" 的固有代价.
- **结论**: 三 kernel 分解在 MIG 上性能全退 (1.1-1.9x 慢). 数学正确但工程不可行:
  K2 串行 kernel 无 T.Pipelined 重叠 + global 物化 W/U/S 开销 + DV 整块寄存器压力.
  占用率提升 (chain 4→512 blocks) 被 per-block 工作量太轻 + global 往返吃光.
- **回退**: GDN_DECOMP 默认关, 走 matw 保 93 分基线. 代码保留 K1/K2/K3 供报告引用.
- ★ 数学验证脚本 (/tmp/test_scan_math.py): 证实 M[c]/b[c] scan + 三 kernel 分解数学零误差,
  为实验报告提供"精确分解数学正确但 MIG 工程不可行"的完整论证.

## 11. 优化方向穷尽总结 (截至 2026-08-19)

### 已验证不可行
| 方向 | 结果 | 根因 |
|------|------|------|
| 4-WG warp spec (T.ws) | 死锁 | T.ws(3) 不支持, producer_consumer_ws.cc 只支持 2 分区 |
| 4-WG tx 手写 4 分支 (R1) | 2.7% 精度 | 多 WG 同步可见性 (wgmma async proxy) |
| 4-WG + FlashMLA 模板照搬 (R2, fence+wait_wgmma) | 69%→nan | wait_wgmma 破坏 barrier 时序契约; 无 set_max_nreg |
| 4-WG + FlashQLA 配置 (R3a, WY-O+Q'复用K) | chunk0 对, chunk1+ 错(0.05) | Q'覆盖 K, S 读错 Kᵀ@V_new; 跨 chunk state 竞争 |
| 4-WG + 原始 O 形式 (R3b) | kernel 调用异常 | O 依赖 V_new → barrier 环形依赖死锁 (WY-O 是必须的) |
| 4-WG + WY-O+Qp 独立 buffer (R3c, ns=1/2) | nan(89-99%) | 独立 Qp 反而 nan, 跨 chunk state 传递竞争未解 |
| 4-WG + TL_DISABLE_WGMMA | 2.7% 不变 | 非 wgmma 问题 |
| threads=512 matw | wide/deep FAIL | 精度回归 |
| per-case 调参 | 无提升 | DV=128/th=256/st=1 已最优 |
| split-DV (DV=64) | 全线更慢 | TC 效率损失 > grid 收益 |
| 2-WG producer GEMM (单 buffer) | 慢 1.6-2x | 无重叠 (同 chunk 串行) |
| 2-WG producer GEMM (双 buffer DV=64) | 慢 2x | DV=64 TC 降 + barrier 开销 |
| chunk 间 state 并行 | 死路 | 需 128×128 矩阵 scan, shared 放不下 |
| @autotune 提交 OJ | 6 case 超时 0 分 | JIT 搜参超 5min walltime |
| WY-O 单 WG | 慢 (chain 0.72 vs 0.53) | +1 GEMM 串行 > 消 ew 收益, 需 4-WG 兑现 (但 4-WG 不可行) |
| 完整 WY (物化 T) | shared 超 232KB | T[128,128] 无法放下 |
| **WY-O + async wgmma (单 WG)** | **全 PASS 但慢 40-80%** (chain 0.954, long_low 4.529) | T.wgmma_gemm async issue 但 HW OoO 重叠有限; fragment 复用 copy + 寄存器压力反收益 |
| **WY-O 同步对照 (单 WG)** | **慢 2.3x** (chain 1.230, long_low 4.604) | 无 async 收益, 纯 +1 GEMM 串行, 证明 async 给了微小但不够的收益 |
| **WY-O + GEMM 融合 F1 (A@[βγK\|βV]→[W\|U], N=256)** | **慢 2.6x** (chain 1.363, long_low 5.468) | 拼接 ew + frag→shared 中转 copy 开销 > 省 1 GEMM; N=256 大 tile 占用未兑现 |
| WY-O + F2/F3 融合 | 未测 (F1 已证净负) | F2 fragment [128,DV] 寄存器更紧; F3 需重写时序; 趋势单调下降无逆转迹象 |
| **TMA load (T.copy 替代 T.Parallel, 单 WG, ns=1)** | **全 PASS, 长序列慢 ~20-27%** | 见下表; TMA bulk load 未兑现 load↔compute 重叠 |
| **TMA load ns=2 (显式 ping-pong _2 buffer)** | shared 超 232KB (273KB) | T.Pipelined 已自动 multi-version, 显式 _2 双重计算爆 shared |
| **三 kernel 分解 (K1 并行 W/U/ds + K2 串行 S + K3 并行 O)** | **全 PASS 但慢 1.1-1.9x** (chain 1.211, long_low 6.584, wide 12.077) | K2 串行无 T.Pipelined 重叠; W/U/S global 物化往返; DV 整块寄存器压力; 占用率收益被 per-block 工作量轻吃光 |
| **4-WG 第五轮 照搬 FlashQLA hopper** | 死锁 (kernel 挂起, 5min walltime timeout) | 13 barrier handshake 极复杂; 多 WG arrive_count (96/384/416/256/128) 时序对不齐; 子agent确认 T.barrier 是 T.mbarrier 糖不自动 fence, 但 FlashQLA 靠 T.tma_copy(barrier=) 自动 arrive 避免显式 fence, 我的照搬仍有时序错配 |
| **Qhat/Khat 换元 (单 WG, 精确零近似)** | **精度全过但性能退步** (chain 0.729 vs 0.53, long_low 3.499 vs 3.04) | 换元 Qhat=γ·Q/Khat=γ_inv·K 消 ds gate + O γ 两处 ew stall, 但 Q/K load 改 T.Parallel element-wise (无法 T.copy 进 pipeline), Khat_shared 新增 16KB + bkg 多读 K, pipeline 退化. 与 TMA load 实验同类失败: element-wise load 在长序列 chunk 多时累积开销大. tilelang 下 load 必须走 T.copy 才进 T.Pipelined |
| **Prefix-state block-constant output (单 WG, 近似)** | **精度 FAIL, 近似不可用** (block=16 52% mismatch abs 0.10; block=32 62%; block=64 68%) | 数学 O[i]≈scale·γ[i]·Q[i]@(S_old+P[block_start]) 消 Q@Kᵀ+ds@V_new 两 GEMM, state 精确. 但 block-constant 忽略 chunk 内 j<i 的因果 prefix 贡献 (output_in_chunk 项被丢), 误差远超 RTOL=5e-3. torch FP32 验证: block=16 rel>5e-3 frac=97%, abs max 9.4; block=32 rel 98%, abs 18; block=64 rel 99%, abs 18. 近似在 GDN 上不可用 (in-chunk 因果项贡献大, 不能用 block-constant 忽略) |

### 4-WG 三轮失败的根本原因 (最终结论)
1. **tilelang 无 4-WG 官方范本**: FlashQLA hopper 是唯一 4-WG (手写 tx), 但其 barrier 极复杂 (13 barrier + 4 producer sub-warp), 远超手写调通能力
2. **chunk0 PASS 但 chunk1+ 错**: 证明单 chunk 结构/数学/fence/nreg 全对, bug 在跨 chunk 边界的 state/barrier 时序竞争
3. **远程 5min walltime 限制**: 无法用 ncu/cuda-gdb 逐指令定位跨 chunk 竞争, 只能靠二分猜测
4. **WY-O 是必须的但不够**: 消除 O→V 依赖避免死锁, 但跨 chunk state 传递仍有竞争

### 当前最优 (466c000, OJ 93 分)
- 短序列 (T<=2048): 结合律版 DV=64/th=128/st=2, 3 case 超 FlashQLA (120/102/101 分)
- 长序列 (T>2048) per-case 分发:
  - 小 grid (B*Hv<=4, 即 chain_equal/hidden-2): 物化W版 DV=64/th=128/st=2, grid 翻倍提 SM 占用
    - chain_equal 0.53ms (95分), 比 DV=128 的 0.84ms(74分) 快 37%
  - 大 grid (B*Hv>4): 物化W版 DV=128/th=256/st=1, 大 tile 最优
    - long_low 3.16ms(74分), wide 4.14ms(74分), deep 4.85ms(74分), batch_split 2.40ms(76分)
- **OJ 93 分**: 公开 89.4 + 隐藏 98.0, 12 case 全 PASS

### per-case 分发调参细节 (chain_equal 专用)
- chain_equal (B=1, Hv=4, T=8192): grid=4 (DV=128) 或 8 (DV=64)
- 实测对比:
  - matw DV=64/th=128/st=2 = 0.527ms (最优, OJ 95分)
  - matw DV=64/th=256/st=1 = 0.652ms (慢, th=256 对 chain 无益)
  - 结合律 DV=64/th=128/st=2 = 0.584ms (慢, matw 少一次 GEMM 更快)
  - matw DV=128/th=256/st=1 = 0.84ms (慢, grid=4 SM 占用不足)
- chain t100=0.498ms, 当前 0.53ms 差 6%, 已接近极限

### OJ 分数演变
| 版本 | OJ 分 | 关键变化 |
|------|-------|---------|
| 5d25f18 (DV=64 全局) | 92 | 长序列全 DV=64, chain 快但 long_low 慢(4.06ms) |
| 95ba1d4 (DV=128 全局) | 89 | 长序列 DV=128, long_low 快(3.18)但 chain 慢(0.84ms) |
| 466c000 (per-case) | 93 | chain DV=64 + 长序列 DV=128, 取两者最优 |

---

## 7. 下一步优化方向 (截至 2026-08-20, Qhat/Khat + Prefix-state 两路证伪后)

### 目标: OJ 110 分 (★ 门槛是 p≈1.0)
评分公式 p=t100/t. 110 分需长序列 p≈1.0 (追平 FlashQLA).
- long_low: 3.04ms → 需 ~1.86ms (1.6x), wide 3.77→2.43, deep 4.46→2.83, batch 2.26→1.53

### ★ Qhat/Khat 换元证伪 (2026-08-20, 精确零近似但性能退步)
更聪明AI建议第一优先级: 换元 Qhat=γ·Q, Khat=γ_inv·K 消 ds gate (γ_i/γ_j) 与 O γ 两处 ew stall.
- 数学验证正确 (torch FP32 等价). ds=tril(Qhat@Khatᵀ), O=scale·(Qhat@S_old+ds@V_new), S_new=γr·S_old+Khatᵀ@(γr·V_new).
- 集群实测 (GDN_QHAT=1): chain_equal 0.729ms (vs matw 0.53, +37%), long_low 3.499ms (vs 3.04, +15%). 精度全过.
- ★ 退步根因: Q/K load 改 T.Parallel element-wise (Qhat/Khat load 时缩放, 无法 T.copy 走 TMA),
  T.Pipelined 的 ClassifyCopyLikeStage 只认 T.copy(CopyNode), ew load 从 producer 变 consumer 串行 → pipeline 退化.
  Khat_shared 新增 16KB + bkg 多读一次 K_shared. 与 TMA load 实验 (长序列 +23%) 同类失败模式.
- ★ 核心教训: **tilelang 下 load 必须走 T.copy 才进 T.Pipelined pipeline**. 任何把 load 改 element-wise 的换元都会退步.
  消 ew stall 的收益 < load 退出 pipeline 的代价, 尤其长序列 chunk 多时累积.

### ★ Prefix-state block-constant output 证伪 (2026-08-20, 近似不可用)
更聪明AI建议第一优先级: O[i]≈scale·γ[i]·Q[i]@(S_old+P[block_start]) 消 Q@Kᵀ + ds@V_new 两 GEMM, state 精确.
- 集群实测 (GDN_PREFIX=1, block=16): chain_equal 52% mismatch abs 0.10. block=32/64 更差.
- ★ torch FP32 验证 (block_prefix=16/32/64, state 精确零误差):
  - block=16: abs max 9.4, rel>5e-3 frac 97%
  - block=32: abs max 18.0, rel>5e-3 frac 98%
  - block=64: abs max 18.1, rel>5e-3 frac 99%
- ★ 根因: block-constant 忽略 chunk 内 j<i 的因果 prefix 贡献 (output_in_chunk = scale·(scores·decay)@v_new 项被丢).
  GDN 的 in-chunk 因果项贡献大 (decay=tril(exp(g_i-g_j)) 对角线=1, 近对角项不衰减), 不能用 block-constant 忽略.
  与 attention 的因果 mask 不同: GDN decay 是 exp 差, block 内近对角 token 贡献接近 1, 近似误差大.
- ★ tilelang 限制: T.gemm 不支持 shared 操作数带首维 offset 切片 ("offset of first dim must be 0"),
  需用独立 sub-shared buffer (Q_sub/K_sub/Vn_sub) 存子块, 增 shared 开销.
- ★ 结论: **block-constant 近似在 GDN 上不可用**. 需 local window 修正 (block-prefix + 最近 L token 精确),
  但 local window 需带宽 GEMM (tril 局部 scores), 又把消掉的 GEMM 加回来, 收益存疑.

### ★ WY-O 三路全部证伪 (2026-08-18, 单 WG 精度全过但性能全退)
子agent调研发现 tilelang 有官方 async API (T.wgmma_gemm+T.wait_wgmma, flashmla 先例),
此前"WY-O 单 WG 慢需 4-WG 兑现"的结论是基于默认同步 T.gemm 的误判. 重测三路:
- **async (T.wgmma_gemm+wait_wgmma(0))**: 8 case 全 PASS, 但全线慢 40-80%
  (chain 0.954 vs 0.53, long_low 4.529 vs 3.16, wide 5.974 vs 4.14)
- **同步对照 (T.gemm)**: 慢 2.3x (chain 1.230, long_low 4.604), 证明 async 给了微小但不够的收益
- **F1 融合 (A@[βγK|βV]→[W|U], N=256)**: 慢 2.6x (chain 1.363, long_low 5.468)
  ★ 关键发现: T.gemm M=128/N=256 codegen 支持 (子agent1确认 gemm.cc:96/wgmma.h:220),
    fragment 偏移索引 (frag[t,DK+d]) 零先例 → 改用 shared 切片中转 (bkbv_shared 复用) 成功编译运行.
    但拼接 ew + frag→shared 中转 copy 开销 > 省 1 GEMM, N=256 大 tile 的 TC 效率未兑现.
- **F2/F3 未测**: F1 已证净负, F2 (M=128 fragment [128,DV]) 寄存器更紧, F3 需重写时序,
  趋势单调下降无逆转迹象, 停测.

### 三路失败的统一根因 (★ 核心教训)
1. **tilelang 无 GEMM↔ew 重叠机制**: T.Pipelined 只重叠 load↔compute (pipeline_planning.cc:602 ClassifyCopyLikeStage
   只认 CopyNode). 单 WG 内 GEMM↔ew 重叠靠 HW OoO, 不可控 (flashmla 重叠的是 GEMM↔TMA load, 不是 GEMM↔ew).
2. **fragment 复用成本**: WY-O 需 U 驻留 (不覆写), 多一个 [64,DV] shared + fragment 生命周期拉长,
   threads=256 下寄存器压力激增 → spill 或占用降.
3. **融合的 copy 开销**: 拼接操作数需 ew 填 shared (block_S×(DK+DV) 次写), 拆结果需 frag→shared→切片读,
   这些 copy 把省下的 1 GEMM 时间吃光还倒贴. 大 tile 的 TC amortize 收益抵不过 copy.
4. **async 的 wait(0) 用太狠**: 每 Phase 末 wait(0) 等全部完成, 等于又变同步. 真正藏 ew 需 wait(N) 保留 N 在飞,
   但 fragment 复用让多 GEMM 在飞会冲突 (同一 fragment 被多 GEMM 写).
5. **(新增) load 退出 pipeline 的代价**: 任何把 T.copy load 改 element-wise 的换元 (Qhat/Khat) 都会让 load 退出
   T.Pipelined pipeline, 长 chunk 累积开销 > 消 ew 收益. tilelang 下 load 必须保持 T.copy.

### ★ 4-WG 已穷尽 (三轮失败, 详见第 6 节 + 第 11 节)
- R1: 手写 tx + matw → 2.7%
- R2: FlashMLA 模板 + fence/wait_wgmma → nan
- R3a: FlashQLA 配置 + WY-O + Q'复用K → chunk0 对 chunk1+ 错
- R3b: 原始 O 形式 → 死锁 (WY-O 是必须的)
- R3c: WY-O + Qp 独立 → nan
- 根因: tilelang 无 4-WG 范本, FlashQLA barrier 极复杂(13 barrier), 跨 chunk state 竞争无法定位
- **结论: 4-WG 在当前 tilelang + MIG + 远程 5min walltime 下不可行**

### 替代方向 (均预期收益有限, 未验证)
- **state ping-pong shared**: state 不常驻 fragment, 两份 shared ping-pong 降寄存器. 风险: shared 预算紧
- **per-case autotune (离线)**: 离线搜参后硬编码 (OJ 禁 autotune). 当前 DV 配置已接近最优, 收益小
- **TMA 替代 async_copy**: T.tma_copy + mbarrier. 需 warp spec 配合, 4-WG 不可行则受限
- **local window prefix**: block-prefix + 最近 L=4/8 token 精确修正 (带宽 GEMM). 但把消掉的 GEMM 加回来, 收益存疑
- **实验报告**: 作业页面明确"实验报告是主要评分依据". 三轮 4-WG 踩坑 + WY-O 数学推导 + FlashQLA 源码分析 +
  WY-O+async+融合三路证伪 + Qhat/Khat 换元证伪 + prefix-state 近似证伪 (含 tilelang async API 调研) 是报告亮点,
  可能比 OJ 93→100 更值

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
