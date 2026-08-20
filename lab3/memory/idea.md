# GDN Prefill 110 分新路线设计

- 日期：2026-08-20
- 目标：在当前约 93 分基线之上，继续寻找能够把长序列 case 推向 FlashQLA (`p≈1.0`) 的大收益路线。
- 当前基线：`lab3/student/tilelang_fwd.py` 的 matw 路线，长序列约 `chain 0.53ms / long_low 3.04ms / wide 3.77ms / deep 4.46ms`。
- 本文件用途：作为高层 AI 与低级实现 AI 之间的优化设计与实验记录平台。低级 AI 每次实现后，应在本文件追加：commit、开关、编译结果、8 case 精度、每 case t100、失败原因。

## 0. 已经确认的事实

### 0.1 不再把 Qhat/Khat 当主路线

数学换元 `Qhat=exp(g)Q, Khat=exp(-g)K` 正确，但之前的实现退步：global load 被改成 element-wise，破坏了 TileLang pipeline；长序列也退步。除非保留原始 `T.copy` load、在 load 完成后才原地缩放，否则不再优先投入时间。

### 0.2 不再尝试无修正的 block-prefix

`O_i≈exp(g_i)Q_i(S_old+P_block)` 会丢掉 chunk 内近邻因果项。GDN 的近对角 decay 接近 1，block=16 也会有巨大误差。state 精确并不能挽救 output 误差。

### 0.3 单独删除 QK 和 dsV 的收益上限不够

对于 `chunk=64, DK=DV=128`，主要 FLOP 近似为：

```text
W=A@(beta*gamma*K)       1.048M FMA
U=A@(beta*V)             1.048M FMA
W@S                      1.048M FMA
Q@S                      1.048M FMA
K^T@V_new                1.048M FMA
Q@K^T                    0.524M FMA
ds@V_new                 0.524M FMA
```

QK 与 dsV 只占约 16.7%。即使完全免费，纯计算上限也约为 1.20x。因此冲 110 不能只依靠 prefix 数学化简，必须同时攻击 GEMM 等待、shared round-trip、寄存器压力、三角结构或跨 head 重复计算。

---

# 1. 第一主路线：非 WY 的纯精确 WGMMA wavefront

## 1.1 核心想法

不改变任何数学公式，不改变输入输出 dtype，不新增 shared buffer。把当前每个 `T.gemm` 后的隐式 `wait_group 0` 改成按依赖图批量发射：

```text
当前：每个 GEMM -> commit -> wait -> 下一个 GEMM
目标：无依赖 GEMM 连续 issue -> 一次 wait -> 消费结果
```

之前 `WY-O + async` 退步不能否定这条路线，因为 WY-O 额外增加了 `ds@W`、Q'、fragment copy 和寄存器压力。本路线只对原始 matw 做 async 调度。

## 1.2 推荐依赖批次

### Batch A：四个独立 GEMM

在所有输入 shared 尚未覆盖前，连续发射：

```python
T.wgmma_gemm(A_shared, bkg_shared, W_fragment, clear_accum=True)
T.wgmma_gemm(A_shared, bv_shared, U_fragment, clear_accum=True)
T.wgmma_gemm(Q_shared, K_shared, ds_fragment,
             transpose_B=True, clear_accum=True)
T.wgmma_gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
T.wait_wgmma(0)
```

依赖为：

```text
W        <- A, bkg
U        <- A, bv
QK       <- Q, K
a=Q@S    <- Q, S
```

四个 GEMM 的输入只读、输出不同，可以一次性在同一个 warpgroup 发射。TileLang 的 `T.wgmma_gemm` 是明确的 async 接口；只能在这一批所有结果不再被读取前调用一次 `T.wait_wgmma(0)`。

如果一次发射四个 WGMMA 造成寄存器压力、编译失败或实测反而变慢，按以下降级顺序测试：

```text
Batch A1: W, U, QK -> wait
Batch A2: Q@S      -> wait
```

不要直接退回每个 GEMM 后 wait。

### Batch B：W@S

完成 Batch A 后，将 W fragment 放入 W workspace，然后：

```python
T.wgmma_gemm(W_shared, s_shared, WS_fragment, clear_accum=True)
T.wait_wgmma(0)
```

这一步完成后才能把 U 和 WS 做差得到 V_new。

### Batch C：output correction 与 state update

V_new 写入 shared 后：

```python
# 先对 s_fragment 做本 chunk 的整体 gate 缩放
for k, v in T.Parallel(DK, block_DV):
    s_fragment[k, v] = s_fragment[k, v] * exp_g_last

T.wgmma_gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
T.wgmma_gemm(K_shared, V_new_shared, s_fragment,
             transpose_A=True, clear_accum=False)
T.wait_wgmma(0)
```

两者都只读 V_new，输出分别是 O 和 state，因而可以同批发射。

## 1.3 进一步的依赖压缩

如果 Batch A 稳定，可以测试更激进版本：

```text
Batch A: W, U, QK, Q@S -> wait
Batch B: W@S           -> wait
Batch C: ds@V_new, K^T@V_new -> wait
```

目标是把原来约 7 次 `wait_group 0` 压到 3 次。这个路线的收益来自 Tensor Core pipeline 空洞和同步延迟，不依赖删除 FLOP。

## 1.4 正确性限制

- 不能在 `wait_wgmma` 前读取任何输出 fragment。
- 不能在 wait 前覆盖 Q/K/A/V/S 输入 shared。
- 不要添加额外的 `T.wait_wgmma(0)` 到每个 GEMM 后。
- 第一轮只改 `T.gemm`/wait 调度，不做 buffer alias，便于归因。

---

# 2. 第二主路线：精确 buffer lifetime coloring

这条路线不改变数学，只按真实生命周期复用 shared 和 fragment。建议在 Async Batch 单独验证后逐项加入。

## 2.1 生命周期表

| 物理 workspace | 生命周期 | 可替代对象 |
|---|---|---|
| `bkg_shared` | `beta*gamma*K -> W` | 删除独立 `W_shared` |
| `V_shared` | `V -> beta*V -> U -> V_new` | 删除 `bv_shared`、`V_new_shared` |
| `A_shared` | `A -> ds`（W/U 完成后） | 删除 `ds_shared` |
| 一个 `[64,DV]` FP32 fragment | `U -> W@S -> Q@S -> ds@V_new` | 删除独立 `O_fragment` 或 `tmp_dv2` |
| gate slot 0 | `g -> exp(g)` | 删除单独 `g_exp_shared` 或 `g_shared` |
| gate slot 1 | `beta -> exp(-g)` | 删除 `beta_g_shared` |

## 2.2 `bkg -> W`

```python
T.gemm(A_shared, bkg_shared, acc_w, clear_accum=True)
T.copy(acc_w, bkg_shared)
# 之后 bkg_shared 直接作为 W_shared
```

`bkg_shared` 在 W GEMM 完成前不能覆盖；完成后其旧值已经死亡。

## 2.3 `V -> beta*V -> U -> V_new`

推荐将 U 先 copy 回原来的 V workspace，再在 shared 中完成减法：

```python
T.gemm(A_shared, V_workspace, acc, clear_accum=True)
T.copy(acc, V_workspace)                # V_workspace = U
T.gemm(W_workspace, s_shared, acc, clear_accum=True)
for t, d in T.Parallel(block_S, block_DV):
    V_workspace[t, d] = V_workspace[t, d] - acc[t, d]
# V_workspace = V_new
```

当前版本已经先将 U 写成 BF16 shared，再从 shared copy 回 fragment；因此该复用不会额外引入一次新的 BF16 量化点。

## 2.4 `A -> ds`

W/U 两次 GEMM完成后，A 的旧内容死亡：

```python
T.copy(ds_fragment, A_shared)
# 后续以 A_shared 作为 ds_shared
```

不要在 W/U 完成之前覆盖 A。

## 2.5 fragment 复用

最安全的顺序是：

```text
acc = U
copy U -> V_workspace
acc = W@S
V_workspace -= acc
acc = Q@S
acc += ds@V_new
write output
```

这样 `V_new` 一旦写入 shared，原来的 U/WS accumulator 就死亡，可以被 output accumulator 复用。

## 2.6 共享内存估算

采用上述 alias 后，单 stage 逻辑 shared 约为：

```text
s_shared       32 KiB  (persistent)
Q_shared       16 KiB
K_shared       16 KiB
V_workspace    16 KiB
A_workspace     8 KiB
W_workspace    16 KiB
gates          ~1 KiB
```

chunk 工作区约 72 KiB，persistent state 和 gate 加入后约 105 KiB。若两级 pipeline versioning，保守约 177 KiB，仍低于约 227 KiB 上限，理论余量约 50 KiB。

注意：TileLang 实际是否对某个 alias buffer 做 multi-version 需要以编译后的 dynamic shared bytes 验证，不能只依赖手算。

---

# 3. 第三主路线：WGMMA RS，删除 fragment->shared round-trip

这是尚未充分验证的硬件路径，可能比普通 alias 更有收益。

## 3.1 W@S 的方向正好适合 RS

Hopper WGMMA 的 RS 形式是：

```text
A: register operand
B: shared operand
C: register accumulator
```

`W @ S` 的数学方向正是：

```text
W: fragment/register
S: shared
```

因此应制作最小编译 probe：

```python
W_fragment = T.alloc_fragment((block_S, DK), dtype=T.bfloat16)
WS_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
T.gemm(W_fragment, s_shared, WS_fragment, clear_accum=True)
```

如果普通 `T.gemm` 不选择 RS，则试：

```python
T.wgmma_gemm(W_fragment, s_shared, WS_fragment, clear_accum=True)
```

目标是删除：

```text
W_fragment -> W_shared
W_shared   -> W@S
```

## 3.2 ds@V_new 也可尝试 RS

`ds` 本来最终被存成 BF16 shared，而 `ds_fragment` 可尝试直接采用 BF16 accumulator：

```python
ds_fragment = T.alloc_fragment((block_S, block_S), dtype=T.bfloat16)
T.gemm(Q_shared, K_shared, ds_fragment,
       transpose_B=True, clear_accum=True)
# gate + lower mask in ds_fragment
T.gemm(ds_fragment, V_new_shared, O_fragment, clear_accum=False)
```

如果 TileLang 能将第一操作数作为 register operand，这能删除 ds 的 shared round-trip。

精度检查必须覆盖所有 8 case，尤其是 state case。若 BF16 fragment 的误差不通过，则保留 FP32 ds fragment，只测试 W 的 RS。

## 3.3 风险

- TileLang 对 fragment 第一操作数的 layout inference 尚未完全确认。
- WGMMA RS 输入通常要求 BF16/FP16 register operand，不能直接把 FP32 accumulator 当作 BF16 A operand。
- 先做独立 probe，不要直接改主 kernel。

---

# 4. 第四主路线：精确利用下三角结构

当前 `A` 和 chunk 内 gated score 都具有因果下三角结构。当前全矩阵 GEMM 会计算大量之后被清零的上三角。

## 4.1 同时攻击 A@K/A@V 和 QK/dsV

如果把 64 行按 16 或 32 行拆块：

```text
A@B：只计算 row_block >= col_block 的块
QK/dsV：只计算 row_block >= key_block 的块
```

块大小 32 时，有效块为 3/4；块大小 16 时，有效块为 10/16。理论上可以减少三角矩阵的无效乘法。

## 4.2 关键风险

- `M=16/32` 可能无法使用高效 WGMMA，可能退化为 MMA/SIMT。
- 对角块仍需 mask，不能直接认为全部节省。
- TileLang 对 shared 第一维 offset 的 T.gemm region 有限制，可能需要独立 sub-shared。
- 子块 copy 可能重新引入带宽和同步开销。

## 4.3 实验顺序

```text
T=32 row/column tile，先测编译和精度
T=16 row/column tile，再测实际是否仍是 WGMMA
```

只保留满足以下条件的版本：

```text
全 8 case 精度通过
实际 Tensor Core 指令没有严重退化
long_low 或 deep 至少有明确正收益
```

这条路线的意义在于同时减少 A@K/A@V 这两个大 GEMM，而不是只优化 QK/dsV；否则收益不够。

---

# 5. 第五主路线：output-only 自适应局部修正

无修正 block-prefix 已经证伪，但以下版本仍有数学价值：state 精确，只有 output 近似。

## 5.1 公式

在 block 边界保存精确 prefix state：

```text
X_b = S_old + sum_{j < block_start} exp(-g_j) K_j^T V_new_j
```

对 block 内第 i 行：

```text
O_i ≈ scale * [ exp(g_i) Q_i X_b
              + sum_{j=max(block_start,i-L+1)}^i
                exp(g_i-g_j) (Q_i K_j^T) V_new_j ]
```

state 仍使用完整 block：

```text
X_{b+1} = X_b + K_block^T @ (exp(-g_block) V_new_block)
```

因此近似误差不会跨 chunk 累积。

## 5.2 不能使用 L=0

之前 block-constant 失败，说明必须保留：

- 对角项；
- 最近若干 token 的因果项。

建议测试：

```text
block=16, L=4
block=16, L=8
block=32, L=8
```

优先采用 gate-adaptive window：如果 `g_i-g_j` 已经低于阈值，则更远项可忽略；否则延长窗口。

## 5.3 这条路线的真实收益上限

prefix 版本保留 `Q@state` 和 state update，主要减少 QK/dsV。它单独理论收益有限，只有和以下优化叠加才值得：

```text
Async Batch
+ alias
+ 小窗口 output correction
```

如果 L=4/8 的精度无法通过，不再继续扩大复杂度。

---

# 6. 第六主路线：GVA 跨 V head 共享 raw QK

当 `Hv/Hq=4` 时，一个 Q/K head 被四个 V head 复用。可以让一个 CTA 先计算 raw `Q@K^T`，再让四个 V head 复用。

但 gate 通常按 V head 不同，因此只能共享 raw QK，不能直接共享 ds。

理论节省每组约 6.25% 总 FLOP，远不足以单独冲 110；只有在以下条件满足后才值得尝试：

```text
Async Batch 已稳定
workspace alias 已完成
CTA 数量仍足够填满 MIG 的 14 SM
```

优先考虑两 head/CTA，而不是四 head/CTA，以避免 CTA 数量骤减和 state shared 爆炸。

---

# 7. 实现与实验顺序

低级 AI 按以下顺序实现，每一步都保留可回退版本：

## Phase A：无数学变化

1. 纯 Async Batch，保持原 buffer 和 load；
2. Async Batch 只发射 W/U/QK/QS；
3. 再加入 dsV/state 双发射；
4. 比较 3-batch、2-batch、4-batch 的长序列时间和寄存器。

## Phase B：低风险原地复用

5. `bkg -> W`；
6. `V -> betaV -> U -> V_new`；
7. `A -> ds`；
8. `tmp_dv2/O_fragment` 复用。

每次只加一个 alias，避免 shared lifetime 竞争无法定位。

## Phase C：RS probe

9. 单独验证 `W_fragment @ s_shared` 是否能走 WGMMA RS；
10. 再验证 BF16 ds fragment 作为 RS A operand；
11. 只把编译、精度和生成指令都通过的 RS 合入主 kernel。

## Phase D：三角结构

12. 先测 block=32；
13. 再测 block=16；
14. 检查实际 WGMMA/MMA 指令和长序列时间。

## Phase E：近似

15. 先做 torch/reference 的 block=16,L=4/8 精度扫描；
16. 只有参考误差可接受时才实现 TileLang；
17. state 必须始终使用完整精确更新，近似只允许写 output。

## Phase F：GVA fusion

18. 仅在前面路线仍不足且资源允许时测试两 V head/CTA。

---

# 8. 不要继续投入的路线

```text
无修正 block-prefix
Qhat/Khat 的 element-wise global load 版本
单 WG WY-O
WY-O F1/F2/F3 大堆叠融合
TMA 替代现有 load 的默认版本
4-WG barrier 猜测式修复
单独删除 QK/dsV 并期待 1.6x
```

---

# 9. 最重要的判断

如果 Phase A 的纯 Async Batch 也没有明显收益，那么问题不是单个 GEMM 的等待，而是：

```text
1. MIG 上 Tensor Core 吞吐没有被当前 CTA 充分利用；或
2. register/shared spill 已经主导；或
3. 每 chunk 的 global/shared load 与 fragment copy 主导；或
4. 需要真正的 2-WG/跨 head 协作，而不是单 WG 调度。
```

此时下一条最值得赌的路线不是继续改 prefix，而是：

```text
A@K/A@V 下三角分块 + QK/dsV 下三角分块
```

因为它同时攻击两个大 GEMM 家族，才有机会接近 1.4x；再叠加 Async Batch 和 alias，才可能向 110 分靠近。

---

# 10. 第二波新想法：共享右操作数的“纵向堆叠 GEMM”

- 日期：2026-08-20
- 状态：本节是新的高层设计，尚未实现；低级 AI 应按 Phase G 顺序试验。
- 目标：不是继续微调 wait，而是利用三个 GEMM 对同一个操作数的共享关系，把 **两次独立 GEMM 合成一次更大的 WGMMA**，同时用生命周期别名把新增的大 fragment 抵消掉。

## 10.1 最值得先做：`[ds; K^T] @ V_new` 一次完成 output correction + state update

当前最后两次 GEMM 是：

```text
Y_o = ds @ V_new       [64,64]  @ [64,128] -> [64,128]
Y_s = K^T @ V_new      [128,64] @ [64,128] -> [128,128]
```

二者的右操作数完全相同，应该做纵向堆叠：

```text
P = [ ds^T | K ]        shape [64, 64+128] = [64,192]
P^T @ V_new             shape [192,128]

上 64 行 = ds @ V_new
下 128 行 = K^T @ V_new
```

这里 `P` 的布局是故意选择为 `[64,192]`，这样一个 `transpose_A=True` 的 GEMM 同时完成两个结果：

```python
# P[:, :64]   = ds^T
# P[:, 64:]   = K
T.gemm(P, V_new_shared, state_o_fragment,
       transpose_A=True, clear_accum=False)
```

### 为什么这是新的高价值点

- 不是只减少 launch；它把两个结果共享的 `V_new` 读取、WGMMA issue 和 accumulator 处理合并为一个 `M=192` GEMM。
- 当前两次 GEMM 的总输出元素是 `64*128 + 128*128 = 24576` 个 FP32 元素。
- 堆叠后的 `[192,128]` 也是 `24576` 个 FP32 元素，**输出 fragment 字节数完全相同**。
- 当前 `ds_shared` 8 KiB + `K_shared` 16 KiB = 24 KiB；`P_shared[64,192]` BF16 也是 24 KiB，**输入 shared 字节数完全相同**。
- 关键不是增加内存，而是把“两个输出”变成一个完整的 WGMMA accumulator。

### 精确的 state/output fragment 复用方案

当前有：

```text
s_fragment  [128,128] FP32 = 64 KiB   # 跨 chunk 持久 state
O_fragment  [64,128]  FP32 = 32 KiB   # 当前 chunk output
```

恰好可以改成一个：

```text
state_o_fragment [192,128] FP32 = 96 KiB

state_o_fragment[0:64,   :] = 当前 chunk 的 output correction
state_o_fragment[64:192, :] = 下一 chunk 要继续使用的 state
```

每个 chunk 的顺序必须是：

```text
1. 先把 state_o_fragment[64:192] 复制到 s_shared，作为 S_old。
2. 用 s_shared 完成 W@S、Q@S 和 V_new。
3. 将 state_o_fragment 上半 64x128 清零。
4. 将 state_o_fragment 下半 128x128 乘以 gamma_last。
5. 发射一次 P^T @ V_new：
   - 上半累加 ds@V_new；
   - 下半累加 K^T@V_new。
6. 上半加到 Q@S 的临时 output fragment 后写回 O。
7. 下半保留为下一 chunk 的 state。
```

`Q@S` 仍需要一个 `[64,128]` FP32 临时 fragment，但可以复用原来计算 W 的 `tmp_dv[64,DK]`：

```text
W 计算完成并 copy 到 shared 后，tmp_dv 不再保存 W；
tmp_dv 复用为 Q@S/output-base（DK=DV=128 的 OJ case 形状相同）。
```

因此 target case 的 fragment 峰值可以保持不变：

```text
原来：s_fragment 64 + tmp_dv 32 + tmp_dv2 32 + O_fragment 32 = 160 KiB（另加 ds_tmp）
现在：state_o_fragment 96 + tmp_dv 32 + tmp_dv2 32       = 160 KiB（另加 ds_tmp）
```

这是本文件目前最重要的“空间换时间甜点”：**用统一的 state/output fragment 把 M=192 堆叠 GEMM 做出来，而不是简单额外分配一个 96 KiB fragment。**

### P_shared 的生命周期

```text
P_shared[:, 64:] 先装 K，整个 chunk 保留；
QK 计算使用 P_shared[:,64:]，不需要独立 K_shared；
QK 结束后，把 ds_tmp 的转置结果写入 P_shared[:,:64]；
P_shared 变成 [ds^T | K]，最后一次 GEMM 直接使用。
```

所需 shared：

```text
P_shared: 64*192*2 = 24 KiB
原 K_shared + ds_shared: 16 + 8 = 24 KiB
```

因此 `P_shared` **不增加 shared memory**，只是要求：

- `T.copy(K_global, P_shared[:,64:])` 的列切片合法；
- 用一个小的 `T.Parallel` 把 `ds_tmp[i,j]` 转置写成 `P_shared[j,i]`；
- `T.gemm(..., M=192, N=DV, K=64, transpose_A=True)` 的 WGMMA lowering 可用。

TileLang 的 Hopper warp partition 条件是 M 为 16 的倍数、N 为 8 的倍数，并要求 M 能被一个 warpgroup 的行覆盖；`M=192,N=128,K=64` 应当满足 128/256 threads 下的分区条件，但必须先做单 kernel probe。若当前版本的宏生成器拒绝 M=192，直接停止这条实现，不要偷偷用普通 MMA 替代；退路是保留两次 GEMM但用同一批 `T.wgmma_gemm` 发射。

### 数值顺序要求

`state_o_fragment` 下半必须在 GEMM 前完成：

```text
state_o[64:192] *= gamma_last
```

然后用 `clear_accum=False` 加 `K^T@V_new`。上半必须清零后再加 dsV。不能把下半 state 先 copy 成 BF16 再用作累加器；只允许 copy 到 `s_shared` 作为 W/Q 的输入，FP32 state 本体一直留在 fragment。

## 10.2 第二个高价值堆叠：`A @ [βγK | βV]` 一次完成 W 和 U

当前：

```text
W = A @ (βγK)  [64,64] @ [64,128] -> [64,128]
U = A @ (βV)    [64,64] @ [64,128] -> [64,128]
```

改成：

```text
B_wu = [βγK | βV]      [64,256] BF16
WU = A @ B_wu          [64,256] FP32
```

### 存储刚好相等

```text
bkg_shared 16 KiB + bv_shared 16 KiB = 32 KiB
B_wu_shared [64,256] BF16             = 32 KiB

tmp_dv [64,128] FP32 + tmp_dv2 [64,128] FP32 = 64 KiB
WU_fragment [64,256] FP32                    = 64 KiB
```

实现顺序：

```text
1. B_wu[:,:DK]   写 βγK；
2. B_wu[:,DK:]   写 βV；
3. 一次 T.gemm(A_shared, B_wu, WU_fragment)；
4. 把 WU 上半 copy 回 B_wu[:,:DK]，把下半 copy 回 B_wu[:,DK:]；
5. B_wu 之后同时充当 W 和 U workspace。
```

这条路线删除一次 `A` 的 WGMMA，并且 A 只从 shared 读取一次。它不改变数学，也不需要新增 shared；风险主要是：

- `N=256` 的 WGMMA 指令序列变长；
- `[64,256]` fragment 的 layout/copy split 必须可用；
- WU_fragment 的形状不能直接当作 `[64,128]` fragment，后续应单独保留一个可复用的 `work_dv`，不能强行用带 offset 的 fragment 当 C。

因此它应在 10.1 的 M=192 probe 之后独立测试，不要两条堆叠同时改，避免无法归因。

## 10.3 第三个堆叠候选：`[Q; W] @ S_old`

```text
[Q; W] @ S_old = [Q@S_old; W@S_old]
```

理论上是一个 `M=128,N=DV,K=DK` GEMM，shared 输入大小等于 `Q_shared+W_shared`，输出元素数等于两个原输出之和。

但它的收益低于 10.1/10.2，原因是：

- Q 和 W 的生命周期不同时结束，必须把 Q/W 放进一个 `[128,DK]` packed shared；
- 两个输出要拆成 `Q@S` 和 `W@S`，之后还要继续做 `V_new=U-W@S`；
- C 的下半/上半 fragment 不能直接当两个独立 WGMMA accumulator，TileLang 对 fragment 首维 offset 没有稳定先例；
- 如果为拆分额外增加一个 32 KiB fragment，可能抵消 WGMMA 节省。

只作为 10.1/10.2 成功后的 probe，不作为主路线。

# 11. 新的 output-only 数学近似：精确 state + 低秩 causal prefix

之前的 block-prefix 失败，是把整个 chunk 内 prefix 当成一个常数；这里不再做这个错误近似。state 仍然完全按原公式精确更新，只对 output 的 `ds@V_new` 做一次性近似，因此误差不会跨 chunk 累积。

## 11.1 公式

令：

```text
φ_q(i) = projection(Q_i)       [r]
φ_k(j) = projection(K_j)       [r]
```

用低秩近似：

```text
Q_i K_j^T ≈ φ_q(i) · φ_k(j)
```

则：

```text
ds@V_new[i]
≈ exp(g_i) * φ_q(i) @
   prefix_sum_{j<=i}( exp(-g_j) * φ_k(j) ⊗ V_new[j] )
```

其中 prefix state 形状为 `[r,DV]`，只服务当前 output，不写回真实 `S`。

推荐先测固定正交投影/分块投影的 `r=16,32`：

```text
r=16:
  两次 Q/K 投影约 2*64*128*16 = 262K FMA
  prefix state 更新 + query read 约 2*64*16*128 = 262K FMA
  总量约 524K FMA，低于原 QK+dsV 约 1.048M FMA

r=32:
  约 1.048M FMA，主要收益来自不再物化 64x64 ds 和更好的带宽/寄存器行为
```

`r=16` 可能误差较大，但它是 output-only；即使误差不合格，也不会污染下一 chunk 的 state。不要把低秩结果用于 `K^T@V_new` 或 `V_new`。

## 11.2 不要用普通逐 token 串行循环

直接按 token 更新 `[r,DV]` 会把并行 GPU 退化成 64 步串行。可实现的版本应该是：

```text
1. 每个 token 生成 z_j = exp(-g_j) * φ_k(j)；
2. 对 z_j[:,d] 做 64 长度并行 prefix-scan；
3. 每行用 φ_q(i) 与 prefix state 做 r×DV 乘法；
4. 乘 exp(g_i)，写 output。
```

可用两个 `[64,r,DV]` BF16/FP16 scratch 做 Hillis-Steele，或者先用 shared 上的 `[r,DV]` block scan probe。scan scratch 的空间：

```text
[64,r,DV] BF16:
  r=16 -> 64*16*128*2 = 256 KiB，不可接受；
  r=16 只能使用按 DV tile 的局部 scan，或逐 r 分批复用 4 KiB scratch；
```

因此推荐的实现不是完整三维 scratch，而是：

```text
每个 DV tile 单独处理；
[64,r,block_DV] BF16 双 buffer；
block_DV=64,r=16 -> 128 KiB/个 buffer，双 buffer 为 256 KiB，仍然过大；
或对每个 r/d 采用寄存器 + warp shuffle 做 prefix。
```

这是高风险近似路线，只有精确堆叠优化完成后再试。

## 11.3 自适应 rank

不能一开始固定 r=16 直接送 OJ。建议根据每个 case 离线测：

```text
- r=8/16/32；
- 只替换 output_in_chunk；
- state 与 final_state 使用原精确路径；
- 误差全过才进入 dispatch。
```

如果某 case 的误差不过，完全回退到精确 ds GEMM；不要让错误近似影响所有 case。

# 12. 新的 output-only 近似：最终 state 补偿 + 近未来修正（只作为备选）

另一条等价恒等式是：

```text
H_new = S_new / exp(g_last)
      = S_old + Σ_j exp(-g_j) K_j^T V_new_j

O_i / scale
= exp(g_i) Q_i H_new
  - exp(g_i) Σ_{j>i} exp(-g_j)(Q_i K_j^T)V_new_j
```

可以用 `Q@H_new` 作为 base，只对最近 `w` 个未来 token 做精确补偿，远未来项直接丢弃。它不改变 state，因此没有跨 chunk 误差累积。

但这条路线必须有输入条件判断：若未来 gate tail 不小，补偿项可能很大，且是两个大数相减，数值稳定性差。只有当：

```text
max_{j>i+w} |exp(g_i-g_j)| < threshold
```

对整个 query tile 成立时才启用；否则回退精确路径。推荐 probe：`w=4,8,16`，只对 output 近似，不要改 state。

这条路线理论上可把当前 `Q@S_old + QK + dsV` 的 3 个 output GEMM 降为 `Q@S_new + local correction`，但未经误差/性能验证，优先级低于 10.1 和 10.2。

# 13. 当前应执行的实现顺序（只给低级 AI）

## Phase G0：先验证 shape，不碰主 kernel

写最小 probe，验证以下三种 WGMMA shape 是否能编译并正确：

```text
G0.1: M=192,N=128,K=64, transpose_A=True, C=FP32
G0.2: M=64,N=256,K=64, C=FP32
G0.3: M=128,N=128,K=128, C=FP32（[Q;W]@S 备用）
```

若 G0.1 失败，不得把 M=192 改成普通 MMA 混进主 kernel。

## Phase G1：最终两 GEMM 合并（主路线）

只做 10.1：

1. `P_shared[64,192]` 替换 `K_shared + ds_shared`；
2. `state_o_fragment[192,DV]` 替换 `s_fragment + O_fragment`；
3. `tmp_dv` 在 W 完成后复用为 Q@S；
4. 保持 W/U 两次普通同步 GEMM；
5. 保持 V_new 与 state gate 的原数值顺序；
6. 先用 `block_DV=64, threads=128` probe，再用 `block_DV=128, threads=256`。

验收：8 case 精度全过，且至少一个长序列 case 比 matw 明显下降。若寄存器溢出，先只对 `block_DV=64` 开关，不要改数学。

## Phase G2：W/U 横向堆叠

只做 10.2，保持 G1 关闭：

1. `B_wu[64,DK+DV]`；
2. `WU_fragment[64,DK+DV]`；
3. split copy 到 W/U workspace；
4. 其余计算保持原 matw；
5. 分别比较 `N=256` 与两次 `N=128` 的时间。

## Phase G3：G1 + G2 组合

只有 G1/G2 各自通过精度且单独加速，才组合。组合后重新调：

```text
block_DV=64/128
threads=128/256
num_stages=1/2
```

## Phase G4：output-only 低秩 prefix

仅在精确路线不够时测试 r=8/16/32；默认 dispatch 必须有精确回退。

# 14. 对“中间存储到底够不够”的最终结论

针对 `DK=DV=128, CHUNK=64`：

| 方案 | 新增/减少 shared | fragment 变化 | 结论 |
|---|---:|---:|---|
| `[ds;K^T]@V_new`，独立 state + output | 0 KiB（pack=K+ds） | +32 KiB | 能编译但寄存器风险较高 |
| `[ds;K^T]@V_new`，统一 state/output fragment | 0 KiB | **0 KiB**（state64+O32=stack96） | **最值得先试**，依赖 fragment offset/slice |
| `A@[βγK|βV]` | **0 KiB**（bkg+bv=pack） | 0 KiB（两个 32K=一个64K） | **第二优先** |
| `[Q;W]@S` | 0 KiB（Q+W=pack） | 理论 0 KiB，实际 split 可能 +32 KiB | 第三优先 |
| 低秩 causal prefix r=16 | 约 0 KiB（按 tile 复用） | 可少 ds 64x64，但 scan scratch 高风险 | output-only 备选 |

同时，以下原地复用必须始终打开（这些已经是确定正确的）：

```text
bkg_shared: βγK -> W
V_shared:   V -> βV/U -> V_new -> gated V_new
A_shared:   A -> ds
```

这些 alias 将单 stage shared 从约 161 KiB 降到约 105 KiB；在 num_stages=2 下约 177 KiB，低于 H800 MIG 的约 227 KiB opt-in 上限，说明“空间不够”不是堆叠方案的硬障碍，真正风险是 fragment layout、M=192 lowering 和寄存器压力。

# 15. 给低级 AI 的硬性实验记录格式

每完成一个 G0/G1/G2/G3/G4 版本，追加：

```text
commit:
env switch:
shape/probe:
compile:
precision (8 cases):
t100 (8 cases):
threads/block_DV/num_stages:
shared bytes:
register/spill:
是否比 matw 快:
失败原因:
```

不要再做以下无关路线：

```text
- 无修正 block-prefix；
- 破坏 T.copy pipeline 的 Qhat/Khat load；
- 单 WG WY-O；
- 只删除 QK+dsV 并期待 1.6x；
- 未证明的 4-WG barrier 猜测；
- 同时改数学、布局、async、dispatch，导致无法归因。
```

---

# 16. 实验记录 (低级 AI 追加)

## 16.1 Phase G0: WGMMA shape probe (2026-08-20)

```text
commit: (uncommitted, probe_g0.py)
env switch: 无 (独立 probe 脚本)
shape/probe:
  G0.1: M=192,N=128,K=64, transpose_A=True, C=FP32  ([ds;Kᵀ]@V_new 堆叠所需)
  G0.2: M=64,N=256,K=64, C=FP32                     (A@[βγK|βV] 堆叠所需)
  G0.3: M=128,N=128,K=128, C=FP32                    ([Q;W]@S 备用)
compile: 全 OK (3 shape 都走 wgmma lowering, 无 fallback)
precision: max_diff = 1.14e-05 (G0.1/G0.2) / 2.29e-05 (G0.3), nan=False, 远低于 RTOL=5e-3
threads/block_DV/num_stages: threads=128, 单 kernel 无 pipeline
是否比 matw 快: N/A (probe 只验证 shape)
失败原因: 无
结论: ★ M=192 transpose_A / N=256 / M=128K=128 三种 WGMMA shape 全部可用且精确.
  idea.md 第10节堆叠方案 (G1/G2) 的 shape 前置条件满足, 可进入 Phase G1 实现.
```

## 16.2 Phase A: 纯 Async Batch matw (2026-08-20)

```text
commit: (uncommitted, _gdn_async_matw_kernel in tilelang_fwd.py)
env switch: GDN_ASYNC=1
shape/probe: matw 数学不变, 仅把 T.gemm 换 T.wgmma_gemm+wait_wgmma(0) 批量调度
  Batch A (4 并发 state-free): W=A@bkg, U=A@bv, QK=Q@Kᵀ, QS=Q@S_old → 4 wgmma → wait(0)
  Batch B (依赖 W): WS=W@S_old → 1 wgmma → wait(0)
  Batch C (依赖 V_new, 累加): O+=ds@V_new, s+=Kᵀ@V_new → 同步 T.gemm (clear_accum=False 需 fence accum)
compile: OK
precision (8 cases): 全 PASS
t100 (8 cases) vs matw 基线:
  chain_equal:    0.865ms (vs matw 0.538, +61% 慢)
  long_low_gva:   3.081ms (vs matw 3.078, +0.1% 持平)
  wide_gva_state: 3.878ms (vs matw 3.77, +2.9% 略慢)
  deep_gva_state: 4.563ms (vs matw 4.46, +2.3% 略慢)
threads/block_DV/num_stages: 同 matw
是否比 matw 快: 否. chain 慢 61%, 长序列持平/略慢.
失败原因:
  (1) chain (小 grid) 慢 61%: async wgmma 在 grid=8 下 launch/同步开销 > 批量发射收益.
  (2) 长序列持平: Batch A 4 wgmma 并发占用 accum fragment, 寄存器压力激增, HW OoO 重叠有限.
  (3) Batch C 必须同步: clear_accum=False 累加需 T.Parallel gate 对 accum 可见, wgmma async 不 fence accum.
结论: ★ idea 第一主路线 (纯 Async Batch) 证伪. 单 WG 下 async wgmma 无收益:
  async wgmma 收益前提是多 WG (多 consumer 并行 issue+wait), 单 WG issue 仍串行, wait 同步点反增开销.
  需 4-WG 才能藏 wgmma 延迟 (但 4-WG 不可行).
```

## 16.3 Phase G1: [ds; Kᵀ]@V_new 堆叠 (2026-08-20)

```text
commit: (uncommitted, _gdn_stack_matw_kernel in tilelang_fwd.py)
env switch: GDN_STACK=1
shape/probe: M=192, N=DV, K=64 (P_shared[192,64]@V_new[64,DV]→state_o[192,DV])
  P_shared[192,64] = [ds(64×64); Kᵀ(128×64)] 纵向 (修正 idea 10.1 横向布局:
    idea 的 P[64,192]=[dsᵀ|K] 横向 + transpose_A 在 tilelang 下 reduction=192≠64, 报错 K_A=128 K_B=192.
    修正: P[192,64] 纵向 [ds; Kᵀ], 无 transpose, reduction=64 ✓)
compile: OK (DV=64 path; DV=128 path shared 247KB > 232KB 超限, 仅 DV=64)
precision:
  state diff 0.0035 (PASS, BF16 噪声, state 路径精确)
  output chunk0 abs 0.059, chunk1 abs 0.053 (FAIL, RTOL=5e-3)
  long_low_gva 55% mismatch abs 0.096
threads/block_DV/num_stages: th=128, DV=64, st=2 (仅小 grid 路径, DV=128 shared 超限)
是否比 matw 快: N/A (精度未过)
失败原因:
  ★ GDN gating 矛盾 (数学 obstruction, 非实现 bug):
    O = scale·(γ⊙(Q@S_old) + ds@V_new)        → ds@V_new 用 UNGATED V_new
    S_new = γr·S_old + Kᵀ@(γr·γ_inv·V_new)    → Kᵀ@V_new 用 GATED V_new (×γr·γ_inv)
    堆叠 [ds; Kᵀ]@V_new 用同一个 V_new 操作数, 无法同时满足两种 gating.
  选 gated V_new 堆叠: state 精确 (Kᵀ@(γr·γ_inv·V_new) ✓), output 近似 (ds@(γr·γ_inv·V_new) 多乘 γr·γ_inv).
    γr·γ_inv[t] = exp(g_last - g[t]), 强衰减 case (g_last >> g[0]) 时远大于 1, output 误差 0.059 > RTOL.
  选 ungated V_new 堆叠: output 精确 (ds@V_new ✓), state 需后处理. 但 S_old 需 ×γr, Kᵀ@V_new 需 ×γr·γ_inv,
    两者混合在 state_o lower 无法分离后乘. 故 state 无法精确.
  ★ 根本: S_old 需 ×γr, Kᵀ@V_new 需 ×γr·γ_inv, 缩放不同 (差 γ_inv per-token), 不能合并到一个 fragment 后统一乘.
    唯一精确方式: 2 GEMM 分开 (ds@V_new ungated + Kᵀ@(gated V_new)), 即原 matw, 无堆叠收益.
结论: ★ idea 10.1 的 [ds; Kᵀ]@V_new 堆叠在 GDN 上数学不可行 (gating 矛盾).
  ds@V_new 和 Kᵀ@V_new 的 V_new gating 不同 (ungated vs ×γr·γ_inv), 共享操作数必然有一方近似.
  state 跨 chunk 累积故必须精确 → 只能牺牲 output, 但 output 近似误差 0.059 > RTOL.
  这条路在 GDN 上是死路 (除非接受 output 近似, 但 RTOL 不允许).
  ★ idea 10.2 (A@[βγK|βV] 堆叠 W/U) 无此矛盾: W=A@βγK, U=A@βV 都用 ungated βγK/βV, 共享 A, 可精确堆叠.
```

## 16.4 Phase G2: A@[βγK|βV] -> [W|U] 横向堆叠 (2026-08-20)

```text
commit: (uncommitted, _gdn_stackwu_matw_kernel in tilelang_fwd.py)
env switch: GDN_STACKWU=1
shape/probe: M=64, N=DK+block_DV=256, K=64 (A[64,64]@B_wu[64,256]->WU[64,256])
  B_wu[64,256] = [βγK(64×128) | βV(64×128)] 横向拼接
  WU_fragment[64,256] = [W(64×128) | U(64×128)], 拆分后 T.copy 到 W_shared/bv_shared
compile: OK (DV=64 path; DV=128 path shared 239KB > 232KB 超限, 仅 DV=64)
precision (8 cases): 全 PASS
  short_tail 0.153, chain 0.630, parallel_equal 0.469, parallel_gva 0.477,
  long_low 4.711, batch_split 2.880, wide 6.018, deep 5.723
t100 (8 cases) vs matw 基线 (DV=128/th=256 for 大 grid):
  chain_equal:    0.630ms (vs matw 0.538, +17% 慢)  [DV=64 path, 同 matw chain 配置]
  long_low_gva:   4.711ms (vs matw 3.078, +53% 慢)  [DV=64 vs matw DV=128, TC 效率损失]
  wide_gva_state: 6.018ms (vs matw 3.77, +60% 慢)   [DV=64]
  deep_gva_state: 5.723ms (vs matw 4.46, +28% 慢)   [DV=64]
  batch_split:    2.880ms (vs matw 2.26, +27% 慢)   [DV=64]
threads/block_DV/num_stages: th=128, DV=64, st=2 (仅 DV=64, DV=128 shared 超限)
是否比 matw 快: 否. 全线退步, 长序列 +28~60%.
失败原因:
  (1) DV=128 shared 超限 (239KB > 232KB): B_wu[64,256]=32KB + WU_fragment[64,256] FP32=64KB
      + 其余 shared (Q/K/V/A/W/ds/V_new/s + gates) 超限. 无法用大 tile.
  (2) DV=64 path TC 效率损失: 长 grid case (long_low/wide/deep/batch) DV=64 比 DV=128 慢 1.5-2x
      (memory 教训 v5: DV=32 令 TC 效率下降, DV=64 同理劣于 DV=128). 堆叠省 1 GEMM 的收益
      被 DV 降级的 TC 效率损失吃光还倒贴.
  (3) WU_fragment[64,256] 拆分需 2 次 T.copy (W→shared, U→shared), 抵消堆叠省的 fragment→shared.
  (4) N=256 wgmma 指令序列变长, 但 M=64 仍单 warpgroup, 无额外并行度.
结论: ★ idea 10.2 (A@[βγK|βV] 堆叠) 精确但性能退步.
  堆叠本身数学正确 (无 GDN gating 矛盾), 但代价是 B_wu[64,256] + WU_fragment[64,256] 增 shared/寄存器,
  DV=128 超限被迫降 DV=64, TC 效率损失 >> 省 1 GEMM 收益.
  ★ 与 WY-O F1 融合实验 (慢 2.6x) 同类失败: 拼接 ew + 大 fragment + 拆分 copy 开销 > 省 GEMM.
  核心教训: MIG shared 232KB 限制下, 任何把 [64,128] 扩到 [64,256] 的堆叠都会挤掉 DV=128,
  而 DV=128 是长序列性能的关键 (TC 效率). 堆叠换 DV 降级是净负.
```



```text
commit: (uncommitted, _gdn_async_matw_kernel in tilelang_fwd.py)
env switch: GDN_ASYNC=1
shape/probe: matw 数学不变, 仅把 T.gemm 换 T.wgmma_gemm+wait_wgmma(0) 批量调度
  Batch A (4 并发 state-free): W=A@bkg, U=A@bv, QK=Q@Kᵀ, QS=Q@S_old → 4 wgmma → wait(0)
  Batch B (依赖 W): WS=W@S_old → 1 wgmma → wait(0)
  Batch C (依赖 V_new, 累加): O+=ds@V_new, s+=Kᵀ@V_new → 同步 T.gemm (clear_accum=False 需 fence accum)
compile: OK
precision (8 cases): 全 PASS
  short_tail_state PASS, chain_equal PASS, parallel_equal PASS, parallel_gva PASS,
  long_low_gva PASS, batch_split_gva PASS, wide_gva_state PASS, deep_gva_state PASS
t100 (8 cases) vs matw 基线:
  chain_equal:    0.865ms (vs matw 0.538, +61% 慢)
  long_low_gva:   3.081ms (vs matw 3.078, +0.1% 持平)
  wide_gva_state: 3.878ms (vs matw 3.77, +2.9% 略慢)
  deep_gva_state: 4.563ms (vs matw 4.46, +2.3% 略慢)
threads/block_DV/num_stages: 同 matw (chain DV=64/th=128/st=2; 大 grid DV=128/th=256/st=2)
是否比 matw 快: 否. chain 慢 61%, 长序列持平/略慢.
失败原因:
  (1) chain (小 grid) 慢 61%: async wgmma 在 grid=8 (DV=64) 下 launch/同步开销 > 批量发射收益.
      小 grid CTA 少, wgmma 并发度不足填充 SM, wait(0) 同步点反而成瓶颈.
  (2) 长序列持平: Batch A 的 4 wgmma 并发本应压缩 3 次同步为 1 次, 但 DV=128/th=256 下
      4 个 wgmma 同时占用 accum fragment (tmp_dv/tmp_dv2/ds_tmp/O_fragment), 寄存器压力激增,
      HW OoO 重叠有限 (单 WG 下 wgmma 仍串行 issue, wait 前不能消费), 收益被抵消.
  (3) Batch C 必须同步: clear_accum=False 累加需 T.Parallel gate (O=γ⊙QS, s*=γr) 对 accum 可见,
      wgmma async 不 fence accum 寄存器 → 累加进 gate 前旧值 → 精度崩. 故 C 不能 async.
结论: ★ idea.md 第一主路线 (纯 Async Batch) 证伪. 单 WG 下 async wgmma 无收益:
  - 小 grid: 同步开销主导 (chain +61%)
  - 大 grid: 寄存器压力 + HW OoO 不可控抵消并发收益 (长序列持平)
  - 累加 GEMM 不能 async (accum 寄存器对 T.Parallel 不可见)
  与 WY-O async 证伪一致: tilelang 单 WG 下 wgmma async 无法兑现 GEMM↔ew/同步压缩收益,
  需 4-WG 多 consumer 并行才能藏 wgmma 延迟 (但 4-WG 不可行).
  ★ 核心教训: async wgmma 的收益前提是多 WG (多 consumer 并行 issue+wait), 单 WG 下
    issue 仍串行, wait 同步点反而增加开销. 这条路在 4-WG 不可行时是死路.
```

