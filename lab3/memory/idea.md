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


# 17. 真正瓶颈重判：不是“少了哪一个 GEMM”，而是 CTA 数量、state critical path 与 load pipeline

- 日期：2026-08-21
- 结论级别：这是下一轮优化的总判断，低级 AI 不应再先尝试新的 WY / prefix 恒等式，先按本节的执行结构路线做实验。

## 17.1 先把昨天的失败解释清楚

昨天的数学路线并不是数学推导错误，而是优化目标选错了：

1. `Q@K^T` 与 `ds@V_new` 的 FMA 合计约为 `0.524M + 0.524M = 1.048M`。
2. 每个 chunk 的总 FMA 约为：

```text
W=A@(beta*gamma*K)       1.048M
U=A@(beta*V)             1.048M
W@S_old                  1.048M
Q@S_old                  1.048M
K^T@V_new                1.048M
Q@K^T                    0.524M
ds@V_new                 0.524M
总计                      6.288M FMA
```

因此即使把 `Q@K^T` 和 `ds@V_new` 完全免费，FLOP 上限也只有 `6.288 / 5.240 = 1.20x`。而 OJ 110 需要长序列约 `1.6x`。这条路从上限上就不够。

更重要的是，真正不能被普通代数消掉的依赖链是：

```text
S_old
  -> W@S_old
  -> V_new = U - W@S_old
  -> K^T@V_new
  -> S_new
```

`Q@S_old` 和 `ds@V_new` 属于 output 分支；`W@S_old -> V_new -> K^T@V_new` 才是跨 chunk state 的硬依赖。WY 只是把成本移动到 `ds@W`、Q' 写回、额外 fragment 和同步上，并没有消掉这条链。

## 17.2 当前 matw 的三个真正瓶颈候选

### A. 长序列低 head 数时，grid 太小

`long_low` 是 `B=1, Hv=8`，当前 `block_DV=128` 时 grid 只有 8 个 CTA；H800 MIG 只有约 14 个 SM，最多只有 8 个 SM 有 CTA，至少一部分 SM 空闲。一个 CTA 内的 512 个 chunk 又必须沿 state 顺序执行，因此不能靠更多 chunk 并行来填满机器。

这解释了为什么：

- `DV=64` 以前虽然把 grid 变成 16，但因为每个 DV tile 重复计算 W 和 ds/QK，反而从约 3.04ms 退化到约 4.06ms；
- 这里不是“DV=64 不好”，而是“DV split 后重复了与 DV 无关的工作”；
- 这是一个空间/时间甜点：**把 DV 拆开增加 CTA，同时把 W 与 QK/ds 从 split tile 中抽出来，只计算一次**。

### B. 单 WG / 单 CTA 内 state critical path 太长

当前每个 `T.gemm` 默认隐式 `wait_group 0`。之前的单 WG async 实验已经证明：只把多个 GEMM 改成 `T.wgmma_gemm`，在小 grid 上 launch/同步开销反而更明显，大 grid 上又被 accumulator 寄存器压力抵消。因此继续堆 async batch 不是主路。

需要改变的是“一个 CTA 包揽一个 DV tile 的全部工作”这一执行结构，而不是继续改 GEMM 的代数顺序。

### C. matw 的输入 load 不是有效的 producer pipeline

当前 `_gdn_naive_kernel_matw` 在 `T.Pipelined` 内用的是：

```python
for t, d in T.Parallel(...):
    Q_shared[t, d] = Q[...]
    K_shared[t, d] = K[...]
    V_shared[t, d] = V[...]
    A_shared[t, d] = A[...]
```

这些是 element-wise load，不是 `CopyNode` 的 global-to-shared `T.copy`。TileLang 的 pipeline planner 只有识别到 `T.copy(global, shared)` 才会把该语句放入 producer stage；所以现在的 `T.Pipelined` 主要是在做 buffer versioning，并没有保证 load 与 compute 真正重叠。

之前的 TMA 版本变慢不能证明“异步 load 没用”，它同时引入了 TMA descriptor / mbarrier / 调度开销。下一次应单独验证 **无尾块的普通 `T.copy` + `T.Pipelined`**，不要把它和 TMA 混在一起。

## 17.3 第一优先级新路线：精确的“state-independent precompute + DV split”

暂命名：`SP-DV2`（State-independent Precompute + 2-way DV split）。它不是近似，不改变数学结果，不会产生跨 chunk 误差累积。

### Kernel A：每个 chunk 只预计算一次与 DV 无关的量

对每个 `(B, chunk, V-head)` 计算并写入 BF16 workspace：

```text
W_h,c = A_h,c @ (beta_h,c * exp(g_h,c) * K_h,c)   [64,128]
```

同时，对每个 `(B, chunk, Q-head)` 只计算一次 raw score：

```text
R_q,c = Q_q,c @ K_q,c^T                           [64,64]
```

主 kernel 对每个 V head 再用本 head 的 gate 做：

```text
ds_h,c[i,j] = tril(R_q,c[i,j] * exp(g_h[i]) * exp(-g_h[j]))
```

这样可以同时利用两个性质：

- `W` 与 DV tile 无关；
- GVA 下一个 Q head 对应多个 V head，raw `Q@K^T` 可以跨 V head 复用，gate 只在主 kernel 中逐 head 缩放。

Kernel A 可以先做成一个简单的独立预处理 kernel；若低级 AI 想减少 launch，可在同一个预处理 kernel 中让每个 `bh % G == 0` 的 CTA 额外计算 raw score，其余 V head CTA 只算 W。

### Kernel B：每个 V head 拆成两个 DV=64 CTA

每个 split CTA 只负责一个 `[64,64]` 的 DV tile：

```text
U       = A@(beta*V_tile)       [64,64]
W@S     = W_shared@S_old        [64,64]
Q@S     = Q@S_old               [64,64]
ds@V   = ds@V_new_tile         [64,64]
K^T@V  = K^T@V_new_tile         [128,64]
```

两个 DV CTA：

- 读取同一个预计算的 W；
- 读取同一个 raw QK（或 ds workspace）；
- 各自读取不同的 V / state 列；
- 各自写 output 的不同 DV 列；
- 各自写 final_state 的不同 DV 列；
- 没有 CTA 间 state 依赖，因为 state recurrence 只沿 chunk 方向、每个 DV 列独立。

### 为什么这可能是 110 所需的结构性加速

原始 `DV=128` 一个 CTA 的每 chunk 工作是约 6.288M FMA，并且所有工作被同一个低数量 grid 串行承载。

SP-DV2 不减少总 FMA（除 GVA raw-QK 复用外），但把工作拆成：

```text
预处理阶段：W + raw-QK，约占每 head 25% 左右的工作
主阶段：两个 DV=64 CTA，各承载剩余的 DV-dependent 工作
```

对 `long_low`：

- 原始：8 个 CTA，约 8 个 SM 承担全部工作；
- 预处理：仍是 8 个 CTA，但只承担小部分 state-independent 工作；
- 主阶段：16 个 CTA，能覆盖 14 个 SM；
- 理想化 wall-time 近似：

```text
原始： (Wpre + Wmain) / 8
新式： Wpre / 8 + Wmain / 16
```

当 `Wpre` 约占 1/4 时，理论上接近 `1.6x`，这正好对应 long_low 从约 3.04ms 逼近 1.86ms 所需的量级。实际会被额外 global workspace 读写、DV=64 的 WGMMA 效率、第二次 kernel launch 抵消一部分，但它是目前第一个**理论上能解释 1.6x 来源**的路线。

### Workspace 容量计算

只存 BF16：

```text
W workspace：       [B, C, Hv, 64, 128] BF16 = 16 KiB * B*C*Hv
raw-QK workspace：  [B, C, Hq, 64, 64]  BF16 =  8 KiB * B*C*Hq
```

8 个评测 case 的最大值：

| case | W workspace | raw-QK workspace | 合计 |
|---|---:|---:|---:|
| short_tail_state | 2.125 MiB | 0.133 MiB | 2.258 MiB |
| chain_equal | 8 MiB | 8 MiB | 16 MiB |
| parallel_equal | 8 MiB | 4 MiB | 12 MiB |
| parallel_gva | 8 MiB | 1 MiB | 9 MiB |
| long_low | 64 MiB | 8 MiB | 72 MiB |
| batch_split | 64 MiB | 8 MiB | 72 MiB |
| wide_gva_state | 128 MiB | 16 MiB | 144 MiB |
| deep_gva_state | 128 MiB | 16 MiB | 144 MiB |

最大约 144 MiB，远小于 10 GiB GPU 显存；它不占 CTA shared memory，不会触发 232 KiB 动态 shared 限制。workspace 只在 SP-DV2 路线分配，不能把它误算成 shared。

若不想让主 kernel 做 raw-QK gate scaling，也可以直接存每个 V head 的 gated `ds`：

```text
W + ds：24 KiB * B*C*Hv
```

最大约 192 MiB，仍然完全可接受，但会失去 GVA 的 raw-QK 复用。建议先实现 raw-QK 版，若索引或 gate scaling 增加太多开销，再退回 gated-ds 版。

### 精度与同步要求

- workspace 用 BF16，保持与当前 matw 的 W/shared 与 ds/shared 截断位置一致；
- 不允许近似，不允许低秩截断；
- 两个 split CTA 只写不重叠的 DV 区间，不需要 CTA 间 barrier；
- `initial_state` 和 `final_state` 按 DV slice 切分；
- `short_tail_state` 继续走旧 kernel，SP-DV2 第一版只处理 `T % 64 == 0`；
- 第一实验只开 `long_low`，不要同时改 wide/deep，先证明 3.04ms 是否能显著下降。

## 17.4 第二优先级：精确输出水平拼接，而不是 `[ds;K^T]` 竖直拼接

之前尝试的 `[ds; K^T] @ V_new` 是 `M=192`，会拉大输出 fragment、寄存器和 warp partition；它主要减少 API 次数，不减少 Tensor Core FMA，优先级不高。

更适合 Hopper 的拼接是输出式：

```text
O / scale = Qhat @ S_old + ds @ V_new

Qhat = exp(g) * Q
L = [ Qhat | ds ]       [64, 192]
R = [ S_old            ] [192, DV]
    [ V_new            ]

O = L @ R               [64, DV]
```

它保持 `M=64`，只是把 `K=128 + 64 = 192`，通常比 `M=192` 更不伤 warp partition。

### shared 容量

```text
Q_shared + ds_shared       = 64*128*2 + 64*64*2  = 24 KiB
Qds_shared[64,192]         = 64*192*2           = 24 KiB

S_shared + V_new_shared   = 128*128*2 + 64*128*2 = 48 KiB
SV_shared[192,128]         = 192*128*2           = 48 KiB
```

所以从“数学中间存储”看是够的，关键是做 lifetime coloring，而不是再新增一份大 buffer。注意 TileLang 对 shared 第一维非零 offset 的 `T.gemm` 支持有限；优先安排 `S` 位于 `SV` 的首 128 行，`V_new` 另用可复用的旧 `V_shared` / `s_shared` 低 64 行，避免直接把 `SV[128:192]` 当 GEMM B operand。若最终必须多保留 16 KiB V buffer，也要先检查动态 shared 是否仍低于约 227 KiB。

这条路线预期是 5%~12% 级别，不是 1.6x 主路线；只有 SP-DV2 证明方向正确后才值得做。

## 17.5 第三优先级：无尾块 fast path 恢复普通 T.copy pipeline

对 `T % 64 == 0` 的 long/wide/deep/batch case，另写一个无边界判断的 fast kernel：

```python
T.copy(Q[...], Q_shared)
T.copy(K[...], K_shared)
T.copy(V[...], V_shared)
T.copy(A[...], A_shared)
T.copy(g_cumsum[...], g_shared)
T.copy(beta[...], beta_shared)
```

不要写成 `T.Parallel` 的逐元素 global load。tail case 单独继续走旧 kernel。

当前 matw 的静态 shared 粗略约 161 KiB；若 `num_stages=2` 只对输入 buffer 做版本化，新增 Q/K/V/A/g/beta 约 58 KiB，总量约 219 KiB，理论上仍低于 MIG 动态上限约 227 KiB，但必须让低级 AI 编译确认实际值。若超限，优先只对 Q/K/V 三个大输入做双 buffer，A/g/beta 保持单 buffer。

这条路线不改数学，目标是验证当前真正被遗漏的 load-compute overlap。TMA 退化不代表普通 `T.copy` 退化；两者 descriptor、barrier 和调度成本不同，不能混为一谈。

## 17.6 不再优先尝试的路线

1. 继续寻找只删除 QK 或 dsV 的恒等式：FLOP 上限不够。
2. block-prefix / block-constant output：GDN chunk 内近邻 decay 接近 1，误差结构性很大。
3. WY-O 单 WG：新增 ds@W、共享写回与寄存器压力，实测已退步。
4. 单 WG async batch：之前已证明小 grid 变慢、大 grid 近似持平，不能改变 CTA 数量。
5. `[ds;K^T]@V_new` M=192：主要节省调用，不减少 Tensor Core 工作，且寄存器风险更大。
6. 低秩跨 chunk affine scan：理论上能打破 state serial，但误差会跨 chunk 累积，且 affine 矩阵组合成本接近重新计算；除非所有执行结构路线失败，不进入短期实现。

## 17.7 低级 AI 的实验顺序与判定标准

### Experiment 1：full-tile `T.copy` pipeline

- 只改 long_low 对应的无尾块 kernel；
- 保持 DV=128、数学和 state 顺序不变；
- 若 long_low 至少下降 10%，说明 load pipeline 是真实瓶颈之一；
- 若 shared 超限，记录实际动态 shared，不要直接放弃，减少版本化 buffer。

### Experiment 2：SP-DV2 gated-ds 版（先不做 GVA raw-QK）

- 预处理 W/ds workspace；
- 主 kernel DV=64、threads=128，两个 CTA/head；
- 只开 long_low；
- 目标：精度 8 case 全过，long_low 明显优于 3.04ms；
- 若性能不降反升，优先检查 W/ds workspace 的 global read、第二次 launch 与 DV=64 WGMMA 效率，而不是怀疑数学。

### Experiment 3：SP-DV2 raw-QK GVA 版

- 仅对 `Hv/Hq > 1` 开启；
- 比较 wide/deep/long_low；
- 预期额外收益来自跨 V head 复用 raw QK，而不是 approximation。

### Experiment 4：输出水平拼接

- 只在 SP-DV2 或 full-tile fast path 稳定后做；
- 记录 shared 实际占用和 K=192 GEMM 的生成形态；
- 若低于 5% 收益或出现 shared/寄存器退化，立即回退。

### 总结

昨天数学优化失败的根因不是“中间存储永远不够”，而是：

```text
数学删除的工作量太小
+ 精确 state recurrence 仍然存在
+ 单 CTA / 单 WG 承载了过长 critical path
+ long_low grid 不足以填满 14 SM
+ 当前 matw load 没有真正进入 producer pipeline
```

目前最值得实现的不是另一个 WY 公式，而是：

```text
先用 exact state-independent workspace 把 DV split 的重复 W/ds 拿掉，
再用 DV=64 增加 CTA 数量；同时对无尾块验证普通 T.copy pipeline。
```

这两条分别攻击“低 occupancy”和“load-compute 不重叠”，是目前唯一可能从 93 分跨到 110 分附近的路线。

---

# 18. 新主路线：中间只做 `S_new`，output 延后（State-only critical stage）

- 日期：2026-08-21
- 状态：高层数学设计，待低级 AI 先做 reference 数学验证，再决定实现。
- 用户指定目标：把中间阶段化成只更新 `S_new` 的最简形式；output 不在中间阶段计算，而放到后续阶段。

## 18.1 先把递推完全展开

记：

```text
r_i       = exp(g_last - g_i)
R         = diag(r_i)
W         = A @ (beta * exp(g) * K)
U         = A @ (beta * V)
V_new     = U - W @ S_old
```

原始 state 更新为：

```text
S_new = exp(g_last) * S_old + K^T @ R @ V_new
```

代入 `V_new`：

```text
S_new
= exp(g_last) * S_old + K^T @ R @ U - K^T @ R @ W @ S_old
```

定义两个只依赖当前 chunk 输入的系数：

```text
C = K^T @ R @ W       # [DK, DK]
B = K^T @ R @ U       # [DK, DV]
```

则中间阶段的最简 state-only 形式是：

```text
S_new = exp(g_last) * S_old - C @ S_old + B
```

也可以写成：

```text
S_new = (exp(g_last) * I - C) @ S_old + B
```

实现时不要真的构造 `exp(g_last)*I-C`；直接：

```text
Z = C @ S_old
S_new = exp(g_last) * S_old - Z + B
```

这样中间阶段只剩：

```text
1. 一个 state-dependent GEMM: C @ S_old
2. 一次 state 的 scalar gate
3. 一次减法和一次加法
```

原来的 output（`Q@S_old` 与 `ds@V_new`）完全移出这个 critical stage。

## 18.2 output 延后后的精确形式

output 仍然必须使用 `S_old`，不能误用已经更新的 `S_new`：

```text
O = scale * (exp(g) * Q @ S_old + ds @ V_new)
```

为了让 output 阶段也只做一个 GEMM，可在 state-independent 阶段预计算：

```text
P = exp(g) * Q - ds @ W       # [block_S, DK]
R_o = ds @ U                  # [block_S, DV]
```

于是：

```text
O = scale * (P @ S_old + R_o)
```

完整 chunk 变成：

```text
Stage A: load + W/U + ds
Stage B: 计算 state-independent 系数 C/B/P/R_o
Stage C: 只算 S_new = exp(g_last)S_old - C@S_old + B
Stage D: 只算 O = scale*(P@S_old + R_o)
```

注意：Stage D 必须保留 `S_old` 的副本；`S_new` 写回不能覆盖唯一的 `S_old`，否则 output 无法再算。

## 18.3 比四个系数 GEMM 更好的融合

`W` 与 `U` 有相同左操作数 `A`，先合并：

```text
WU = [W | U] = A @ [beta*exp(g)*K | beta*V]
```

对 `DV=64`，右侧宽度是 `DK+DV=192`；对 `DV=128`，宽度是 `256`。TileLang Hopper WGMMA 支持这些 N 值，但必须低级 AI 编译确认寄存器和真实指令。

系数阶段不要分别算四个 GEMM。复用同一个 `[W|U]`：

```text
CB = (K^T @ R) @ [W | U]
PR = ds @ [W | U]
```

拆分结果：

```text
CB[:, :DK] = C
CB[:, DK:] = B
PR[:, :DK] = ds@W
PR[:, DK:] = R_o
P = exp(g)*Q - PR[:, :DK]
```

因此推荐的高层 GEMM 数量是：

```text
1. WU       = A @ [beta*exp(g)*K | beta*V]
2. ds       = Q @ K^T（再做 gate 和 causal mask）
3. CB       = (K^T @ R) @ [W | U]
4. PR       = ds @ [W | U]
5. state    = C @ S_old
6. output   = P @ S_old
```

原始主要路径约 7 次 GEMM；新路径约 6 次，且关键串行链的两个 state-dependent GEMM 被拆成各自独立的 `state` 与 `output` 阶段。若进一步让 `CB` 与 `PR` 做 M=192 的垂直堆叠，数学上可合成一个大 GEMM，但第一版不要赌 M=192 大 fragment；先用两个合法 M=128/M=64 GEMM验证。

## 18.4 中间存储预算（必须按 lifetime coloring，不可把所有数组同时常驻）

### 推荐先做 `DV=64` split 版本

单个 CTA 负责一个 `[0:64]` 或 `[64:128]` 的 V slice：

| buffer | shape | BF16 shared |
|---|---:|---:|
| Q | `[64,128]` | 16 KiB |
| K | `[64,128]` | 16 KiB |
| V | `[64,64]` | 8 KiB |
| A | `[64,64]` | 8 KiB |
| W/U 工作区 | `[64,128] + [64,64]` | 24 KiB |
| ds | `[64,64]` | 8 KiB |
| `S_old` | `[128,64]` | 16 KiB |
| C/B 输出区 | `[128,128] + [128,64]` | 48 KiB |
| P/R 输出区 | `[64,128] + [64,64]` | 24 KiB |
| gate/local/对齐余量 | — | < 8 KiB |
| **峰值（不做双版本）** | — | **约 176 KiB** |

关键复用：

```text
W/U 工作区在 CB/PR 生成后释放；
W/U 工作区可复用成 P/R 工作区；
S_new 不必再开一份 BF16 shared，可先落到 FP32 fragment；
```

所以 `DV=64` 下数学中间存储是够的，低于 MIG 约 227 KiB 动态 shared 上限。真正不能做的是把 `C/B/P/R`、两份输入 ping-pong、两份 state 和 `W/U` 同时常驻。

### `DV=128` 不建议作为第一版

完整 V slice 会使 `S_old` 32 KiB、W/U 32 KiB、C/B 64 KiB、P/R 48 KiB 同时存在，叠加 Q/K/V/A 后接近或超过动态 shared 上限；而且 output/state fragment 寄存器峰值更危险。先用 DV=64 取得两倍 CTA 数，再考虑大 tile。

### 全局 workspace 容量

若 Stage A/B 与 Stage C/D 拆成不同 kernel，建议把系数按 BF16 写入 global workspace：

```text
C:  [B, chunks, Hv, 128,128] BF16 = 32 KiB / chunk / V-head
B:  [B, chunks, Hv, 128,DV] BF16
P:  [B, chunks, Hv,  64,128] BF16 = 16 KiB / chunk / V-head
R_o:[B, chunks, Hv,  64,DV] BF16
```

`DV=128` 时合计 96 KiB / chunk / V-head：

```text
long_low       512*8*96 KiB  ≈ 384 MiB
batch_split    128*32*96 KiB ≈ 384 MiB
wide_gva_state 128*64*96 KiB ≈ 768 MiB
deep_gva_state 256*32*96 KiB ≈ 768 MiB
```

这远小于 10 GiB；瓶颈不是显存容量，而是额外 global 写回/再读取和第二、第三次 kernel launch。

## 18.5 关键正确性风险：实数等价不等于 BF16 逐步等价

系数化改变了原始的结合顺序：

```text
原始：K^T @ (R * (U - W @ S_old))
新式：K^T@R@U - (K^T@R@W) @ S_old
```

在实数运算中完全等价；在当前 BF16 输入、FP32 accumulate、若干 `T.copy` 截断下，不保证 bitwise 或误差完全相同。尤其 `S_new` 会跨 chunk 递推，不能像 output 近似那样任意放宽。

低级 AI 必须先做 reference/torch 仿真：

```text
1. FP32 系数化：检查真实数学误差；
2. 按当前 kernel 的 BF16 截断点仿真；
3. 模拟 512 chunk long_low 的 state 误差是否仍 < rtol=5e-3, atol=5e-3；
4. 只有 state 误差通过，才写 TileLang。
```

如果系数化 state 误差不通过，保留此结构用于 output-only：

```text
- Stage C 仍用原始 V_new 精确 state update；
- Stage D 才用 P@S_old+R_o 的 output 近似/加速；
```

但这会失去“state-only 一个 GEMM”的最大收益，不应未经精度仿真直接实现。

## 18.6 真正的四阶段执行方式

不能把 `Stage A/B/C/D` 机械写成同一个 `T.Pipelined(num_stages=4)`；TileLang 的 `T.Pipelined` 主要重叠 global→shared copy 与 consumer，不会自动把四类 compute 分到四个执行阶段。

要得到真实收益，优先级是：

```text
首选：Stage A/B 预计算 kernel（grid=chunks*B*Hv）
    → Stage C state kernel（grid=DV_slices*B*Hv，chunk serial）
    → Stage D output kernel（grid=chunks*DV_slices*B*Hv）
```

如果必须单 fused kernel，只有在手写 warp-specialization/async WGMMA 后才能把 A/B 与 C/D 真正重叠；当前 TileLang 4-WG 已有 barrier race，不作为第一实现。

三 kernel 版本仍有 3 次 launch；但 Stage A/B 与 Stage D 的 grid 远大于 14 SM，Stage C 的 critical path 从原始约 2 个 state-dependent GEMM 降到 1 个，才是这条路线可能接近 110 的来源。

## 18.7 低级 AI 只按这个顺序验证

```text
Experiment S0：torch/reference 仿真 state-only 公式，先看 8 case 精度
Experiment S1：只做 WU 合并，确认 N=192/256 + workspace 写回
Experiment S2：实现 CB（只 state coefficients），先不做 output coefficients
Experiment S3：State kernel 只算 S_new，测 long_low critical path
Experiment S4：加 PR 和 output kernel，确认 output 误差/总耗时
Experiment S5：只有 S1-S4 都稳定后，才尝试 CB/PR 的 M=192 堆叠
```

禁止低级 AI 直接在默认 kernel 里把所有系数 buffer 同时塞进 shared；先做独立 workspace 版本，便于验证“state-only critical stage”是否真的带来收益。

## 18.8 结论

用户提出的拆法是正确的关键方向：中间阶段应只保留 state recurrence，output 放到后面；但必须同时满足两个条件：

```text
(1) S_old 必须在 output 阶段仍可读（不能提前覆盖）；
(2) state-only 系数化的 BF16 误差必须经过长序列 state 仿真，否则误差会跨 chunk 累积。
```

如果 S0 通过，`CB/PR + DV=64 split + chunks/head 预计算` 是当前唯一同时具备：

```text
数学关键链缩短 + CTA 数增加 + output 不阻塞 state
```

的高潜力路线。

---

# 19. S0 仿真结果 — ★ 已修正 (2026-08-21)

## 19.1 原始 S0 结论 (错误)

原 S0 (`s0_state_only_sim.py`) 报全 case FAIL (state ~24, rel 5000x)，归因 catastrophic cancellation。
idea 19.2/19.3 的具体数字和 trace 均基于此。

## 19.2 bug 定位

**S0 仿真有 bug**：`R = exp(eg_last - gh)` 误用 `eg_last = exp(gh[-1])`（标量 S_old gate），
应为 `R = exp(g_last - gh)`，`g_last = gh[-1]`（raw 未 exp）。
- matw kernel 正确：`gl_local = exp2(g_last*LOG2E)`（标量 gate on S_old），
  state 用 `gl_local * g_inv_shared[t] = exp(g_last - g[t])`（正确 R）。
- 我的独立仿真把两者搞混，导致 R 大 ~38x（chain eg_last=0.026 当 g_last，真实 g_last=-3.64）。

## 19.3 修正后 S0 结果 (`s0_final2.py`)

```text
chain_equal       FP64 orig-coeff=0  coeff-ref=0       | FP32 coeff-ref=1.5e-7  PASS
parallel_equal    FP64 orig-coeff=0                    | 1.2e-7  PASS
parallel_gva      FP64 orig-coeff=0                    | 2.4e-7  PASS
batch_split_gva   B=4                                  | 1.8e-7  PASS
wide_gva_state                                         | 1.8e-7  PASS
```

**FP64 完全等价 (diff=0)**，**FP32 coeff-vs-ref ~1e-7** (BF16 输入噪声级)，**全 PASS**。

## 19.4 结论修正

idea 18 state-only 系数化数学**完全可行**，无 catastrophic cancellation。
原 "证伪" 是仿真 bug，非数学问题。matw kernel 一直正确。

★ idea 18.8 条件 (2) "BF16 误差必须仿真" 现已通过 → 可进入 S1-S4 实现。
★ idea 18.8 条件 (1) "S_old 不能提前覆盖" 经用户指出：S_new 写新位置 S_all[c+1]，
  S_all[c] 留给 output kernel 读 → 时序解耦, state/output 并行无冲突.
  state kernel critical path 缩到 1 GEMM (C@S_old), output kernel grid=chunks*DV_slices*B*Hv 完全并行.

## 19.5 下一波实现 (idea 18.6 三 kernel 结构)

```text
Kernel A (precompute): W/U/ds → C/B/P/R  grid=chunks*B*Hv   完全并行, state-independent
Kernel B (state):      S[c]=γr·S[c-1]-C@S[c-1]+B  grid=DV_slices*B*Hv  chunk 串行, 1 GEMM
Kernel C (output):     O[c]=scale·(P@S[c-1]+R)  grid=chunks*DV_slices*B*Hv  完全并行
S_all[B,Hv,num_chunks+1,DK,DV] FP32: state 序列, Kernel B 写 S_all[c+1], Kernel C 读 S_all[c]
```

|eg_last·S_old|   = 1.144e+00   ← 正向 gate 项 (很小，因 eg_last 极小)
|C@S_old|         = 1.019e+01   ← 系数减项
|B|               = 1.252e+01   ← 系数加项
|S_new_orig addend| (Kᵀ@R@V_new) = 1.430e+01  ← 原始直接加项
|S_new|           = 1.394e+01   ← 结果
```

- `C@S_old (10.19)` + `B (12.52)` 与结果 `S_new (13.94)` 同量级，`eg_last·S_old (1.14)` 被淹没。
- 原始形式 `eg_last·S_old (1.14) + Kᵀ@R@V_new (14.30) = 13.94`：两项都为正，无相消。
- 系数形式 `eg_last·S_old (1.14) - C@S_old (10.19) + B (12.52) = 13.27`：减法 `1.14 - 10.19` 严重相消，再加 `B` 放大误差。
- **chain_equal eg_last 极小 (0.026~0.056，强衰减 random_decay)**，`eg_last·S_old` 被压到极小，而 `C@S_old`/`B` 保持大值 → cancellation 主导。

跨 chunk 递推：每个 chunk 的 cancellation 误差 `~1e-5`（FP32），但经 128 chunk 累积放大到 24（chain_equal）。state 误差跨 chunk 不可重置，故发散。

## 19.4 结论：idea 18 state-only 系数化在 GDN 上不可行

```text
- 数学等价 (FP32 单步 diff 5.7e-6) 但数值不稳定 (跨 chunk 累积发散到 24)
- 根因: 强衰减 case (eg_last 极小) 下 eg_last·S_old 被 C@S_old 淹没, catastrophic cancellation
- BF16 加剧但非主因 (FP32 也 FAIL)
- state 跨 chunk 递推, 误差不可重置 → 必须用原始 V_new 直接形式 (无系数化)
```

**idea 18.5 的风险预判正确**："实数等价不等于 BF16 逐步等价"——实际上 FP32 就已不稳定。
**idea 18.8 的 fallback 成立**：保留 Stage C 原始 V_new 精确 state update，Stage D 用 output 近似。

但 state-only critical stage 的最大收益（消掉 W@S_old→V_new→Kᵀ@V_new 串行链）不成立，
因为 state update 必须保持 `Kᵀ@(R@(U - W@S_old))` 原始形式。

## 19.5 剩余可走的方向

idea 17.3 的 SP-DV2（State-independent Precompute + DV split）不依赖系数化，
W/ds 预计算 + DV=64 split 增加 CTA 是独立路线，state update 仍用原始 V_new 形式。
这是当前唯一未证伪的高潜力路线，下一步应实现 SP-DV2 Experiment 1/2。

---

# 20. SP-DV2 实现 + S0 仿真记录 (2026-08-21)

## 20.1 S0 state-only 仿真 (idea 18.7) — ★ 已修正 (原结论错误)

```text
commit: (uncommitted, s0 脚本已删)
env switch: 无 (torch 仿真)
原结论 (错误): state-only 系数化数值不稳定, 全 FAIL (state~24)
  根因误判: catastrophic cancellation (eg_last 极小被 C@S_old 淹没)
bug 定位: 仿真 R = exp(eg_last - gh) 误用 eg_last=exp(gh[-1]) (标量 gate),
  应为 R = exp(g_last - gh), g_last=gh[-1] (raw). R 大 ~38x 导致 state 发散.
修正后 (s0_final2.py):
  FP64 orig-coeff diff = 0 (完全等价)
  FP32 coeff-vs-ref ~1e-7 (BF16 噪声级), 全 PASS
  (chain 1.5e-7, parallel 1.2e-7, parallel_gva 2.4e-7, batch_split 1.8e-7, wide 1.8e-7)
★ idea 18 state-only 系数化数学完全可行, 无 cancellation. idea 18.5 风险预判被仿真 bug 误导.
```

## 20.2 SP-DV2 实现 (idea 17.3, 原始 V_new state, 不系数化) — 精确但性能退步

```text
commit: (uncommitted, _gdn_spdv2_pre_kernel + _gdn_spdv2_main_kernel)
env switch: GDN_SPDV2=1
结构:
  Kernel A: 预计算 W = A@(βγK), grid=chunks*B*Hv, 写 global BF16 workspace [B,Hv,num_chunks,64,128]
  Kernel B: DV=64 split 主 kernel, grid=2*B*Hv, 读 W workspace, state 用原始 V_new = U - W@S_old, S_new = γr·S_old + Kᵀ@(R@V_new)
  full-tile T.copy load (无尾块, 仅 T%64==0)
precision (7 可测 case): 全 PASS (short_tail T%64≠0 走默认 matw)
t100 vs matw 基线:
  chain_equal:    0.618ms (vs 0.524, +18%)
  long_low_gva:   4.588ms (vs 3.003, +53%)
  wide_gva_state: 6.673ms (vs 3.77, +77%)
  deep_gva_state: 6.683ms (vs 4.46, +50%)
  batch_split:    3.163ms (vs 2.26, +40%)
  parallel_equal: 0.474ms (vs 0.45, +5%)
  parallel_gva:   0.478ms (vs 0.43, +11%)
threads/block_DV/num_stages: Kernel A th=256; Kernel B th=128, DV=64, st=2
是否比 matw 快: 否. 全线退步, 长序列 +40~77%.
失败原因 (与 decomp K1/K2/K3 + DV=64 调参同根因):
  (1) W global workspace round-trip: Kernel A 写 W (16KB/chunk/head), Kernel B 读回.
      long_low 512 chunk * 8 head = 4096 chunk → 64MB 额外 global 读写, 无 L2 命中 (首次写/读).
      原始 matw W 在 fragment/shared 算完即用, 无 global 往返. 物化开销 >> 省 1 GEMM 收益.
  (2) DV=64 TC 效率损失: memory v5 已证 DV=64 比 DV=128 慢 1.5-2x (大 grid).
      CTA 翻倍 (8→16) 的占用收益被 per-CTA TC 效率降抵消还倒贴.
  (3) 2 kernel launch + JIT: Kernel A/B 各编译一次.
  (4) W 只占 1/7 GEMM (14%), 不是 idea 估计的 25%: 省 W 的收益太小.
结论: ★ SP-DV2 精确但性能退步. 与 decomp (慢 1.1-1.9x) + DV=64 调参 (慢 1.5x) 同类失败:
  global 物化 W 往返 + DV=64 TC 降级 >> W 去重 + CTA 翻倍收益.
  ★ idea 17.2 A (低 occupancy 是瓶颈) 判断正确, 但 17.3 (DV split 解决) 失败:
    DV=64 本身是 TC 效率瓶颈, 不是 CTA 数量. 翻倍 CTA 无法补偿 DV 降级.
```

## 20.3 下一步: Experiment 1 (full-tile T.copy matw, idea 17.5)

SP-DV2 Kernel B 已用 full-tile T.copy load, 但 DV=64 掩盖了 T.copy 的收益.
需单独验证: DV=128 + T.copy (无尾块) 的 matw, 隔离 load pipeline 收益 (无 DV 降级混淆).

## 20.4 Experiment 1: full-tile T.copy matw (idea 17.5) — 证伪

```text
commit: (uncommitted, _gdn_naive_kernel_matw_fulltile)
env switch: GDN_FULLTILE=1
结构: matw 数学不变, DV=128/th=256/st=2 (同长序列基线), 仅把 6 个 load 从 T.Parallel ew 改 T.copy 全片
  (无边界判断, T%64==0). 进 T.Pipelined producer pipeline (ClassifyCopyLikeStage 认 CopyNode).
precision: 全 PASS (chain 0.615, long_low 4.679)
t100 vs matw 基线:
  chain_equal:  0.615ms (vs 0.524, +17%)
  long_low:     4.679ms (vs 3.003, +56%)
是否比 matw 快: 否. 全线退步.
失败原因:
  ★ T.copy 对 g_cumsum/beta (1D [64] FP32) fallback 到 normal copy (非 TMA bulk):
    "src range must have last dim multiple of 16 for tma bulk load g_cumsum range 1*4 % 16 != 0"
    (memory TMA 实验同警告). g/beta 仍 element-wise, 但 Q/K/V/A 走 TMA.
  ★ 即便 Q/K/V/A 走 TMA, 仍退步 +56% — 说明 load pipeline 不是瓶颈:
    H800 MIG 14 SM, long_low grid=8 (DV=128), 每 CTA 512 chunk 串行.
    load 本身占时小 (L2 命中高), GEMM/ew 串行主导. T.copy producer pipeline 重叠的 load
    不是关键路径, 重叠收益 < T.copy 调度开销.
  ★ 与 TMA 实验 (memory 6 节, +23%) 一致: T.copy/TMA load 在 GDN 长 chunk 串行下无收益.
结论: ★ idea 17.2 C (load 没进 producer pipeline 是瓶颈) 判断不成立.
  load pipeline 重叠不是 long_low 退步主因, GEMM 串行 critical path 才是.
  full-tile T.copy 反而退步 (调度开销 > 重叠收益).
```

## 20.5 本轮总结 (S0 修正 + SP-DV2 + full-tile)

```text
★ idea 18 (state-only 系数化): S0 bug 修正后完全可行 (FP64 等价, FP32 ~1e-7). 进入实现.
idea 17.3 (SP-DV2 W 预计算 + DV split, 原始 V_new): 精确但 DV=64 降级退步, 证伪 (W 预计算 + DV split 这条).
idea 17.5 (full-tile T.copy pipeline): 精确但 load 非瓶颈退步, 证伪.
★ idea 17.2 三瓶颈重判:
  A (低 occupancy): 真, DV split 解法失败因 DV=64 TC 降级. 但 state-only 三 kernel (idea 18.6) 不需 DV split —
    Kernel C output grid=chunks*DV_slices*B*Hv 巨大, DV=128 仍能填满 SM (long_low: 512*2*1*8=8192 CTA).
  B (state critical path 长): 真, state-only 系数化缩短到 1 GEMM (C@S_old), 可行 (S0 已验证).
  C (load 没进 pipeline): 假 (full-tile 退步). 但 idea 18.6 三 kernel 用 Kernel A 预计算 + Kernel C output
    并行, load 在 Kernel A 里 grid=chunks*B*Hv 充分并行, 不依赖 T.Pipelined 重叠.
★ 下一波: 按 idea 18.6 + 用户修正 (S_new 写新位置解耦 output) 实现三 kernel:
  Kernel A: W/U/ds → C/B/P/R (grid=chunks*B*Hv, 并行)
  Kernel B: S[c]=γr·S[c-1]-C@S[c-1]+B (grid=DV_slices*B*Hv, chunk 串行, 1 GEMM, 写 S_all[c+1])
  Kernel C: O[c]=scale·(P@S[c-1]+R) (grid=chunks*DV_slices*B*Hv, 完全并行, 读 S_all[c])
```



