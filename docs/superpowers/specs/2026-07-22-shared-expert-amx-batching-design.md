# 共享专家多 Token AMX 批处理设计

**日期：** 2026-07-22  
**范围：** Lab 2 的 `lab2/student/moe_opt.cpp`；仅批处理共享专家，路由专家继续采用逐 token 路径。

## 目标

恢复并优化 AMX 路径中的共享专家计算。共享专家对所有 token 使用相同权重，因此按最多 16 个 token 为一组执行 AMX 矩阵乘法，提高 tile 的列维利用率，并保持现有 MoE 的数值语义：

```text
y_t = x_t + shared_ffn(x_t) + sum_k(normalized_gate[t][k] * routed_ffn(expert[t][k], x_t))
```

## 非目标

- 不改变 router、Top-K 选择或 gate 归一化算法。
- 不对路由专家按 expert 聚合、分桶或批处理。
- 不修改框架、reference 实现或输入/权重的数据格式。
- 不为了批处理改变每 token 动态量化策略。

## 当前问题

当前 `IS_AMX=1` 路径没有预打包 shared gate/up/down 权重，且 `moe_forward_optimized` 在 AMX 条件分支中只执行路由专家累加，遗漏了共享专家及残差初始化。因此输出未满足规定的 MoE 公式。

## 方案

### 阶段 A：每 Token 预处理

使用 OpenMP 并行遍历 token，完成：

1. router affinity 与 sigmoid；
2. Top-K 选择及 gate 归一化；
3. 输入的每 token 动态量化，写入 `xq_total[t][d]`；
4. 保存 `s_x_total[t]`、`topk_idx_total[t][k]` 与 `gate_total[t][k]`。

该阶段保持现有 router 和量化结果语义。将路由元数据从后续共享专家计算中解耦。

### 阶段 B：共享专家 AMX 批处理

以连续 token block 执行并行循环，每块大小：

```text
B = min(16, num_tokens - block_begin)
```

对每个 block：

1. 将 `B` 个 token 的 token-major `xq_total` 重排为 AMX `dpbusd` 的 activation tile 布局：K 维每 4 个 signed int8 值组成一行，列方向为 token；布局形状为 `K/4 × (B*4)` 字节。
2. 用共享 `w_sh_gate` 和 `w_sh_up` 对同一 activation tile 做 AMX `dpbusd`，计算 `d_ff × B` 的 gate/up 累加值。权重保持 `uint8 = int8 + 128` 编码；对每个 token 的结果加 `-128 * sum(xq_t)` 补偿。
3. 按 token 分别反量化 gate/up，计算 `SiLU(gate) * up`，并分别统计该 token 的 hidden maximum absolute value。
4. 每个 token 以独立 `s_h` 量化 hidden activation；将量化后的 hidden 重排为 `d_ff/4 × (B*4)` 的 AMX activation tile。
5. 用共享 `w_sh_down` 执行 AMX down projection，按 token 加 `-128 * sum(hq_t)` 补偿、反量化后得到 `o_shared[t]`。
6. 直接写出 `y[t][d] = x[t][d] + o_shared[t][d]`。

`B < 16` 的尾块使用匹配实际 B 的 tile 配置与重排，不读取或写入越界内存。

### 阶段 C：路由专家累加

使用 OpenMP 按 token 并行。复用已有逐 token `expert_ffn`（AVX-512 VNNI）计算每一个已选专家，并执行：

```text
y[t] += gate_total[t][k] * routed_output
```

阶段 C 不重新计算 router、Top-K 或输入量化。

## AMX Tile 配置

每个共享 batch 都配置以下 tile（K 分块为 64）：

| Tile | 用途 | 行数 | 每行字节数 |
|---|---|---:|---:|
| `tmm0` | gate/down accumulator | 16 | `B * 4` |
| `tmm1` | up accumulator | 16 | `B * 4` |
| `tmm2` | gate/down weight tile | 16 | 64 |
| `tmm3` | up weight tile | 16 | 64 |
| `tmm4` | packed activation tile | 16 | `B * 4` |

AMX configuration is per-thread architectural state. Each OpenMP worker that executes a shared block must call the tile-config load routine before tile operations. The routine accepts `B` from 1 through 16.

## 权重预处理

`preprocess` 必须为 shared gate/up/down 分配并填充 packed `uint8` 权重，格式与 routed expert 的 VNNI packing 保持一致：输出通道以 16 为一组，K 维以 4 为一组。AMX 和非 AMX 路径都复用这些共享权重缓冲区。

## 并行与存储约束

- 不在热路径中动态分配内存。
- 批处理函数使用有界局部缓冲区，最大尺寸由 `MAX_D_MODEL=1024`、`MAX_D_FF=512`、`B=16` 确定。
- 各 OpenMP iteration 写入不重叠 token 的临时数据和 `y` 行，不产生数据竞争。
- 不在每次 forward 中调用全局 `omp_set_num_threads`；使用外部线程设置或 OpenMP runtime 默认设置，以免干扰基准环境。

## 回退行为

在没有 AMX 编译/运行支持的情况下，保留逐 token 的共享专家实现：先计算 shared expert 并写入 `x + shared`，随后累加路由专家。该路径也必须满足完整 MoE 公式。

## 验证

1. 在小、中、大尺寸及 `num_tokens` 非 16 倍数的输入上运行默认 correctness check。
2. 覆盖 `top_k` 为 1 到 4 的配置。
3. 使用 `--benchmark` 观察优化实现的性能，重点对比共享专家批处理前后的吞吐。
4. 验证 AMX 和非 AMX 构建路径都能编译；AMX 路径仅在 Sapphire Rapids 目标机运行。

## 验收标准

- AMX 路径的输出通过框架逐元素正确性检查。
- 输出包含残差、共享专家和所有选择的路由专家。
- 任意合法 token 数（包括非 16 倍数）均能执行。
- 共享专家 gate、up 和 down 均使用多 token AMX batch 计算。
- 路由专家继续保持逐 token 实现，未引入 expert 分桶逻辑。
