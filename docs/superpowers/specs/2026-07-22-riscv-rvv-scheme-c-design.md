# Lab 2 RISC-V RVV 重写设计（方案 C）

**日期：** 2026-07-22  
**范围：** `lab2/student/moe_opt.cpp`；仅实现 RISC-V V（RVV）路径，暂不使用 SpaceMiT IME。

## 目标

在不改变 `MoEWeights`、`preprocess`/`moe_forward_optimized` 公共接口以及 MoE 数值语义的前提下，将现有 x86 AVX-512 VNNI 实现改写为 RVV 实现。原文件中的解释性注释保留；不在当前 Windows 主机执行性能或功能测试。

## 数据布局

`preprocess` 生成三类转置缓存：

1. Router：`[d_model][num_experts]`，使同一个输入维度对应的一组专家权重连续，RVV 可在 expert 维上执行 fp32 FMA。
2. gate/up：`[d_model][d_ff]`，使输入维度固定时的一组 FFN 输出通道权重连续，RVV 可在输出通道维上执行 int32 MAC。
3. down：`[d_ff][d_model]`，使用同一模式计算输出模型维度。

量化权重仍保持 signed int8；方案 C 不使用 VNNI 所需的 uint8 偏置编码，也没有 IME tile packing。

## RVV 计算路径

- Router：对一段 experts 加载 fp32 权重并累加 `w * x[d]`；sigmoid、Top-K 选择仍采用标量操作以严格保留原语义。
- FFN 投影：将一段 signed int8 权重扩展为 signed int32，并用 `vmacc.vx` 对一个 signed int8 activation 标量累加，得到 int32 gate/up/down 输出。
- 非线性和动态量化：保留 `expf`、`lrintf` 与标量 max-reduction，避免引入近似 exp 或改变 rounding，从而降低 Top-K、量化边界与参考实现产生不一致的风险。
- 合并：使用 RVV 处理 `x + shared_out` 和 `y + gate * routed_out`。

## 非 RISC-V 回退

当没有定义 `__riscv_vector` 时，使用相同的转置数据布局和标量 helper。这让宿主机能够做基础构建检查；RISC-V 测试机编译时由 CMake 的 `-march=rv64gcv` 分支启用 RVV。

## 并行与验证

按 token 的 OpenMP 并行保留，每次迭代只写一行 `y`，不产生写竞争。目标机应依次执行普通 correctness、`num_tokens` 非向量长度倍数的 case、以及 benchmark；在方案 C 通过后，再独立接入 IME 4x4x8 微内核。
