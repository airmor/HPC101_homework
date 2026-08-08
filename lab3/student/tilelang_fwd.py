# GDN Prefill 前向 — 朴素 V_new 形式 fused kernel (阶段 A, 正确性优先)
#
# 结构: 单 fused kernel, state 跨 chunk 驻留 shared/fragment 不写回 global,
#       外层 chunk 循环用 T.Pipelined 重叠 load↔compute。
# V_new 被复用三次 (造 O_in / 造 S_new), 与 references/torch_gdr.py 逐行对应。
#
# 正确性要点 (对照 reference 校验):
#   1. scale = 128**-0.5
#   2. γr = g_cumsum[start + ℓ - 1]  (尾块取最后【有效】token, 不是 index 63)
#   3. 尾块 Q/K/V/β 补零; A padding 列已是单位行 (kkt_solve 保证)
#   4. gate 用差值 exp2((γr-g)*log2e), 不用比值 γr/γ (尾块 padding 会 nan)
#   5. GVA: bhg = bh // (Hv//Hq), index 映射不展开
#   6. O 用 S_old (更新 S 之前算 O); output BF16, final_state FP32

import torch
import tilelang
import tilelang.language as T

CHUNK_SIZE = 64
HEAD_DIM = 128
LOG2E = 1.4426950408889634


@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_naive_kernel(B, S, Hq, Hv, DK, DV, block_DV, threads, num_stages):
    block_S = CHUNK_SIZE
    num_chunks = (S + block_S - 1) // block_S
    G = Hv // Hq

    QK_shape = (B, S, Hq, DK)
    V_shape = (B, S, Hv, DV)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    init_shape = (B, Hv, DK, DV)
    O_shape = (B, S, Hv, DV)
    final_shape = (B, Hv, DK, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(QK_shape, dtype=T.bfloat16),
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        V: T.Tensor(V_shape, dtype=T.bfloat16),
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        initial_state: T.Tensor(init_shape, dtype=T.float32),
        O: T.Tensor(O_shape, dtype=T.bfloat16),
        final_state: T.Tensor(final_shape, dtype=T.float32),
    ):
        # block 映射: dv 维 × (batch*Hv)。每 block 持 (b, hv, dv_slice), state [128, block_DV] 驻留。
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G   # GVA: 这个 value head 用的 q/k head

            # ---- state: shared (GEMM 操作数) + fragment (累加器) ----
            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            # ---- 当前 chunk 数据 ----
            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            # ---- state-free 中间量 ----
            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)   # βγK
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)  # βV
            # W = A@(βγK): 先算进 fragment, 再 copy 到 shared 供下一 GEMM 当操作数
            W_fragment = T.alloc_fragment((block_S, DK), dtype=T.float32)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # U = A@(βV)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            # ---- gate ----
            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)   # γr = exp(g_last * log2e)

            # ---- ds = Lower(QKᵀ) ⊙ decay  [64,64] ----
            ds_fragment = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)   # BF16 供 GEMM 操作数

            # ---- 输出累加器 ----
            O_st_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            O_in_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
            })
            T.use_swizzle(10)

            # ---- 初始化 state ----
            # wrapper 保证 initial_state 非 None (None 时传零张量), 这里统一 copy
            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            # ================ chunk 递推循环 ================
            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)   # 尾块有效长度 ℓ

                # ---- 1. load Q/K/V/A/g/beta (尾块补零) ----
                for t, d in T.Parallel(block_S, DK):
                    if left + t < S:
                        Q_shared[t, d] = Q[bb, left + t, bhg, d]
                        K_shared[t, d] = K[bb, left + t, bhg, d]
                    else:
                        Q_shared[t, d] = 0
                        K_shared[t, d] = 0
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        V_shared[t, d] = V[bb, left + t, bh, bv * block_DV + d]
                    else:
                        V_shared[t, d] = 0
                for t, d in T.Parallel(block_S, block_S):
                    if left + t < S:
                        A_shared[t, d] = A[bb, left + t, bh, d]
                    else:
                        A_shared[t, d] = 1 if t == d else 0   # padding 行单位行
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                # γr = g_cumsum[left + ℓ - 1]
                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                # ---- P1: state-free 计算 ----
                # 2. βγK = K ⊙ β ⊙ γ
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32)
                        * beta_shared[t]
                        * T.exp2(g_shared[t] * LOG2E),
                        T.bfloat16,
                    )
                # W = A @ βγK  (A [64,64] × βγK [64,128] -> W [64,128])
                T.gemm(A_shared, bkg_shared, W_fragment, clear_accum=True)
                T.copy(W_fragment, W_shared)   # fragment -> shared, 供 W@S 用

                # 3. βV = V ⊙ β,  U = A @ βV
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t],
                        T.bfloat16,
                    )
                T.gemm(A_shared, bv_shared, U_fragment, clear_accum=True)

                # 4. ds = Lower(QKᵀ) ⊙ exp(g_i - g_j)
                T.gemm(Q_shared, K_shared, ds_fragment, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_fragment[i, j] = ds_fragment[i, j] * T.exp2(
                            (g_shared[i] - g_shared[j]) * LOG2E
                        )
                    else:
                        ds_fragment[i, j] = 0
                T.copy(ds_fragment, ds_shared)   # FP32 fragment -> BF16 shared, 供 ds@V_new GEMM

                # ---- P2: 递推 (依赖 S_old) ----
                # 5. V_new = U - W @ S_old
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, V_new_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    V_new_fragment[t, d] = U_fragment[t, d] - V_new_fragment[t, d]

                # ---- P3: 输出 (用 S_old, 在更新 S 之前!) ----
                # 6. O_st = scale * γ ⊙ (Q @ S_old)
                T.gemm(Q_shared, s_shared, O_st_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_st_fragment[t, d] = (DK ** -0.5) * T.exp2(g_shared[t] * LOG2E) * O_st_fragment[t, d]

                # 7. O_in = scale * (ds @ V_new)
                T.copy(V_new_fragment, V_new_shared)
                T.gemm(ds_shared, V_new_shared, O_in_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_in_fragment[t, d] = (DK ** -0.5) * O_in_fragment[t, d]

                # 8. O = O_st + O_in, 写回 (尾块只写前 ℓ 行)
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            O_st_fragment[t, d] + O_in_fragment[t, d], T.bfloat16
                        )

                # ---- P2 续: 更新 state ----
                # 9. gate V_new: V_new *= exp(γr - g)
                for t, d in T.Parallel(block_S, block_DV):
                    V_new_fragment[t, d] = V_new_fragment[t, d] * T.exp2(
                        (g_last_local[0] - g_shared[t]) * LOG2E
                    )
                T.copy(V_new_fragment, V_new_shared)
                # 10. S *= γr
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                # 11. S += Kᵀ @ V_new
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True)

            # ---- epilogue: 存 final_state ----
            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN prefill 前向 (朴素 V_new 形式)。"""
    batch_size, num_tokens, num_heads_qk, head_dim_k = q.shape
    _, _, num_heads_v, head_dim_v = v.shape

    # initial_state 为 None 时, 传零张量 (TileLang kernel 参数不能为 NULL)
    if initial_state is None:
        initial_state = torch.zeros(
            (batch_size, num_heads_v, head_dim_k, head_dim_v),
            dtype=torch.float32, device=q.device,
        )

    # block_DV: dv 切片大小 (split-dv 的杠杆, 见 LEARN 第 7 章)。
    # 按 B*Hv 并行度选: 高并行度用大 block_DV (少 split, 省重复 load);
    # 低并行度用小 block_DV (多 split, 提 SM 占用)。
    # MIG 10G 只有 14 SM, 阈值按实测调。
    parallelism = batch_size * num_heads_v
    if parallelism >= 64:
        block_DV = head_dim_v            # 不 split (128)
    elif parallelism >= 32:
        block_DV = min(64, head_dim_v)
    elif parallelism >= 16:
        block_DV = min(32, head_dim_v)
    else:
        block_DV = min(32, head_dim_v)    # 极低并行度, 最细 split
    threads = 128
    num_stages = 2

    kernel = _gdn_naive_kernel(
        batch_size, num_tokens, num_heads_qk, num_heads_v,
        head_dim_k, head_dim_v, block_DV, threads, num_stages,
    )
    output, final_state = kernel(q, k, v, g_cumsum, beta, A, initial_state)
    return output, final_state
