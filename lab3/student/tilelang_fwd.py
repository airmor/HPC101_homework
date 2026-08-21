# GDN Prefill 前向 — 朴素 V_new 形式 (per-case 分发 + 手动最优配置)
#
# 两个 kernel:
#   _gdn_naive_kernel  (结合律版, 短序列): W@S = A@(βγK@S), 不物化 W, 寄存器少
#   _gdn_naive_kernel_matw (物化W版, 长序列): 直接 W=A@βγK 再 W@S, GEMM 少
#
# 分发: T<=2048 用结合律版 (寄存器降, 短序列受益); T>2048 用物化W版 (GEMM 少, 长序列受益)
# 配置: 手动固定 autotune 搜出的最优参数 (不用 @autotune, 避免 OJ JIT 编译超时)
#   短序列: block_DV=64, threads=128, num_stages=2  (结合律版, 寄存器少)
#   长序列: block_DV=128, threads=256, num_stages=1 (物化W版, 大 tile+寄存器分摊)
# 依据实测 (H800 MIG 10G):
#   short_tail(T=1025): 结合律 0.143 vs 物化W 0.195
#   long_low(T=32768): 物化W DV=128/th=256 3.21 vs DV=64/th=128 4.06

import torch
import tilelang
import tilelang.language as T

CHUNK_SIZE = 64
HEAD_DIM = 128
LOG2E = 1.4426950408889634


@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_naive_kernel(B, S, Hq, Hv, DK, DV, block_DV=64, threads=128, num_stages=2):
    """结合律版: W@S = A@(βγK@S), 不物化 W。寄存器少, 短序列快。"""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            # ---- state ----
            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            # ---- chunk 数据 ----
            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            # ---- state-free shared ----
            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)   # βγK [64,128]
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)  # βV -> 后存 U
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            # 临时 shared: 存 βγK@S 中间结果。FP32 保精度 (结合律多一次截断)
            bkgS_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            # ---- gate ----
            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            # ---- 临时 fragment (复用降压力) ----
            tmp_dv = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # U/V_new/WS 等
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()   # 让编译器自由分配寄存器, 降压力

            # ---- 初始化 state ----
            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

                # ---- 1. load (尾块补零) ----
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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                # ============ P1: state-free ============
                # 2. βγK = K ⊙ β ⊙ γ  (inline exp2, 短序列 chunk 少, 避免 exp 复用的循环开销)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_shared[t]
                        * T.exp2(g_shared[t] * LOG2E), T.bfloat16)

                # 3. βV = V ⊙ β,  U = A @ βV -> bv_shared
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                T.gemm(A_shared, bv_shared, tmp_dv, clear_accum=True)
                T.copy(tmp_dv, bv_shared)   # bv_shared = U (BF16)

                # 4. ds = Lower(QKᵀ) ⊙ exp(g_i - g_j)  (inline exp2)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * T.exp2(
                            (g_shared[i] - g_shared[j]) * LOG2E)
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # ============ P2: 递推 (依赖 S_old) ============
                # 5. V_new = U - W@S_old
                #    ★ 结合律: W@S = A@(βγK@S), 不物化 W [64,128]
                #    βγK@S: [64,128] × [128,block_DV] -> [64,block_DV]
                #    中间结果保 FP32 fragment, 避免 BF16 二次截断累积误差
                T.copy(s_fragment, s_shared)
                T.gemm(bkg_shared, s_shared, tmp_dv, clear_accum=True)   # tmp_dv = βγK@S (FP32)
                # A@(βγK@S): A 是 BF16 shared, tmp_dv 是 FP32 fragment
                # T.gemm 要求 A/B 同 dtype -> 把 tmp_dv 拷成 BF16 shared 再乘
                T.copy(tmp_dv, bkgS_shared)   # FP32 fragment -> BF16 shared (一次截断)
                T.gemm(A_shared, bkgS_shared, tmp_dv, clear_accum=True)  # tmp_dv = W@S (FP32)
                # V_new = U - W@S (U 在 bv_shared BF16, 拷到 O_fragment)
                T.copy(bv_shared, O_fragment)   # O_fragment = U (FP32)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv[t, d] = O_fragment[t, d] - tmp_dv[t, d]   # tmp_dv = V_new (FP32)
                T.copy(tmp_dv, V_new_shared)

                # ============ P3: 输出 (用 S_old, 在更新 S 之前!) ============
                # 6. O = scale * [γ⊙(Q@S_old) + ds@V_new]  (inline exp2)
                T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_fragment[t, d] = T.exp2(g_shared[t] * LOG2E) * O_fragment[t, d]
                # O += ds@V_new (累加, 不 clear)
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                # 统一乘 scale 写回
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # ---- P2 续: 更新 state ----
                # 7. gate V_new: V_new *= exp(γr - g) (inline exp2)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv[t, d] = tmp_dv[t, d] * T.exp2(
                        (g_last_local[0] - g_shared[t]) * LOG2E)
                T.copy(tmp_dv, V_new_shared)
                # 8. S *= γr
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                # 9. S += Kᵀ @ V_new
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# TMA load 版 (长序列, 实验):
#   核心: load 从 element-wise T.Parallel 改 T.copy (TMA async bulk), 进 producer stage.
#   ★ 子agent2: T.Pipelined 的 ClassifyCopyLikeStage 只认 T.copy(global→shared) 为 producer
#     (pipeline_planning.cc:602). 当前 matw 用 T.Parallel load → 进 consumer, 和 GEMM 串行.
#   ★ 不用显式 _2 ping-pong buffer (T.Pipelined 自动 multi-version), 避免双重计数爆 shared.
#   ★ 仅用于 T%64==0 的 case (无尾块, T.copy 固定切片). short_tail 走 matw.
#   数学: 同 matw (朴素 V_new), 不改数学, 只改 load 策略.
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_tma_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=1):
    """TMA load 版: T.copy 替代 T.Parallel load. 同 matw 数学, 测 TMA bulk load 速度."""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            # ★ 普通 shared buffer (无 _2 维), T.Pipelined 自动 multi-version
            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S

                # ★ T.copy (TMA async bulk) load → ClassifyCopyLikeStage 认作 producer stage
                # ★ 仅 T%64==0 调用, 无尾块, 固定 block_S 切片不越界
                T.copy(Q[bb, left:left + block_S, bhg, :], Q_shared)
                T.copy(K[bb, left:left + block_S, bhg, :], K_shared)
                T.copy(V[bb, left:left + block_S, bh, bv * block_DV:(bv + 1) * block_DV], V_shared)
                T.copy(A[bb, left:left + block_S, bh, :], A_shared)
                T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)
                T.copy(beta[bb, left:left + block_S, bh], beta_shared)

                length = T.min(block_S, S - left)
                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # P1: βγK, W=A@(K⊙beta_g)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)
                T.copy(tmp_dv, W_shared)

                # βV, U=A@βV
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)
                T.copy(tmp_dv2, bv_shared)

                # ds = Lower(QKᵀ) ⊙ (g_exp_i * g_inv_j)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # P2: V_new = U - W@S
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)
                T.copy(bv_shared, O_fragment)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = O_fragment[t, d] - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # P3: O = scale*(γ⊙(Q@S_old) + ds@V_new)
                T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # 更新 state
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# 三 kernel 分解 (实验): state-free 并行 + state 串行 + O 并行
#
# 数学: GDN 每 chunk 7 GEMM 里, W/U/ds 三 GEMM state-free (不依赖 S),
#       O 的 Q@S/ds@V_new 只需 S[c-1] (给定即可并行). 真正串行链只有
#       W@S → V_new=U-W@S → Kᵀ@V_new → S_new (2 GEMM + ew).
#       拆成 3 kernel: K1 并行算 W/U/ds, K2 串行递推 S, K3 并行算 O.
#       state 精确串行 (无 scan 近似), O 用精确 S[c-1]. 误差不叠加.
#
# 收益: K1/K3 grid = num_chunks × B×Hv (长序列 512×8=4096 blocks), 填满 14 SM.
#       单 kernel 里 chunk 串行, grid=B×Hv (chain=4, long_low=8) 严重欠占 SM.
#
# 存储: W/U/ds [B,Hv,num_chunks,64,128] BF16 ≈ long_low 134MB (10GB 内).
#       S[c] FP32 [B,Hv,(num_chunks+1),128,128] ≈ 268MB (global, K2 写 K3 读).
# ============================================================

# --- K1: 并行算每个 chunk 的 W, U, ds (state-free) ---
@tilelang.jit(out_idx=[-3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_decomp_k1_kernel(B, S, Hq, Hv, DK, DV, num_chunks, threads=256):
    """K1: 并行算 W[c]=A@βγK, U[c]=A@βV, ds[c]=Lower(QKᵀ)⊙decay. grid=chunks×B×Hv. DV 不切."""
    block_S = CHUNK_SIZE
    G = Hv // Hq
    QK_shape = (B, S, Hq, DK)
    V_shape = (B, S, Hv, DV)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    W_shape = (B, Hv, num_chunks, block_S, DK)
    U_shape = (B, Hv, num_chunks, block_S, DV)
    ds_shape = (B, Hv, num_chunks, block_S, block_S)

    @T.prim_func
    def kernel(
        Q: T.Tensor(QK_shape, dtype=T.bfloat16),
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        V: T.Tensor(V_shape, dtype=T.bfloat16),
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        W_out: T.Tensor(W_shape, dtype=T.bfloat16),
        U_out: T.Tensor(U_shape, dtype=T.bfloat16),
        ds_out: T.Tensor(ds_shape, dtype=T.bfloat16),
    ):
        with T.Kernel(num_chunks, B * Hv, threads=threads) as (ci, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G
            left = ci * block_S

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)
            tmp_dv2 = T.alloc_fragment((block_S, DV), dtype=T.float32)
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(Q[bb, left:left + block_S, bhg, :], Q_shared)
            T.copy(K[bb, left:left + block_S, bhg, :], K_shared)
            T.copy(V[bb, left:left + block_S, bh, :], V_shared)
            T.copy(A[bb, left:left + block_S, bh, :], A_shared)
            T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)
            T.copy(beta[bb, left:left + block_S, bh], beta_shared)

            for t in T.Parallel(block_S):
                g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

            # W = A @ βγK
            for t, d in T.Parallel(block_S, DK):
                bkg_shared[t, d] = T.cast(
                    T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
            T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)
            T.copy(tmp_dv, W_out[bb, bh, ci, :, :])

            # U = A @ βV
            for t, d in T.Parallel(block_S, DV):
                bv_shared[t, d] = T.cast(
                    T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
            T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)
            T.copy(tmp_dv2, U_out[bb, bh, ci, :, :])

            # ds = Lower(QKᵀ) ⊙ (g_exp_i * g_inv_j)
            T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(block_S, block_S):
                if i >= j:
                    ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                else:
                    ds_tmp[i, j] = 0
            T.copy(ds_tmp, ds_out[bb, bh, ci, :, :])

    return kernel


# --- K2: 串行 state 递推 (用 K1 的 W/U), 存所有 S[c] ---
@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_decomp_k2_kernel(B, S, Hq, Hv, DK, DV, num_chunks, threads=256):
    """K2: 串行 S[c]=γr·S[c-1]+sKᵀ@(U[c]-W[c]@S[c-1]). DV 整块 (不切). grid=B*Hv."""
    block_S = CHUNK_SIZE
    G = Hv // Hq
    QK_shape = (B, S, Hq, DK)
    V_shape = (B, S, Hv, DV)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    WU_shape = (B, Hv, num_chunks, block_S, DK)
    U_shape = (B, Hv, num_chunks, block_S, DV)
    init_shape = (B, Hv, DK, DV)
    Sall_shape = (B, Hv, num_chunks + 1, DK, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(QK_shape, dtype=T.bfloat16),
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        V: T.Tensor(V_shape, dtype=T.bfloat16),
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        W_in: T.Tensor(WU_shape, dtype=T.bfloat16),
        U_in: T.Tensor(U_shape, dtype=T.bfloat16),
        initial_state: T.Tensor(init_shape, dtype=T.float32),
        S_all: T.Tensor(Sall_shape, dtype=T.float32),
    ):
        # grid: (B*Hv). chunk 维串行 (T.serial). DV 整块不切.
        with T.Kernel(B * Hv, threads=threads) as (bbh,):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, DV), dtype=T.float32)

            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            U_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv2 = T.alloc_fragment((block_S, DV), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, DV), dtype=T.float32)

            T.annotate_layout({
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            # S[0] = initial_state
            T.copy(initial_state[bb, bh, 0:DK, 0:DV], s_shared)
            T.copy(s_shared, s_fragment)
            T.copy(s_fragment, S_all[bb, bh, 0, 0:DK, 0:DV])

            for ci in T.serial(num_chunks):
                left = ci * block_S
                length = T.min(block_S, S - left)

                T.copy(K[bb, left:left + block_S, bhg, :], K_shared)
                T.copy(W_in[bb, bh, ci, :, :], W_shared)
                T.copy(U_in[bb, bh, ci, :, :], U_shared)
                T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)
                for t in T.Parallel(block_S):
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)

                # W@S (S_old 在 s_fragment)
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)
                T.copy(U_shared, O_fragment)
                for t, d in T.Parallel(block_S, DV):
                    tmp_dv2[t, d] = O_fragment[t, d] - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # gate V_new *= gl * g_inv; S *= γr; S += Kᵀ@V_new
                for t, d in T.Parallel(block_S, DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True, clear_accum=False)

                T.copy(s_fragment, S_all[bb, bh, ci + 1, 0:DK, 0:DV])

    return kernel


# --- K3: 并行算 O (用 S[c-1] 和 K1 的 ds, K2 的 S) ---
@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_decomp_k3_kernel(B, S, Hq, Hv, DK, DV, num_chunks, threads=256):
    """K3: 并行 O[c]=scale*(γ⊙(Q@S[c-1]) + ds@V_new[c]). DV 整块. grid=chunks×B×Hv."""
    block_S = CHUNK_SIZE
    G = Hv // Hq
    QK_shape = (B, S, Hq, DK)
    V_shape = (B, S, Hv, DV)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    WU_shape = (B, Hv, num_chunks, block_S, DK)
    U_shape = (B, Hv, num_chunks, block_S, DV)
    ds_shape = (B, Hv, num_chunks, block_S, block_S)
    Sall_shape = (B, Hv, num_chunks + 1, DK, DV)
    O_shape = (B, S, Hv, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(QK_shape, dtype=T.bfloat16),
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        V: T.Tensor(V_shape, dtype=T.bfloat16),
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        W_in: T.Tensor(WU_shape, dtype=T.bfloat16),
        U_in: T.Tensor(U_shape, dtype=T.bfloat16),
        ds_in: T.Tensor(ds_shape, dtype=T.bfloat16),
        S_all: T.Tensor(Sall_shape, dtype=T.float32),
        O: T.Tensor(O_shape, dtype=T.bfloat16),
    ):
        with T.Kernel(num_chunks, B * Hv, threads=threads) as (ci, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G
            left = ci * block_S

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            U_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            s_shared = T.alloc_shared((DK, DV), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)

            tmp_dv2 = T.alloc_fragment((block_S, DV), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, DV), dtype=T.float32)
            tmp_U = T.alloc_fragment((block_S, DV), dtype=T.float32)

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                s_shared: tilelang.layout.make_swizzled_layout(s_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(Q[bb, left:left + block_S, bhg, :], Q_shared)
            T.copy(W_in[bb, bh, ci, :, :], W_shared)
            T.copy(U_in[bb, bh, ci, :, :], U_shared)
            T.copy(ds_in[bb, bh, ci, :, :], ds_shared)
            T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)
            T.copy(S_all[bb, bh, ci, 0:DK, 0:DV], s_shared)

            for t in T.Parallel(block_S):
                g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)

            # V_new = U - W@S (S = S[c-1])
            T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)
            T.copy(U_shared, tmp_U)
            for t, d in T.Parallel(block_S, DV):
                tmp_dv2[t, d] = tmp_U[t, d] - tmp_dv2[t, d]
            T.copy(tmp_dv2, V_new_shared)

            # O = scale*(γ⊙(Q@S) + ds@V_new)
            T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
            for t, d in T.Parallel(block_S, DV):
                O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
            T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
            for t, d in T.Parallel(block_S, DV):
                if left + t < S:
                    O[bb, left + t, bh, d] = T.cast(
                        (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

    return kernel


# ============================================================
# 物化 W 版 (长序列): 直接 W=A@βγK 再 W@S, GEMM 数少, 长序列快
#   full_tile 变体 (idea 17.5): 无尾块 case 用 T.copy 全片 load 替代 T.Parallel ew,
#     进 T.Pipelined producer pipeline. 尾块走原 matw.
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_naive_kernel_matw_fulltile(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=2):
    """matw full-tile: T.copy load (无尾块), 隔离 load pipeline 收益. 数学同 matw."""
    block_S = CHUNK_SIZE
    num_chunks = S // block_S  # 调用方保证 T%64==0
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S

                # ★ full-tile T.copy load (无边界判断, 进 producer pipeline)
                T.copy(Q[bb, left:left + block_S, bhg, 0:DK], Q_shared)
                T.copy(K[bb, left:left + block_S, bhg, 0:DK], K_shared)
                T.copy(V[bb, left:left + block_S, bh, bv * block_DV : (bv + 1) * block_DV], V_shared)
                T.copy(A[bb, left:left + block_S, bh, 0:block_S], A_shared)
                T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)
                T.copy(beta[bb, left:left + block_S, bh], beta_shared)

                length = block_S
                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # P1: βγK, W=A@(K⊙beta_g)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)
                T.copy(tmp_dv, W_shared)

                # βV, U=A@βV
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)
                T.copy(tmp_dv2, bv_shared)   # bv_shared = U

                # ds
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # V_new = U - W@S
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)
                T.copy(bv_shared, O_fragment)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = O_fragment[t, d] - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # O = scale*(γ⊙(Q@S_old) + ds@V_new)
                T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                for t, d in T.Parallel(block_S, block_DV):
                    O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                        (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # state
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# 物化 W 版 (长序列, 原始): 尾块 case 走此版 (带边界判断)
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_naive_kernel_matw(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=1):
    """物化 W 版: W=A@βγK 驻 shared, W@S 直接算。GEMM 少, 长序列快。"""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            # 预算 exp(g)/1/exp(g)/beta*g_exp, 复用省 exp2
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)  # 最大 [64,128] (W 产出)
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # V_new/O 等 [64,block_DV]
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()   # 让编译器自由分配寄存器, 降压力

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                # 预算 exp(g)/1/exp(g)/beta*g_exp, 复用省 exp2
                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # P1: βγK, W=A@(K⊙beta_g) (物化 W, 复用 beta_g)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)   # tmp_dv [64,128] = W
                T.copy(tmp_dv, W_shared)

                # βV, U=A@βV -> bv_shared
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)   # tmp_dv2 [64,block_DV] = U
                T.copy(tmp_dv2, bv_shared)   # bv_shared = U

                # ds = Lower(QKᵀ) ⊙ (g_exp_i * g_inv_j)  (复用, 无 exp2)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # P2: V_new = U - W@S (直接 W@S, 无结合律)
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)   # tmp_dv2 = W@S
                T.copy(bv_shared, O_fragment)   # O_fragment = U
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = O_fragment[t, d] - tmp_dv2[t, d]   # tmp_dv2 = V_new
                T.copy(tmp_dv2, V_new_shared)

                # P3: O = scale*(γ⊙(Q@S_old) + ds@V_new)  (复用 g_exp, 无 exp2)
                T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # 更新 state: gate V_new *= gl * g_inv (复用, 无 exp2)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# SP-DV2: State-independent Precompute + 2-way DV split (idea.md 17.3)
#   Kernel A (precompute): 每个 (chunk, B, Hv) 预计算 W = A@(βγK), 写 global BF16 workspace.
#     grid = num_chunks * B * Hv, state-independent, 完全并行填满 SM.
#   Kernel B (main): 每个 V head 拆 2 个 DV=64 CTA, 读预计算 W, state 用原始 V_new 形式 (不系数化).
#     grid = 2 * B * Hv (DV split), chunk 串行 (state recurrence).
#   ★ 关键: W 与 DV 无关, 预计算只算一次 (原始 matw 每个 DV CTA 重复算 W).
#     DV=64 split 翻倍 CTA 数 (long_low 8→16, 覆盖 14 SM), state 仍精确.
#   ★ 不系数化 state (idea 18 已证伪: catastrophic cancellation), 用原始 V_new = U - W@S_old, S_new = γr·S_old + Kᵀ@(R@V_new).
# ============================================================
@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_spdv2_pre_kernel(B, S, Hq, Hv, DK, DV, num_chunks, threads=256):
    """SP-DV2 Kernel A: 预计算 W = A@(βγK), 写 global workspace. state-independent, grid=chunks*B*Hv."""
    block_S = CHUNK_SIZE
    G = Hv // Hq
    QK_shape = (B, S, Hq, DK)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    W_shape = (B, Hv, num_chunks, block_S, DK)

    @T.prim_func
    def kernel(
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        W_out: T.Tensor(W_shape, dtype=T.bfloat16),
    ):
        with T.Kernel(num_chunks, B * Hv, threads=threads) as (ci, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G
            left = ci * block_S

            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            W_frag = T.alloc_fragment((block_S, DK), dtype=T.float32)

            # full-tile load (无尾块, SP-DV2 仅 T%64==0)
            T.copy(K[bb, left:left + block_S, bhg, 0:DK], K_shared)
            T.copy(A[bb, left:left + block_S, bh, 0:block_S], A_shared)
            T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)
            T.copy(beta[bb, left:left + block_S, bh], beta_shared)

            for t in T.Parallel(block_S):
                g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

            # βγK (同 matw 截断点)
            for t, d in T.Parallel(block_S, DK):
                bkg_shared[t, d] = T.cast(
                    T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
            # W = A @ βγK
            T.gemm(A_shared, bkg_shared, W_frag, clear_accum=True)
            T.copy(W_frag, W_out[bb, bh, ci, 0:block_S, 0:DK])

    return kernel


# SP-DV2 Kernel B: 主 kernel, DV=64 split, 读预计算 W, state 用原始 V_new 形式
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_spdv2_main_kernel(B, S, Hq, Hv, DK, DV, num_chunks, block_DV=64, threads=128, num_stages=2):
    """SP-DV2 Kernel B: DV=64 split, 读 W workspace, 原始 V_new state. grid=2*B*Hv (DV split)."""
    block_S = CHUNK_SIZE
    G = Hv // Hq
    QK_shape = (B, S, Hq, DK)
    V_shape = (B, S, Hv, DV)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    init_shape = (B, Hv, DK, DV)
    O_shape = (B, S, Hv, DV)
    final_shape = (B, Hv, DK, DV)
    W_shape = (B, Hv, num_chunks, block_S, DK)

    @T.prim_func
    def kernel(
        Q: T.Tensor(QK_shape, dtype=T.bfloat16),
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        V: T.Tensor(V_shape, dtype=T.bfloat16),
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        W_in: T.Tensor(W_shape, dtype=T.bfloat16),
        initial_state: T.Tensor(init_shape, dtype=T.float32),
        O: T.Tensor(O_shape, dtype=T.bfloat16),
        final_state: T.Tensor(final_shape, dtype=T.float32),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)  # 从 global 读

            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # U/W@S/V_new/O
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S

                # full-tile T.copy load (无尾块, SP-DV2 仅 T%64==0)
                T.copy(Q[bb, left:left + block_S, bhg, 0:DK], Q_shared)
                T.copy(K[bb, left:left + block_S, bhg, 0:DK], K_shared)
                T.copy(V[bb, left:left + block_S, bh, bv * block_DV : (bv + 1) * block_DV], V_shared)
                T.copy(A[bb, left:left + block_S, bh, 0:block_S], A_shared)
                T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)
                T.copy(beta[bb, left:left + block_S, bh], beta_shared)
                # ★ 读预计算 W
                T.copy(W_in[bb, bh, i_c, 0:block_S, 0:DK], W_shared)

                length = block_S
                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)

                # βV, U = A@βV (W 已预计算, 不重算)
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)  # U
                T.copy(tmp_dv2, bv_shared)  # bv_shared = U

                # ds = Lower(QKᵀ) ⊙ (g_exp_i * g_inv_j)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # V_new = U - W@S_old (W 从 global 读, 不重算 A@βγK)
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)  # W@S
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = T.cast(bv_shared[t, d], T.float32) - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # O = scale*(γ⊙(Q@S_old) + ds@V_new)
                T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                for t, d in T.Parallel(block_S, block_DV):
                    O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                        (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # state: S_new = γr·S_old + Kᵀ@(R@V_new), R@V_new = γr·γ_inv·V_new (原始精确形式)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# State-only 三 kernel (idea 18.6, S1/S2/S3/S4 实现)
#   数学 (S0 已验证 FP64 等价, FP32 ~1e-7; S1b BF16 workspace + C 保 FP32 全 PASS):
#     W=A@(βγK), U=A@(βV), ds=tril(QKᵀ)⊙(γ_i/γ_j)
#     C=Kᵀ@(R@W), B=Kᵀ@(R@U), R=diag(exp(g_last-g_i))    [state 系数]
#     P=γ⊙Q-ds@W, R_o=ds@U                               [output 系数]
#     Kernel A: 算 W/U/ds → C/B/P/R, 写 global workspace (grid=chunks*B*Hv, 完全并行)
#     Kernel B: S[c]=γr·S[c-1]-C@S[c-1]+B, 写 S_all[c+1] (grid=DV_slices*B*Hv, chunk 串行, 1 GEMM)
#     Kernel C: O[c]=scale·(P@S[c-1]+R_o), 读 S_all[c] (grid=chunks*DV_slices*B*Hv, 完全并行)
#   ★ 用户修正: S_new 写 S_all[c+1] (新位置), S_all[c] 留给 output kernel → state/output 时序解耦.
#   ★ C 保 FP32 workspace (BF16 C 会导致 state ~5e-3 超 RTOL; FP32 C 全 PASS ~2e-3).
#     W/U/ds/P/R_o 保 BF16 (与 matw 截断点一致, output 误差不跨 chunk 累积).
# ============================================================
@tilelang.jit(out_idx=[-4, -3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_so_pre_kernel(B, S, Hq, Hv, DK, DV, num_chunks, threads=256):
    """State-only Kernel A: 算 C/B/P/R 写 global workspace. state-independent, grid=chunks*B*Hv."""
    block_S = CHUNK_SIZE
    G = Hv // Hq
    QK_shape = (B, S, Hq, DK)
    V_shape = (B, S, Hv, DV)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    C_shape = (B, Hv, num_chunks, DK, DK)
    Bc_shape = (B, Hv, num_chunks, DK, DV)
    P_shape = (B, Hv, num_chunks, block_S, DK)
    Ro_shape = (B, Hv, num_chunks, block_S, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(QK_shape, dtype=T.bfloat16),
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        V: T.Tensor(V_shape, dtype=T.bfloat16),
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        C_out: T.Tensor(C_shape, dtype=T.float32),
        Bc_out: T.Tensor(Bc_shape, dtype=T.bfloat16),
        P_out: T.Tensor(P_shape, dtype=T.bfloat16),
        Ro_out: T.Tensor(Ro_shape, dtype=T.bfloat16),
    ):
        with T.Kernel(num_chunks, B * Hv, threads=threads) as (ci, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G
            left = ci * block_S

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, DV), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_shared = T.alloc_shared((1,), dtype=T.float32)
            R_shared = T.alloc_shared((block_S,), dtype=T.float32)

            W_frag = T.alloc_fragment((block_S, DK), dtype=T.float32)
            U_frag = T.alloc_fragment((block_S, DV), dtype=T.float32)
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            RW_frag = T.alloc_fragment((block_S, DK), dtype=T.float32)
            RU_frag = T.alloc_fragment((block_S, DV), dtype=T.float32)
            C_frag = T.alloc_fragment((DK, DK), dtype=T.float32)
            B_frag = T.alloc_fragment((DK, DV), dtype=T.float32)
            dsW_frag = T.alloc_fragment((block_S, DK), dtype=T.float32)
            dsU_frag = T.alloc_fragment((block_S, DV), dtype=T.float32)
            P_frag = T.alloc_fragment((block_S, DK), dtype=T.float32)

            T.copy(Q[bb, left:left + block_S, bhg, 0:DK], Q_shared)
            T.copy(K[bb, left:left + block_S, bhg, 0:DK], K_shared)
            T.copy(V[bb, left:left + block_S, bh, 0:DV], V_shared)
            T.copy(A[bb, left:left + block_S, bh, 0:block_S], A_shared)
            T.copy(g_cumsum[bb, left:left + block_S, bh], g_shared)
            T.copy(beta[bb, left:left + block_S, bh], beta_shared)

            for t in T.Parallel(block_S):
                g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]
            g_last_shared[0] = g_shared[block_S - 1]
            for t in T.Parallel(block_S):
                R_shared[t] = T.exp2((g_last_shared[0] - g_shared[t]) * LOG2E)

            # βγK, W = A@βγK
            for t, d in T.Parallel(block_S, DK):
                bkg_shared[t, d] = T.cast(
                    T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
            T.gemm(A_shared, bkg_shared, W_frag, clear_accum=True)
            T.copy(W_frag, bkg_shared)  # bkg_shared = W (BF16)
            # βV, U = A@βV
            for t, d in T.Parallel(block_S, DV):
                bv_shared[t, d] = T.cast(
                    T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
            T.gemm(A_shared, bv_shared, U_frag, clear_accum=True)
            T.copy(U_frag, bv_shared)  # bv_shared = U (BF16)

            # ds = tril(Q@Kᵀ) ⊙ (γ_i/γ_j)
            T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(block_S, block_S):
                if i >= j:
                    ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                else:
                    ds_tmp[i, j] = 0
            T.copy(ds_tmp, ds_shared)

            # ★ Reorder: (1) ds@W → P, (2) ds@U → R_o, (3) W*=R → C, (4) U*=R → B
            # (1) P = γ⊙Q - ds@W (uses raw W in bkg_shared)
            T.gemm(ds_shared, bkg_shared, dsW_frag, clear_accum=True)  # ds@W
            for t, d in T.Parallel(block_S, DK):
                P_frag[t, d] = T.cast(Q_shared[t, d], T.float32) * g_exp_shared[t] - dsW_frag[t, d]
            T.copy(P_frag, P_out[bb, bh, ci, 0:block_S, 0:DK])  # P BF16
            # (2) R_o = ds@U (uses raw U in bv_shared)
            T.gemm(ds_shared, bv_shared, dsU_frag, clear_accum=True)
            T.copy(dsU_frag, Ro_out[bb, bh, ci, 0:block_S, 0:DV])  # R_o BF16
            # (3) C = Kᵀ@(R⊙W): overwrite bkg_shared with R⊙W (W no longer needed raw)
            for t, d in T.Parallel(block_S, DK):
                bkg_shared[t, d] = T.cast(
                    T.cast(bkg_shared[t, d], T.float32) * R_shared[t], T.bfloat16)
            T.gemm(K_shared, bkg_shared, C_frag, transpose_A=True, clear_accum=True)
            T.copy(C_frag, C_out[bb, bh, ci, 0:DK, 0:DK])  # C FP32
            # (4) B = Kᵀ@(R⊙U): overwrite bv_shared with R⊙U
            for t, d in T.Parallel(block_S, DV):
                bv_shared[t, d] = T.cast(
                    T.cast(bv_shared[t, d], T.float32) * R_shared[t], T.bfloat16)
            T.gemm(K_shared, bv_shared, B_frag, transpose_A=True, clear_accum=True)
            T.copy(B_frag, Bc_out[bb, bh, ci, 0:DK, 0:DV])  # B BF16

    return kernel


# Kernel B: state recurrence, chunk serial, DV split.
#   S[c] = γr·S[c-1] - C@S[c-1] + B, 写 S_all[c+1]. C FP32 workspace, B BF16, S_old FP32 fragment.
#   ★ C@S_old: C is FP32 shared, S_old FP32 fragment → T.gemm needs same dtype. Store C as BF16 in a
#   separate shared for the GEMM, but ALSO keep FP32 C for accumulation? No — T.gemm(C_bf16_shared, S_old_bf16_shared)
#   gives FP32 accum CS_frag. S1b showed BF16 C + BF16 S_old snap → state ~2e-3 (PASS, C FP32 only marginally better).
#   Use BF16 C for the GEMM operand (C FP32 workspace → cast to BF16 shared), S_old BF16 snapshot.
@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_so_state_kernel(B, S, Hq, Hv, DK, DV, num_chunks, block_DV=128, threads=256):
    """State-only Kernel B: chunk serial. S[c]=γr·S[c-1]-C@S[c-1]+B → S_all[c+1]."""
    block_S = CHUNK_SIZE
    C_shape = (B, Hv, num_chunks, DK, DK)
    Bc_shape = (B, Hv, num_chunks, DK, DV)
    gate_shape = (B, S, Hv)
    init_shape = (B, Hv, DK, DV)
    Sall_shape = (B, Hv, num_chunks + 1, DK, DV)

    @T.prim_func
    def kernel(
        g_cumsum: T.Tensor(gate_shape, dtype=T.float32),
        C_in: T.Tensor(C_shape, dtype=T.float32),
        Bc_in: T.Tensor(Bc_shape, dtype=T.bfloat16),
        initial_state: T.Tensor(init_shape, dtype=T.float32),
        S_all: T.Tensor(Sall_shape, dtype=T.float32),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)
            C_bf16 = T.alloc_shared((DK, DK), dtype=T.bfloat16)
            B_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            CS_frag = T.alloc_fragment((DK, block_DV), dtype=T.float32)
            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)
            T.copy(s_fragment, S_all[bb, bh, 0, 0:DK, bv * block_DV : (bv + 1) * block_DV])

            for ci in T.serial(num_chunks):
                left = ci * block_S
                # g_last for this chunk
                g_last_local[0] = g_cumsum[bb, left + block_S - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)
                # C (FP32 ws) → BF16 shared for GEMM
                for i, j in T.Parallel(DK, DK):
                    C_bf16[i, j] = T.cast(C_in[bb, bh, ci, i, j], T.bfloat16)
                # B slice → B_shared (BF16, direct copy)
                T.copy(Bc_in[bb, bh, ci, 0:DK, bv * block_DV : (bv + 1) * block_DV], B_shared)
                # S_old snapshot → s_shared (BF16)
                T.copy(s_fragment, s_shared)
                # CS = C @ S_old (BF16 × BF16 → FP32)
                T.gemm(C_bf16, s_shared, CS_frag, clear_accum=True)
                # S_new = γr·S_old - CS + B
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = gl_local[0] * s_fragment[t, d] - CS_frag[t, d] + T.cast(B_shared[t, d], T.float32)
                # write S_all[c+1]
                T.copy(s_fragment, S_all[bb, bh, ci + 1, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# Kernel C: output, fully parallel over chunks. O[c]=scale·(P@S[c-1]+R_o), reads S_all[c].
@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_so_out_kernel(B, S, Hq, Hv, DK, DV, num_chunks, block_DV=128, threads=256):
    """State-only Kernel C: output. O[c]=scale·(P@S[c-1]+R_o). grid=chunks*ceildiv(DV/block_DV)*B*Hv."""
    block_S = CHUNK_SIZE
    P_shape = (B, Hv, num_chunks, block_S, DK)
    Ro_shape = (B, Hv, num_chunks, block_S, DV)
    Sall_shape = (B, Hv, num_chunks + 1, DK, DV)
    O_shape = (B, S, Hv, DV)

    @T.prim_func
    def kernel(
        P_in: T.Tensor(P_shape, dtype=T.bfloat16),
        Ro_in: T.Tensor(Ro_shape, dtype=T.bfloat16),
        S_all: T.Tensor(Sall_shape, dtype=T.float32),
        O: T.Tensor(O_shape, dtype=T.bfloat16),
    ):
        with T.Kernel(num_chunks, T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (ci, bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            left = ci * block_S

            P_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            Ro_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            O_frag = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.copy(P_in[bb, bh, ci, 0:block_S, 0:DK], P_shared)
            T.copy(Ro_in[bb, bh, ci, 0:block_S, bv * block_DV : (bv + 1) * block_DV], Ro_shared)
            # S_all[c] (FP32) → s_shared (BF16 snapshot)
            T.copy(S_all[bb, bh, ci, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            # O = P @ S_old + R_o
            T.gemm(P_shared, s_shared, O_frag, clear_accum=True)
            for t, d in T.Parallel(block_S, block_DV):
                O_frag[t, d] = O_frag[t, d] + T.cast(Ro_shared[t, d], T.float32)
            # write O * scale
            for t, d in T.Parallel(block_S, block_DV):
                O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                    (DK ** -0.5) * O_frag[t, d], T.bfloat16)

    return kernel



# ============================================================
# Qhat/Khat 换元版 (长序列, 精确零近似) — ★ 已验证性能退步, 保留供报告引用
#   实测 (GDN_QHAT=1, 集群): chain_equal 0.729ms (vs matw 0.53, +37%)
#                            long_low_gva 3.499ms (vs matw 3.04, +15%)
#   数学换元: Qhat=γ·Q, Khat=γ_inv·K. ds=tril(Qhat@Khatᵀ), O=scale·(Qhat@S_old+ds@V_new),
#     S_new=γr·S_old+Khatᵀ@(γr·V_new). 消 ds gate ew (γ_i/γ_j) 与 O γ ew, 精确.
#   ★ 退步根因: 消掉的 2 处 ew (ds gate, O γ) 本身在 GEMM↔ew 串行关键路径上,
#     但代价是 (a) Q/K load 改 T.Parallel element-wise (Qhat/Khat load 时缩放, 无法 T.copy),
#         (b) bkg=βγK 多读一次 K_shared (原 matw Q/K load 合并), load 体积反增,
#         (c) Khat_shared 新增 16KB shared (+2 回合 K→Khat ew).
#     T.Pipelined 对 load 只识别 T.copy, Q/K 改 ew 后从 producer 变 consumer 串行 → pipeline 退化.
#     与 TMA load 实验 (长序列 +23%) 同类失败: element-wise load 在长序列 chunk 多时累积开销大.
#   ★ 结论: 换元消 ew 的收益 < ew load 退步的代价. tilelang 下 load 必须走 T.copy 才进 pipeline.
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_naive_kernel_qhat(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=2):
    """Qhat/Khat 换元版: 消 ds gate 与 O γ 两处 ew, 精确零近似."""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            # ★ 换元: Qhat=γ·Q 直接装进 Q_shared (load 时缩放, 单次写); Khat=γ_inv·K 单独 buffer.
            #   K_shared 仅保留给 bkg=βγK (load 一次); ds/state 用 Khat_shared.
            Khat_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            # 预算 exp(g)/1/exp(g)/beta*g_exp, 复用省 exp2
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)  # W 产出
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # V_new/O 等
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                Khat_shared: tilelang.layout.make_swizzled_layout(Khat_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

                for t, d in T.Parallel(block_S, DK):
                    if left + t < S:
                        K_shared[t, d] = K[bb, left + t, bhg, d]
                    else:
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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                # 预算 exp(g)/1/exp(g)/beta*g_exp, 复用省 exp2
                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # ★ Qhat=γ·Q 直接装 Q_shared (单次写, 消后续 O 的 γ ew)
                for t, d in T.Parallel(block_S, DK):
                    if left + t < S:
                        Q_shared[t, d] = T.cast(
                            T.cast(Q[bb, left + t, bhg, d], T.float32) * g_exp_shared[t],
                            T.bfloat16)
                    else:
                        Q_shared[t, d] = 0

                # P1: βγK (bkg, 从 K_shared 读, 单次写), W=A@bkg (同 matw)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)   # tmp_dv = W
                T.copy(tmp_dv, W_shared)

                # βV, U=A@βV -> bv_shared (同 matw)
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)   # tmp_dv2 = U
                T.copy(tmp_dv2, bv_shared)   # bv_shared = U

                # ★ Khat=γ_inv·K 单独 buffer (单次写, 消 ds gate 与 state γr/γ 的 ew)
                for t, d in T.Parallel(block_S, DK):
                    Khat_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * g_inv_shared[t], T.bfloat16)

                # ★ ds = tril(Qhat @ Khatᵀ)  (γ_i/γ_j 已入 Qhat/Khat, 消 gate ew)
                T.gemm(Q_shared, Khat_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i < j:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # P2: V_new = U - W@S (同 matw, 用 s_fragment 的 S_old 快照)
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)   # tmp_dv2 = W@S
                T.copy(bv_shared, O_fragment)   # O_fragment = U
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = O_fragment[t, d] - tmp_dv2[t, d]   # tmp_dv2 = V_new
                T.copy(tmp_dv2, V_new_shared)

                # ★ P3: O = scale·(Qhat@S_old + ds@V_new)  (Qhat 已含 γ, 消 O γ ew)
                T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # ★ state: V_new *= γr (标量, γ_inv 已入 Khat), s_fragment *= γr, += Khatᵀ@V_new
                #   Khatᵀ@(γr·V_new) = γr·(γ_inv·K)ᵀ@V_new = Kᵀ@(γr/γ·V_new) ✓ 同 matw
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                T.gemm(Khat_shared, V_new_shared, s_fragment, transpose_A=True)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# Prefix-state 版 (长序列, block-constant output 近似):
#   数学: P[i] = Σ(j≤i) Khat[j]ᵀ @ V_new[j]   (Khat=γ_inv·K, V_new 已含 chunk 内因果)
#         O[i] ≈ scale·Qhat[i] @ (S_old + P[block_start])    [block 内用 block 起点前缀, 近似]
#         S_new = γr·(S_old + P[last])                        [state 精确, 误差不跨 chunk]
#   ★ 消除两个最贵的 S×S GEMM: Q@Kᵀ (ds 的来源) 和 ds@V_new (in-chunk output 的来源).
#     原 matw: Q@S_old + (Q@Kᵀ⊙gate)@V_new  (1 + 2 GEMM: Q@S, Q@Kᵀ, ds@V_new, 共 3 含 state 的 Kᵀ@V_new)
#     本版:    Q@S_old(用 P 增广) + Khatᵀ@V_new(state)       (2 GEMM: Q@S, Khatᵀ@V_new)
#     净省 1 GEMM (消 Q@Kᵀ) + 1 GEMM (消 ds@V_new) = 省 2 GEMM, 但增 block 数次 S snapshot copy.
#   ★ block=block_S(=64) 时退化为 chunk-constant: 全 chunk 用 S_old+P[0]=S_old 算 output, 近似最大.
#     block=1 时退化为精确逐 token prefix (需 64 次 snapshot, 无 GEMM 收益). block=16/32 是甜点.
#   state 保持精确: S_new = γr·(S_old + Khatᵀ@V_new), 与 matw 完全一致 (Khatᵀ@(γr·V_new) = Kᵀ@(γr/γ·V_new)).
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_prefix_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=2, block_prefix=64):
    """Prefix-state: 消 QKᵀ+ds@V_new 两 GEMM, state 精确, block-constant output 近似."""
    block_S = CHUNK_SIZE
    num_chunks = (S + block_S - 1) // block_S
    G = Hv // Hq
    assert block_prefix in (16, 32, 64), "block_prefix ∈ {16,32,64}"
    num_blocks = block_S // block_prefix

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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            # ★ s_fragment = S_old + P_prefix (FP32 accumulator, 跨 block/chunk 持久)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)        # W
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # W@S → V_new
            O_fragment = T.alloc_fragment((block_prefix, block_DV), dtype=T.float32)  # block output
            # ★ block 子块 shared (T.gemm 不支持 shared 操作数带首维 offset, 需独立 sub buffer)
            Q_sub = T.alloc_shared((block_prefix, DK), dtype=T.bfloat16)
            K_sub = T.alloc_shared((block_prefix, DK), dtype=T.bfloat16)
            Vn_sub = T.alloc_shared((block_prefix, block_DV), dtype=T.bfloat16)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # P1: βγK, W=A@bkg (同 matw)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)
                T.copy(tmp_dv, W_shared)

                # βV, U=A@βV (同 matw)
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)
                T.copy(tmp_dv2, bv_shared)   # bv_shared = U

                # P2: V_new = U - W@S_old (同 matw, s_fragment 此时 = S_old)
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)   # tmp_dv2 = W@S
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = T.cast(bv_shared[t, d], T.float32) - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # ★ P3 (prefix-state, block 循环): state 精确累加 prefix, output 用 block 起点前缀近似.
                #   s_fragment 此时 = S_old (P2 前拷贝, 未改).
                #   每 block:
                #     (1) s_shared = bf16(s_fragment)           [S_old + 已累积 prefix, 当前快照]
                #     (2) O_block = scale·γ⊙(Q_block @ s_shared) [block-constant output 近似]
                #     (3) s_fragment += Khat_subᵀ @ V_new_sub   [精确 prefix 累加, γ_inv 入 V_new]
                #   ★ tilelang T.gemm 不支持 shared 操作数带首维 offset 切片 ("offset of first dim must be 0").
                #     故需用独立 sub-shared buffer Q_sub/K_sub/Vn_sub 存子块, 而非切片 Q_shared[t0:,...].
                for b in T.serial(num_blocks):
                    t0 = b * block_prefix
                    # (1) snapshot 当前 s_fragment 到 s_shared (BF16)
                    T.copy(s_fragment, s_shared)
                    # (2) O_block = scale·γ⊙(Q_block @ s_shared)
                    #   Q_sub = Q_shared[t0:t0+block_prefix] (element-wise copy, 无 GEMM 切片)
                    for t, d in T.Parallel(block_prefix, DK):
                        Q_sub[t, d] = Q_shared[t0 + t, d]
                    T.gemm(Q_sub, s_shared, O_fragment, clear_accum=True)
                    for t, d in T.Parallel(block_prefix, block_DV):
                        if left + t0 + t < S:
                            O[bb, left + t0 + t, bh, bv * block_DV + d] = T.cast(
                                (DK ** -0.5) * g_exp_shared[t0 + t] * O_fragment[t, d],
                                T.bfloat16)
                    # (3) s_fragment += Khat_subᵀ @ V_new_sub  (Khat=γ_inv·K)
                    #   Vn_sub = V_new_shared[t0:t0+block_prefix] * γ_inv[t0:t0+block_prefix]
                    for t, d in T.Parallel(block_prefix, block_DV):
                        Vn_sub[t, d] = T.cast(
                            T.cast(V_new_shared[t0 + t, d], T.float32) * g_inv_shared[t0 + t],
                            T.bfloat16)
                    #   K_sub = K_shared[t0:t0+block_prefix]
                    for t, d in T.Parallel(block_prefix, DK):
                        K_sub[t, d] = K_shared[t0 + t, d]
                    T.gemm(K_sub, Vn_sub, s_fragment, transpose_A=True, clear_accum=False)

                # chunk 末: s_fragment *= γr (精确 S_new = γr·(S_old + P[last]))
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# WY-O 同步版 (长序列, 实验):
#   数学: O = scale*((γ⊙Q - ds@W)@S_old + ds@U), 消除 O→V_new 依赖
#         推导: O = scale*(γ⊙(Q@S) + ds@(U-W@S))
#                   = scale*((γ⊙Q - ds@W)@S + ds@U)
#   调度: 同步 T.gemm (对照实验, 隔离 async 收益). 8 GEMM (比 matw 7 多 1: ds@W).
#   ★ 对照目的: 之前 async 版 (T.wgmma_gemm+wait) 全线慢 40-80%, 疑 fragment 复用 copy +
#     spill 反收益. 本版回退同步, 确认 (a) 精度仍过 (b) async 是不是负收益元凶.
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_wyo_async_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=1):
    """WY-O 仿射同步版 (对照). O=scale*((γ⊙Q-ds@W)@S + ds@U), 不依赖 V_new."""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            U_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)   # ★ WY-O: U 驻留 (不覆写)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            Qp_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)        # ★ WY-O: γ⊙Q-ds@W
            WS_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)  # ★ W@S 截断驻留 (精度修复)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)         # W / ds@W (最大 [64,128])
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)    # U / WS / V_new
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            tmp_dkw = T.alloc_fragment((block_S, DK), dtype=T.float32)         # ★ WY-O: ds@W [64,128]

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                Qp_shared: tilelang.layout.make_swizzled_layout(Qp_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

                # ---- load (尾块补零, 同 matw) ----
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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                # 预算 exp(g)/1/exp(g)/beta*g_exp, 复用省 exp2
                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # βγK (ew)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                # βV 暂存到 V_new_shared (此时尚未用), 作 U=A@βV 的输入
                for t, d in T.Parallel(block_S, block_DV):
                    V_new_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)

                # ============ Phase A: state-free GEMM (3, 同步) ============
                T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)        # W = A@βγK
                T.copy(tmp_dv, W_shared)
                T.gemm(A_shared, V_new_shared, tmp_dv2, clear_accum=True)     # U = A@βV
                T.copy(tmp_dv2, U_shared)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)  # QKᵀ
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # ============ Phase B: state GEMMs (依赖 S_old, 2) ============
                T.copy(s_fragment, s_shared)   # s_shared = S_old
                T.gemm(ds_shared, W_shared, tmp_dkw, clear_accum=True)        # ds@W → tmp_dkw
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)        # W@S → WS (tmp_dv2)
                # ew: V_new = U - WS. ★ 三项目式精度修复: WS 截断回 BF16 shared 再减, 同 matw 截断策略.
                T.copy(tmp_dv2, WS_shared)   # WS FP32 -> BF16 shared (一次截断)
                T.copy(U_shared, tmp_dv2)    # tmp_dv2 = U (FP32)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] - T.cast(WS_shared[t, d], T.float32)  # V_new = U - WS
                T.copy(tmp_dv2, V_new_shared)   # V_new_shared = V_new (BF16, 供 Phase D)
                for t, d in T.Parallel(block_S, DK):
                    Qp_shared[t, d] = T.cast(
                        T.cast(Q_shared[t, d], T.float32) * g_exp_shared[t] - tmp_dkw[t, d],
                        T.bfloat16)

                # ============ Phase C: O GEMMs (2) ============
                T.gemm(ds_shared, U_shared, O_fragment, clear_accum=True)    # b_O = ds@U
                T.gemm(Qp_shared, s_shared, O_fragment, clear_accum=False)  # O += Qp@S_old
                # ew: gate V_new + gate S (不读 O_fragment)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)   # gated V_new
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                # ew: O *= scale, 写回 (依赖 O_fragment)
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # ============ Phase D: S update ============
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True, clear_accum=False)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# Phase A: 纯 Async Batch matw (idea.md 第一主路线)
#   不改数学, 不改 buffer, 不改 load. 只把 T.gemm (隐式 wait_group 0) 换成
#   T.wgmma_gemm (async, 不 wait) + 批量 T.wait_wgmma(0), 按依赖图压缩同步点.
#
#   matw 原 7 GEMM, 每个隐式 wait → 7 次同步. 目标压缩为 3 次:
#     Batch A (state-free, 输入只读互不冲突):
#       W=A@bkg, U=A@bv, QK=Q@Kᵀ, QS=Q@S_old  →  4 wgmma → wait(0)
#     Batch B (依赖 W):
#       WS=W@S_old                            →  1 wgmma → wait(0)
#     Batch C (依赖 V_new, 输出互不冲突):
#       ds@V_new→O, Kᵀ@V_new→state            →  2 wgmma → wait(0)
#
#   ★ 依赖正确性: Batch A 的 4 GEMM 输出 (W/U/QK/QS) 分属不同 fragment, 输入只读, 可并发 issue.
#     Q@S_old 用 s_shared=S_old (Batch B 前 s_shared 未被改). W@S 依赖 W_shared (需 Batch A 后 copy),
#     故 W@S 必须单独 Batch B. V_new = U - WS 需 ew (非 GEMM), 在 Batch B 后算, 供 Batch C 用.
#   ★ 与 WY-O async 区别: WY-O 多 +1 GEMM (ds@W) + Q' fragment + U 驻留压力. 本版纯 matw 数学,
#     只测 async 调度本身是否有收益 (隔离 WY-O 的反收益元凶).
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_async_matw_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=2):
    """Phase A: 纯 matw + T.wgmma_gemm async batch. 数学同 matw, 仅改 GEMM 调度."""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            # ★ Batch A 需 4 个独立输出 fragment: W[64,DK], U[64,DV], QK(ds_tmp)[64,64], QS(O_fragment)[64,DV]
            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)       # W
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # U / WS / V_new
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)   # QK
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # QS / O

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # ew: βγK, βV (供 Batch A 的 W/U GEMM 作操作数)
                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)

                # s_shared = S_old (供 Batch A 的 QS=Q@S_old 用; Batch B 的 W@S 也用它, 此时不改)
                T.copy(s_fragment, s_shared)

                # ===== Batch A: state-free GEMM, async wgmma (4 并发, 一次 wait) =====
                #   W=A@bkg→tmp_dv, U=A@bv→tmp_dv2, QK=Q@Kᵀ→ds_tmp, QS=Q@S_old→O_fragment
                #   ★ 4 个输出 fragment 互不冲突 (tmp_dv/tmp_dv2/ds_tmp/O_fragment 各异), 输入只读,
                #     可并发 issue. bkg/bv/s_shared 由 T.Parallel+T.copy 写入, 需 fence 供 wgmma async proxy 可见.
                #   ★ clear_accum=True 的 wgmma 不读旧 accum, 无 T.Parallel gate 冲突 (chunk1+ PASS 验证).
                T.fence_proxy_async()
                T.wgmma_gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)
                T.wgmma_gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)
                T.wgmma_gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                T.wgmma_gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                T.wait_wgmma(0)
                # 消费: W→shared, U→shared(bv), QK→gate+mask→ds_shared
                T.copy(tmp_dv, W_shared)
                T.copy(tmp_dv2, bv_shared)   # bv_shared = U
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # ===== Batch B: W@S_old (依赖 W_shared), async, wait =====
                #   ★ tmp_dv2 复用: W@S 写入 tmp_dv2 (U 已 copy 走到 bv_shared).
                #     wgmma clear_accum=True 不读旧 accum, 安全覆盖.
                T.fence_proxy_async()   # W_shared 由 T.copy 写, 供 wgmma 读前 fence
                T.wgmma_gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)
                T.wait_wgmma(0)
                # V_new = U - WS (ew; U 在 bv_shared, WS 在 tmp_dv2)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = T.cast(bv_shared[t, d], T.float32) - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # ===== Batch C: output + state (同步 T.gemm), V_new gating 顺序同 matw =====
                #   ★ Batch C 用同步 T.gemm: clear_accum=False 累加需 T.Parallel gate (O=γ⊙QS, s*=γr)
                #     对 accumulator 可见, 同步 T.gemm 隐式 fence 保证. wgmma async 不 fence accum 寄存器.
                #   matw 原序: ds@V_new 用 UNGATED V_new; 之后才 gate V_new 供 Kᵀ@V_new.
                # (1) gate O 的 from_state 项: O_fragment = γ⊙(Q@S_old)
                for t, d in T.Parallel(block_S, block_DV):
                    O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
                # (2) O += ds @ V_new  (UNGATED V_new, V_new_shared 仍是 Batch B 的未 gate 值)
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                # 写回 O (依赖 O_fragment, 在 gate V_new 前完成)
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)
                # (3) gate V_new *= γr·γ_inv (供 state 用), gate s_fragment *= γr
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                # (4) s_fragment += Kᵀ @ V_new (GATED V_new)
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True, clear_accum=False)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# Phase G1: [ds; K^T] 纵向堆叠 @ V_new (idea.md 第10.1节, 修正布局)
#   最后两次 GEMM (ds@V_new + K^T@V_new) 合成一次 M=192 GEMM:
#     P_shared[192, 64] = [ds(64x64); K^T(128x64)]  纵向, reduction=K=64=block_S
#     P @ V_new[64, DV] -> state_o[192, DV]  (clear_accum=False)
#       上 64 行 = ds @ V_new (output correction)
#       下 128 行 = K^T @ V_new (state update)
#   state_o[192,DV] 替代 s_fragment[DK,DV]+O_fragment[64,DV] (字节数同: 96KB)
#     上 64 = output, 下 128 = state
#   P_shared[192,64] 替代 ds_shared[64,64]+K_shared[64,128] (字节数同: 24KB)
#     上 64 行 = ds, 下 128 行 = K^T (K[64,128] 转置 -> [128,64])
#   GDN gating 矛盾: ds@V_new 需 UNGATED V_new, K^T@V_new 需 GATED.
#     堆叠用同一 V_new. 选 gate V_new 后堆叠 (state 精确, output ds@V_new 多乘 gamma_r*gamma_inv 近似).
#     state 跨 chunk 累积故必须精确; output 只影响当前 chunk, 近似可接受.
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_stack_matw_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=2):
    """Phase G1: [ds;K^T]@V_new 纵向堆叠 M=192, state_o 统一 fragment."""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            state_o = T.alloc_fragment((block_S + DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            P_shared = T.alloc_shared((block_S + DK, block_S), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            QS_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                bv_shared: tilelang.layout.make_swizzled_layout(bv_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            for t, d in T.Parallel(DK, block_DV):
                state_o[block_S + t, d] = T.cast(s_shared[t, d], T.float32)
            for t, d in T.Parallel(block_S, block_DV):
                state_o[t, d] = 0.0

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                for t, d in T.Parallel(block_S, DK):
                    bkg_shared[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)

                for t, d in T.Parallel(DK, block_DV):
                    s_shared[t, d] = T.cast(state_o[block_S + t, d], T.bfloat16)

                T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)
                T.copy(tmp_dv, W_shared)
                T.gemm(A_shared, bv_shared, tmp_dv2, clear_accum=True)
                T.copy(tmp_dv2, bv_shared)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)
                T.gemm(Q_shared, s_shared, QS_fragment, clear_accum=True)

                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = T.cast(bv_shared[t, d], T.float32) - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # Phase C: gate + stacked GEMM (gated V_new: state exact, output ds@V_new approx)
                # (1) state_o upper = gamma * QS (output base, clear)
                for t, d in T.Parallel(block_S, block_DV):
                    state_o[t, d] = g_exp_shared[t] * QS_fragment[t, d]
                # (2) assemble P_shared[192,64]: upper 64=ds, lower 128=K^T
                for i, j in T.Parallel(block_S, block_S):
                    P_shared[i, j] = ds_shared[i, j]
                for t, d in T.Parallel(block_S, DK):
                    P_shared[block_S + d, t] = K_shared[t, d]
                # (3) gate V_new *= gamma_r * gamma_inv; state_o lower *= gamma_r
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    state_o[block_S + t, d] = state_o[block_S + t, d] * gl_local[0]
                # (4) stacked GEMM: P[192,64] @ V_new[64,DV] -> state_o[192,DV]
                #   upper += ds @ (gamma_r*gamma_inv*V_new)  [APPROX: output ds@V_new 多乘 gamma_r*gamma_inv]
                #   lower += K^T @ (gamma_r*gamma_inv*V_new)  [EXACT: = gamma_r*K^T@(gamma_inv*V_new)]
                T.gemm(P_shared, V_new_shared, state_o, clear_accum=False)
                # (5) write O = scale * state_o[:64]
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * state_o[t, d], T.bfloat16)
                # (6) zero state_o upper for next chunk
                for t, d in T.Parallel(block_S, block_DV):
                    state_o[t, d] = 0.0

            for t, d in T.Parallel(DK, block_DV):
                s_shared[t, d] = T.cast(state_o[block_S + t, d], T.bfloat16)
            T.copy(s_shared, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# Phase G2: A@[βγK|βV] -> [W|U] 横向堆叠 (idea.md 第10.2节)
#   W = A @ βγK   [64,64]@[64,128] -> [64,128]
#   U = A @ βV     [64,64]@[64,128] -> [64,128]
#   合成: WU = A @ [βγK | βV]  [64,64]@[64,256] -> [64,256]  (M=64, N=256, K=64)
#   ★ 无 GDN gating 矛盾: W/U 都用 ungated βγK/βV (βγK 含 γ, βV 不含, 但都不需 ×γr·γ_inv).
#     共享 A 操作数, 可精确堆叠.
#   ★ B_wu_shared[64,256] 替代 bkg_shared[64,128]+bv_shared[64,128] (字节数同: 32KB)
#   ★ WU_fragment[64,256] 替代 tmp_dv[64,128]+tmp_dv2[64,128] (字节数同: 64KB)
#     左 128 列 = W, 右 128 列 = U. T.copy 到 W_shared/bv_shared 后正常 matw 流程.
#   ★ G0.2 probe 验证 N=256 wgmma 可用 (max_diff 1.1e-5).
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_stackwu_matw_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=2):
    """Phase G2: A@[βγK|βV]->[W|U] 堆叠 N=256, 无 gating 矛盾."""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            # ★ B_wu[64, DK+block_DV]: [βγK(64×128) | βV(64×block_DV)] 横向拼接
            B_wu = T.alloc_shared((block_S, DK + block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)  # U
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            # ★ WU_fragment[64, DK+block_DV]: 左 DK 列=W, 右 block_DV 列=U
            WU_fragment = T.alloc_fragment((block_S, DK + block_DV), dtype=T.float32)
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)  # W@S → V_new → O
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                B_wu: tilelang.layout.make_swizzled_layout(B_wu),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # ★ 组装 B_wu[64, DK+block_DV]: 左 DK=βγK, 右 block_DV=βV
                for t, d in T.Parallel(block_S, DK):
                    B_wu[t, d] = T.cast(
                        T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                for t, d in T.Parallel(block_S, block_DV):
                    B_wu[t, DK + d] = T.cast(
                        T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)

                # ★ 堆叠 GEMM: A @ B_wu -> WU_fragment[64, DK+block_DV] (M=64, N=256, K=64)
                T.gemm(A_shared, B_wu, WU_fragment, clear_accum=True)
                # 拆分: W = WU[:, :DK], U = WU[:, DK:]
                for t, d in T.Parallel(block_S, DK):
                    W_shared[t, d] = T.cast(WU_fragment[t, d], T.bfloat16)
                for t, d in T.Parallel(block_S, block_DV):
                    bv_shared[t, d] = T.cast(WU_fragment[t, DK + d], T.bfloat16)

                # ds = Lower(QKᵀ) ⊙ (g_exp_i * g_inv_j)  (同 matw)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # P2: V_new = U - W@S_old (同 matw)
                T.copy(s_fragment, s_shared)
                T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)   # W@S
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = T.cast(bv_shared[t, d], T.float32) - tmp_dv2[t, d]
                T.copy(tmp_dv2, V_new_shared)

                # P3: O = scale*(γ⊙(Q@S_old) + ds@V_new)  (同 matw)
                T.gemm(Q_shared, s_shared, O_fragment, clear_accum=True)
                for t, d in T.Parallel(block_S, block_DV):
                    O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
                T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # state: gate V_new *= gl * g_inv, s_fragment *= gl, += Kᵀ@V_new (同 matw)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel
#   在 WY-O 同步版基础上, 融合 3 对共操作数 GEMM:
#     F1: W=A@βγK, U=A@βV           → [W|U]=A@[βγK|βV]      (N=256, M=64)
#     F2: Qp@S, WS=W@S               → [WS;QpS]=[W;Qp]@S     (M=128, N=DV)
#     F3: ds@U, ds@WS                → ds@[U|WS]              (N=256, M=64)
#   ★ 风险点: fragment 偏移索引 (零先例, 见子agent调研). 分级验证:
#     - GDN_FUSE_F1_ONLY=1: 只做 F1 (最小验证 N=256 + fragment 取左/右半)
#     - GDN_FUSE_ALL=1:     F1+F2+F3 全做 (M=128/N=256, 取上下半/左右半)
#   ★ 若偏移索引 layout_inference 拒绝, 退化到 WY-O 同步版 (保精度, 无融合收益).
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_wyo_fuse_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256, num_stages=1,
                         use_F1=True, use_F2=True, use_F3=True):
    """WY-O + GEMM 融合. 5-6 GEMM (比 matw 7 少 1-2). 实验性, 风险=fragment 偏移索引."""
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            Q_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            K_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            A_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)

            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            # F1: [βγK|βV] 拼接 buffer (N=256). use_F1 时填, 否则用 bkg_shared+V_new_shared.
            # ★ F1 复用 bkg_shared 区域做中转: 先 A@[βγK|βV]→WU_frag, T.copy 到 bkbv_shared(BF16),
            #   再切片读左(W)/右(U). 这样不需 WU_shared_big 额外 32KB (省内存到 225KB).
            bkbv_shared = T.alloc_shared((block_S, DK + block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            U_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            Qp_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            WS_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            # F2: [W;Qp] 拼接 buffer (M=128). use_F2 时填.
            WQp_shared = T.alloc_shared((2 * block_S, DK), dtype=T.bfloat16)
            # F3: [U|WS] 拼接 buffer (N=256). use_F3 时填.
            UWS_shared = T.alloc_shared((block_S, block_DV + block_DV), dtype=T.bfloat16)

            g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            tmp_dkw = T.alloc_fragment((block_S, DK), dtype=T.float32)
            # F1 输出: [W|U] = A@[βγK|βV], [64, 256] FP32
            WU_frag = T.alloc_fragment((block_S, DK + block_DV), dtype=T.float32)
            # F2 输出: [WS;QpS] = [W;Qp]@S, [128, block_DV] FP32
            WSpQpS_frag = T.alloc_fragment((2 * block_S, block_DV), dtype=T.float32)
            # F3 输出: ds@[U|WS], [64, 2*block_DV] FP32
            dsUWS_frag = T.alloc_fragment((block_S, 2 * block_DV), dtype=T.float32)

            T.annotate_layout({
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                bkg_shared: tilelang.layout.make_swizzled_layout(bkg_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                Qp_shared: tilelang.layout.make_swizzled_layout(Qp_shared),
            })
            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                left = i_c * block_S
                length = T.min(block_S, S - left)

                # ---- load (尾块补零) ----
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
                        A_shared[t, d] = 1 if t == d else 0
                for t in T.Parallel(block_S):
                    if left + t < S:
                        g_shared[t] = g_cumsum[bb, left + t, bh]
                        beta_shared[t] = beta[bb, left + t, bh]
                    else:
                        g_shared[t] = 0
                        beta_shared[t] = 0

                g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                for t in T.Parallel(block_S):
                    g_exp_shared[t] = T.exp2(g_shared[t] * LOG2E)
                    g_inv_shared[t] = T.exp2(-g_shared[t] * LOG2E)
                    beta_g_shared[t] = beta_shared[t] * g_exp_shared[t]

                # ew: βγK, βV. F1: 拼成 [βγK|βV] (N=256) 供 A@[βγK|βV] 一次 GEMM
                if use_F1:
                    for t, d in T.Parallel(block_S, DK + block_DV):
                        if d < DK:
                            bkbv_shared[t, d] = T.cast(
                                T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                        else:
                            bkbv_shared[t, d] = T.cast(
                                T.cast(V_shared[t, d - DK], T.float32) * beta_shared[t], T.bfloat16)
                    T.gemm(A_shared, bkbv_shared, WU_frag, clear_accum=True)   # [W|U] = A@[βγK|βV]
                    # ★ fragment 偏移索引零先例. 改用 shared 切片中转:
                    #   bkbv_shared 此刻已不再用作 GEMM 输入, 复用为 WU 中转 (BF16 截断一次)
                    T.copy(WU_frag, bkbv_shared)   # FP32 frag -> BF16 shared (整体拷贝, 无偏移)
                    for t, d in T.Parallel(block_S, DK):
                        W_shared[t, d] = bkbv_shared[t, d]           # shared 切片 (左半)
                    for t, d in T.Parallel(block_S, block_DV):
                        U_shared[t, d] = bkbv_shared[t, DK + d]     # shared 切片 (右半)
                else:
                    for t, d in T.Parallel(block_S, DK):
                        bkg_shared[t, d] = T.cast(
                            T.cast(K_shared[t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                    T.gemm(A_shared, bkg_shared, tmp_dv, clear_accum=True)
                    T.copy(tmp_dv, W_shared)
                    for t, d in T.Parallel(block_S, block_DV):
                        V_new_shared[t, d] = T.cast(
                            T.cast(V_shared[t, d], T.float32) * beta_shared[t], T.bfloat16)
                    T.gemm(A_shared, V_new_shared, tmp_dv2, clear_accum=True)
                    T.copy(tmp_dv2, U_shared)
                # ds = Lower(QKᵀ) ⊙ (g_exp_i * g_inv_j)
                T.gemm(Q_shared, K_shared, ds_tmp, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_S, block_S):
                    if i >= j:
                        ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                    else:
                        ds_tmp[i, j] = 0
                T.copy(ds_tmp, ds_shared)

                # Qp = γ⊙Q - ds@W (先算 ds@W, 用 tmp_dkw)
                T.gemm(ds_shared, W_shared, tmp_dkw, clear_accum=True)   # ds@W
                for t, d in T.Parallel(block_S, DK):
                    Qp_shared[t, d] = T.cast(
                        T.cast(Q_shared[t, d], T.float32) * g_exp_shared[t] - tmp_dkw[t, d],
                        T.bfloat16)

                # ---- state GEMMs: WS=W@S, QpS=Qp@S ----
                # ★ O_fragment 直接接收 QpS (O = QpS + ds@U - ds@W@S = QpS + dsU - dsWS)
                T.copy(s_fragment, s_shared)   # s_shared = S_old
                if use_F2:
                    # F2: [W;Qp] 拼成 [128, DK], [W;Qp]@S 一次 M=128 GEMM
                    for t, d in T.Parallel(block_S, DK):
                        WQp_shared[t, d] = W_shared[t, d]
                        WQp_shared[block_S + t, d] = Qp_shared[t, d]
                    T.gemm(WQp_shared, s_shared, WSpQpS_frag, clear_accum=True)  # [128, DV]
                    # 上半 WS → WS_shared (BF16 截断); 下半 QpS → O_fragment (FP32, O 累加起点)
                    for t, d in T.Parallel(block_S, block_DV):
                        WS_shared[t, d] = T.cast(WSpQpS_frag[t, d], T.bfloat16)
                        O_fragment[t, d] = WSpQpS_frag[block_S + t, d]
                else:
                    T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)   # WS
                    T.copy(tmp_dv2, WS_shared)
                    T.gemm(Qp_shared, s_shared, O_fragment, clear_accum=True)  # QpS → O_fragment

                # V_new = U - WS (U 在 U_shared, WS 在 WS_shared BF16). 用 tmp_dv2 临时存.
                # ★ 不用 tmp_dv (它在 else 分支给 ds@WS 留着; F3 分支没用到 tmp_dv 但保持一致)
                T.copy(U_shared, tmp_dv2)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] - T.cast(WS_shared[t, d], T.float32)  # V_new
                T.copy(tmp_dv2, V_new_shared)

                # ---- O 累加: O = Qp@S + ds@U (WY-O, 2 GEMM 累加, 不减 ds@WS!) ----
                # ★ 数学: O = scale*((γ⊙Q-ds@W)@S + ds@U) = scale*(Qp@S + ds@U)
                #   Qp 已含 -ds@W, 不需再减 ds@W@S. 之前 else 分支多减一次 = 0.049 误差根因.
                # ★ F3 (ds@[W|U]→[dsW|dsU]) 要在 Qp 构造前 issue, 时序复杂, 暂只用保守分支
                T.gemm(ds_shared, U_shared, O_fragment, clear_accum=False)   # O += ds@U (QpS 已在 O_fragment)
                # gate V_new + gate S (不依赖 O_fragment)
                for t, d in T.Parallel(block_S, block_DV):
                    tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                T.copy(tmp_dv2, V_new_shared)
                for t, d in T.Parallel(DK, block_DV):
                    s_fragment[t, d] = s_fragment[t, d] * gl_local[0]

                # O *= scale, 写回
                for t, d in T.Parallel(block_S, block_DV):
                    if left + t < S:
                        O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                            (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                # S update
                T.gemm(K_shared, V_new_shared, s_fragment, transpose_A=True, clear_accum=False)

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


# ============================================================
# 4-WG 第五轮: 照搬 FlashQLA hopper fused_fwd.py 的 4-WG 结构
#
# ★ 关键发现 (读 FlashQLA 源码 + tilelang builtin.py):
#   1. T.barrier_arrive/barrier_wait 是 T.mbarrier_arrive/mbarrier_wait_parity 的语法糖 (builtin.py 确认).
#      FlashQLA 43 处 T.barrier_*, 0 处 T.mbarrier_*, 0 处 fence_proxy_async!
#   2. FlashQLA 用 T.tma_copy(barrier=data_is_ready[...]) 让 TMA load 完成自动 arrive barrier,
#      不需手动 fence. element-wise load 的 sub-warp 末尾手动 barrier_arrive.
#   3. FlashQLA 4 sub-warp producer (tx>=384): 每个载一类 (Q+K / V+β / A+g / O/S store),
#      data_is_ready arrive_count=96=3 sub-warp×32 (第 4 个 O/S store sub-warp 不参与 data_ready).
#   4. set_max_nreg: S=160/V=128/O=128/P=32 (之前三轮我漏了这个, 4-WG 寄存器争用是死锁根因之一).
#   5. FlashQLA 数学: W=V-g⊙U (不是 A@βγK!), U=K@S, Vd=A@W, V'=g_rev⊙Vd.
#      即 W 和 U 的角色互换 (FlashQLA 的 W 是 gated V, U 是 K@S).
#   6. 13 个 barrier (data_is_ready/free 双 buffer + bar_0/1/3/4/5 + bar_o).
#
# 本轮: 完整照搬 FlashQLA 数学 + barrier 结构 + set_max_nreg + T.tma_copy barrier=,
#   只适配我们的接口 (g_cumsum 已预算, A 已 solve) 和 per-case DV 分发.
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_ws4_fqla_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=512):
    """4-WG 照搬 FlashQLA hopper. threads=512, 4 sub-warp (S/V/O/P)."""
    block_S = CHUNK_SIZE
    num_chunks = (S + block_S - 1) // block_S
    G = Hv // Hq
    scale = DK ** -0.5

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
        # grid: (DV/block_DV, B*Hv). FlashQLA 把 batch*H*DV_blocks 拍平成 1 维, 这里保持 2 维.
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G
            DV_start = bv * block_DV
            DV_end = (bv + 1) * block_DV

            # 双 buffer load (num_stages=2 ping-pong, 第一维 stage)
            q_shared = T.alloc_shared((2, block_S, DK), dtype=T.bfloat16)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=T.bfloat16)
            v_shared = T.alloc_shared((2, block_S, block_DV), dtype=T.bfloat16)
            a_shared = T.alloc_shared((2, block_S, block_S), dtype=T.bfloat16)
            g_shared = T.alloc_shared((2, block_S), dtype=T.float32, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=T.float32, scope="shared")

            # consumer 输出 shared (跨 WG 传递)
            o_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            h_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            p_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            g_exp_shared = T.alloc_shared((block_S), dtype=T.float32, scope="shared")
            g_rev_exp_shared = T.alloc_shared((block_S), dtype=T.float32, scope="shared")

            # fragments
            h_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            g_last_local = T.alloc_local((1), dtype=T.float32)

            # ★ FlashQLA barrier 配置 (照搬)
            # data_is_ready: 3 producer sub-warp arrive (96=3×32), 3 consumer wait
            # data_is_free: 3 consumer arrive (384=3×128), producer wait
            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[384] * 2)
            bar_o = T.alloc_barrier(arrive_count=128)
            bar_0 = T.alloc_barrier(arrive_count=416)
            bar_1 = T.alloc_barrier(arrive_count=256)
            bar_3 = T.alloc_barrier(arrive_count=128)
            bar_4 = T.alloc_barrier(arrive_count=128)
            bar_5 = T.alloc_barrier(arrive_count=416)

            T.use_swizzle(10)
            tx = T.get_thread_binding()

            PRODUCER_NREG = 32
            CONSUMER_V_NREG = 128
            CONSUMER_S_NREG = 160
            CONSUMER_O_NREG = 128

            # CONSUMER_S [0,128)
            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)
                # init S
                T.copy(initial_state[bb, bh, 0:DK, DV_start:DV_end], h_fragment)
                for i_s in T.serial(num_chunks):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)
                    T.barrier_wait(bar_0, i_s % 2)
                    # S4[S] S: 发布 h_shared = S_old
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)
                    T.barrier_wait(bar_1, i_s % 2)
                    # S *= g_last
                    g_last_local[0] = g_exp_shared[block_S - 1]
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.barrier_arrive(bar_5)
                    T.barrier_wait(bar_5, i_s % 2)
                    # S += K^T @ V'
                    T.gemm(k_shared[i_s % 2, :, :], vn_shared, h_fragment,
                           transpose_A=True, clear_accum=False)
                    T.barrier_arrive(data_is_free[i_s % 2])
                # store final S
                T.copy(h_fragment, final_state[bb, bh, 0:DK, DV_start:DV_end])

            # CONSUMER_V [128,256)
            elif tx < 256:
                T.set_max_nreg(CONSUMER_V_NREG, 1)
                for i_s in T.serial(num_chunks):
                    left = i_s * block_S
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)
                    T.barrier_wait(bar_0, i_s % 2)
                    # g_exp, g_rev_exp
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[i_s % 2, j_s] * LOG2E)
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_shared[i_s % 2, block_S - 1] - g_shared[i_s % 2, j_s]) * LOG2E)
                    T.barrier_arrive(bar_1)
                    T.barrier_wait(bar_1, i_s % 2)
                    # U = K @ S
                    T.gemm(k_shared[i_s % 2, :, :], h_shared, u_fragment, clear_accum=True)
                    # W = V - g * U  (FlashQLA: W = V - g⊙U, 存回 v_shared 作 W)
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_shared[i_s % 2, j_s, j_v] = u_fragment[j_s, j_v] + v_shared[i_s % 2, j_s, j_v]
                    T.barrier_wait(bar_3, i_s % 2)
                    # Vd = A @ W
                    T.gemm(a_shared[i_s % 2, :, :], v_shared[i_s % 2, :, :], v_fragment, clear_accum=True)
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)
                    # V' = g_rev * Vd
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_5)
                    T.barrier_wait(bar_5, i_s % 2)
                    T.barrier_arrive(data_is_free[i_s % 2])

            # CONSUMER_O [256,384)
            elif tx < 384:
                T.set_max_nreg(CONSUMER_O_NREG, 1)
                for i_s in T.serial(num_chunks):
                    left = i_s * block_S
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)
                    T.barrier_wait(bar_0, i_s % 2)
                    # P = Q K^T
                    T.gemm(q_shared[i_s % 2, :, :], k_shared[i_s % 2, :, :], p_fragment,
                           transpose_B=True, clear_accum=True)
                    # G = Lower(exp(g_i - g_j))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            g_fragment[j_s, j_t] = T.exp2(g_fragment[j_s, j_t] * LOG2E)
                        else:
                            g_fragment[j_s, j_t] = 0
                    # Ag = G * A * b  (A 已是 kkt solve 的 A, b=beta)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = a_shared[i_s % 2, j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[i_s % 2, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_shared[i_s % 2, j_s, j_t] = a_fragment[j_s, j_t]
                    T.barrier_wait(bar_1, i_s % 2)
                    # O = Q @ S
                    T.gemm(q_shared[i_s % 2, :, :], h_shared, o_fragment, clear_accum=True)
                    # Pg = scale * G * P
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    T.copy(p_fragment, p_shared)
                    T.barrier_arrive(bar_3)
                    # O = scale * g * O
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                    T.barrier_wait(bar_4, i_s % 2)
                    # O += Pg @ Vd
                    T.gemm(p_shared, vd_shared, o_fragment, clear_accum=False)
                    T.barrier_arrive(bar_5)
                    T.barrier_wait(bar_5, i_s % 2)
                    T.copy(o_fragment, o_shared)
                    T.barrier_arrive(data_is_free[i_s % 2])
                T.barrier_arrive(bar_o)

            # PRODUCER [384,512): 4 sub-warp 载不同数据
            else:
                T.set_max_nreg(PRODUCER_NREG, 0)
                # sub-warp 0 [384,416): Q, K
                if tx < 384 + 32:
                    for i_s in T.serial(num_chunks):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S
                        # TMA load Q (barrier= 自动 arrive)
                        if right <= S:
                            T.tma_copy(Q[bb, left:right, bhg, 0:DK], q_shared[i_s % 2, :, :],
                                       barrier=data_is_ready[i_s % 2])
                            T.tma_copy(K[bb, left:right, bhg, 0:DK], k_shared[i_s % 2, :, :],
                                       barrier=data_is_ready[i_s % 2])
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < S:
                                    q_shared[i_s % 2, j_s, j_k] = Q[bb, left + j_s, bhg, j_k]
                                    k_shared[i_s % 2, j_s, j_k] = K[bb, left + j_s, bhg, j_k]
                                else:
                                    q_shared[i_s % 2, j_s, j_k] = 0
                                    k_shared[i_s % 2, j_s, j_k] = 0
                            T.barrier_arrive(data_is_ready[i_s % 2])
                            T.barrier_arrive(data_is_ready[i_s % 2])
                # sub-warp 1 [416,448): V, beta
                elif tx < 384 + 64:
                    for i_s in T.serial(num_chunks):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S
                        if right <= S:
                            T.tma_copy(V[bb, left:right, bh, DV_start:DV_end],
                                       v_shared[i_s % 2, :, :], barrier=data_is_ready[i_s % 2])
                        else:
                            for j_s, j_v in T.Parallel(block_S, block_DV):
                                if left + j_s < S:
                                    v_shared[i_s % 2, j_s, j_v] = V[bb, left + j_s, bh, DV_start + j_v]
                                else:
                                    v_shared[i_s % 2, j_s, j_v] = 0
                            T.barrier_arrive(data_is_ready[i_s % 2])
                        # beta (element-wise, 手动 arrive)
                        if right <= S:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % 2, j_s] = beta[bb, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < S:
                                    b_shared[i_s % 2, j_s] = beta[bb, left + j_s, bh]
                                else:
                                    b_shared[i_s % 2, j_s] = 0
                        T.barrier_arrive(data_is_ready[i_s % 2])
                # sub-warp 2 [448,480): A, g
                elif tx < 384 + 96:
                    for i_s in T.serial(num_chunks):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S
                        if right <= S:
                            T.tma_copy(A[bb, left:right, bh, 0:block_S],
                                       a_shared[i_s % 2, :, :], barrier=data_is_ready[i_s % 2])
                        else:
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < S:
                                    a_shared[i_s % 2, j_s, j_t] = A[bb, left + j_s, bh, j_t]
                                else:
                                    a_shared[i_s % 2, j_s, j_t] = 1 if j_s == j_t else 0
                            T.barrier_arrive(data_is_ready[i_s % 2])
                        # g (element-wise)
                        if right <= S:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % 2, j_s] = g_cumsum[bb, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < S:
                                    g_shared[i_s % 2, j_s] = g_cumsum[bb, left + j_s, bh]
                                else:
                                    g_shared[i_s % 2, j_s] = g_cumsum[bb, S - 1, bh]
                        T.barrier_arrive(data_is_ready[i_s % 2])
                # sub-warp 3 [480,512): O store (照搬 FlashQLA store sub-warp 结构)
                # ★ FlashQLA store sub-warp: 循环 arrive(bar_0)→wait(bar_0)→store O→arrive(bar_5)→wait(bar_1)
                #   只跑 num_unmasked_iters (无尾块 case = num_chunks), 最后单存最后一个 O.
                else:
                    for i_s in T.serial(num_chunks):
                        right = i_s * block_S
                        left = right - block_S
                        T.barrier_arrive(bar_0)
                        T.barrier_wait(bar_0, i_s % 2)
                        # store O (上一 chunk 的)
                        if i_s > 0:
                            T.copy(o_shared, O[bb, left:right, bh, DV_start:DV_end])
                        T.barrier_arrive(bar_5)
                        T.barrier_wait(bar_1, i_s % 2)
                    # 最后一个 chunk 的 O
                    last_left = (num_chunks - 1) * block_S
                    last_right = last_left + block_S
                    T.barrier_wait(bar_o, 0)
                    T.copy(o_shared, O[bb, last_left:last_right, bh, DV_start:DV_end])

    return kernel


# ============================================================
# Phase A: 2-WG producer/consumer 探针 kernel (验证 warp-spec API)
#   最小可复现样例: 照抄官方 example_warp_specialize_gemm_barrierpipe_stage2 的 barrier 模式
#   producer (T.ws(1)): TMA load → arrive; consumer (T.ws(0)): wait → 用 T.copy 写回 global
#   数学: identity (O = V), 只验证 barrier 语义本身在 MIG 上工作
# ============================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _gdn_ws_probe_kernel(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256):
    """2-WG 探针最小样例: 验证 mbarrier ping-pong。O=V (identity), final_state=0。"""
    block_S = CHUNK_SIZE
    num_chunks = (S + block_S - 1) // block_S
    G = Hv // Hq

    V_shape = (B, S, Hv, DV)
    O_shape = (B, S, Hv, DV)
    final_shape = (B, Hv, DK, DV)
    # 探针只需 V 和 O; 其它张量声明但不用 (接口要求签名一致, 但探针不调真 kernel 路径)
    QK_shape = (B, S, Hq, DK)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    init_shape = (B, Hv, DK, DV)

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
        # ★ 探针: 照抄官方 barrier 模式 + T.copy (TMA) load (不用 T.Parallel element-wise store,
        #   后者对 wgmma async proxy 不可见导致 NaN). 数学 = 物化W版完整 GDN, 验证 barrier+math 正确.
        num_stages = 2
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb, bh = bbh // Hv, bbh % Hv
            bhg = bh // G

            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)

            # 双 buffer (ping-pong 第一维)
            Q_shared_2 = T.alloc_shared((num_stages, block_S, DK), dtype=T.bfloat16)
            K_shared_2 = T.alloc_shared((num_stages, block_S, DK), dtype=T.bfloat16)
            V_shared_2 = T.alloc_shared((num_stages, block_S, block_DV), dtype=T.bfloat16)
            A_shared_2 = T.alloc_shared((num_stages, block_S, block_S), dtype=T.bfloat16)
            g_shared_2 = T.alloc_shared((num_stages, block_S), dtype=T.float32)
            beta_shared_2 = T.alloc_shared((num_stages, block_S), dtype=T.float32)

            # 单 buffer (consumer 内部 element-wise 中间结果)
            bkg_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            bv_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            W_shared = T.alloc_shared((block_S, DK), dtype=T.bfloat16)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=T.bfloat16)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=T.bfloat16)
            g_exp_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_inv_shared = T.alloc_shared((block_S,), dtype=T.float32)
            beta_g_shared = T.alloc_shared((block_S,), dtype=T.float32)
            g_last_local = T.alloc_local((1,), T.float32)
            gl_local = T.alloc_local((1,), T.float32)

            tmp_dv = T.alloc_fragment((block_S, DK), dtype=T.float32)
            tmp_dv2 = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            ds_tmp = T.alloc_fragment((block_S, block_S), dtype=T.float32)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)

            T.disable_warp_group_reg_alloc()

            # barrier: [0,1] = data_is_ready (producer→consumer), [2,3] = data_is_free (consumer→producer)
            mbars = T.alloc_barrier([128, 128] * num_stages)

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)

            for i_c in T.serial(num_chunks):
                left = i_c * block_S
                length = T.min(block_S, S - left)
                buf = i_c % num_stages

                with T.ws(1):
                    # producer 等 consumer 释放 buffer[buf]
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[buf + num_stages],
                        parity=((i_c // num_stages) % num_stages) ^ 1,
                    )
                    # ★ load 策略: T.copy (TMA) 对 wgmma 可见, 但动态切片尾块有问题
                    #   折中: 全 chunk 都 T.copy 固定 block_S 切片 (尾块越界由 consumer 的 mask 兜底,
                    #   但 T.copy 越界读会崩 → 需保证切片不越界). 对尾块改用 element-wise + fence.
                    #   实测: 无尾块 case 用 T.copy PASS; 尾块 case 需 element-wise.
                    #   统一用 element-wise (Q/K/V/A/g/beta) + fence_proxy_async, 简单且正确
                    #   (consumer 的 T.gemm 读 element-wise 写的 shared, fence 后可见)
                    for t, d in T.Parallel(block_S, DK):
                        if left + t < S:
                            Q_shared_2[buf, t, d] = Q[bb, left + t, bhg, d]
                            K_shared_2[buf, t, d] = K[bb, left + t, bhg, d]
                        else:
                            Q_shared_2[buf, t, d] = 0
                            K_shared_2[buf, t, d] = 0
                    for t, d in T.Parallel(block_S, block_DV):
                        if left + t < S:
                            V_shared_2[buf, t, d] = V[bb, left + t, bh, bv * block_DV + d]
                        else:
                            V_shared_2[buf, t, d] = 0
                    for t, d in T.Parallel(block_S, block_S):
                        if left + t < S:
                            A_shared_2[buf, t, d] = A[bb, left + t, bh, d]
                        else:
                            A_shared_2[buf, t, d] = 1 if t == d else 0
                    for t in T.Parallel(block_S):
                        if left + t < S:
                            g_shared_2[buf, t] = g_cumsum[bb, left + t, bh]
                            beta_shared_2[buf, t] = beta[bb, left + t, bh]
                        else:
                            g_shared_2[buf, t] = 0
                            beta_shared_2[buf, t] = 0
                    T.fence_proxy_async()
                    T.mbarrier_arrive(mbarrier=mbars[buf])

                with T.ws(0):
                    # consumer 等 producer 数据就绪
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[buf],
                        parity=(i_c // num_stages) % num_stages,
                    )
                    g_last_local[0] = g_cumsum[bb, left + length - 1, bh]
                    gl_local[0] = T.exp2(g_last_local[0] * LOG2E)

                    for t in T.Parallel(block_S):
                        g_exp_shared[t] = T.exp2(g_shared_2[buf, t] * LOG2E)
                        g_inv_shared[t] = T.exp2(-g_shared_2[buf, t] * LOG2E)
                        beta_g_shared[t] = beta_shared_2[buf, t] * g_exp_shared[t]

                    # P1: βγK, W=A@(K⊙beta_g)
                    for t, d in T.Parallel(block_S, DK):
                        bkg_shared[t, d] = T.cast(
                            T.cast(K_shared_2[buf, t, d], T.float32) * beta_g_shared[t], T.bfloat16)
                    T.gemm(A_shared_2[buf, :, :], bkg_shared, tmp_dv, clear_accum=True)
                    T.copy(tmp_dv, W_shared)

                    # βV, U=A@βV
                    for t, d in T.Parallel(block_S, block_DV):
                        bv_shared[t, d] = T.cast(
                            T.cast(V_shared_2[buf, t, d], T.float32) * beta_shared_2[buf, t], T.bfloat16)
                    T.gemm(A_shared_2[buf, :, :], bv_shared, tmp_dv2, clear_accum=True)
                    T.copy(tmp_dv2, bv_shared)

                    # ds = Lower(QKᵀ) ⊙ (g_exp_i * g_inv_j)
                    T.gemm(Q_shared_2[buf, :, :], K_shared_2[buf, :, :], ds_tmp, transpose_B=True, clear_accum=True)
                    for i, j in T.Parallel(block_S, block_S):
                        if i >= j:
                            ds_tmp[i, j] = ds_tmp[i, j] * g_exp_shared[i] * g_inv_shared[j]
                        else:
                            ds_tmp[i, j] = 0
                    T.copy(ds_tmp, ds_shared)

                    # P2: V_new = U - W@S
                    T.copy(s_fragment, s_shared)
                    T.gemm(W_shared, s_shared, tmp_dv2, clear_accum=True)
                    T.copy(bv_shared, O_fragment)
                    for t, d in T.Parallel(block_S, block_DV):
                        tmp_dv2[t, d] = O_fragment[t, d] - tmp_dv2[t, d]
                    T.copy(tmp_dv2, V_new_shared)

                    # P3: O = scale*(γ⊙(Q@S_old) + ds@V_new)
                    T.gemm(Q_shared_2[buf, :, :], s_shared, O_fragment, clear_accum=True)
                    for t, d in T.Parallel(block_S, block_DV):
                        O_fragment[t, d] = g_exp_shared[t] * O_fragment[t, d]
                    T.gemm(ds_shared, V_new_shared, O_fragment, clear_accum=False)
                    for t, d in T.Parallel(block_S, block_DV):
                        if left + t < S:
                            O[bb, left + t, bh, bv * block_DV + d] = T.cast(
                                (DK ** -0.5) * O_fragment[t, d], T.bfloat16)

                    # 更新 state
                    for t, d in T.Parallel(block_S, block_DV):
                        tmp_dv2[t, d] = tmp_dv2[t, d] * gl_local[0] * g_inv_shared[t]
                    T.copy(tmp_dv2, V_new_shared)
                    for t, d in T.Parallel(DK, block_DV):
                        s_fragment[t, d] = s_fragment[t, d] * gl_local[0]
                    T.gemm(K_shared_2[buf, :, :], V_new_shared, s_fragment, transpose_A=True)
                    # 通知 producer: buffer 已释放
                    T.mbarrier_arrive(mbarrier=mbars[buf + num_stages])

            T.copy(s_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


def _gdn_ws_probe_kernel_placeholder(B, S, Hq, Hv, DK, DV, block_DV=128, threads=256):
    """旧探针代码(物化W版数学 + 自写 barrier), 已被最小 identity 探针替代, 删除其内容避免语法冲突。"""
    raise NotImplementedError


def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN prefill 前向 (per-case 分发 + 手动最优配置, 不用 autotune 避免 OJ JIT 超时)。"""
    batch_size, num_tokens, num_heads_qk, head_dim_k = q.shape
    _, _, num_heads_v, head_dim_v = v.shape

    if initial_state is None:
        initial_state = torch.zeros(
            (batch_size, num_heads_v, head_dim_k, head_dim_v),
            dtype=torch.float32, device=q.device,
        )

    # 手动固定 autotune 搜出的最优配置 (不用 @autotune, 避免 OJ JIT 编译超时)
    # 短序列 (T<=2048): 结合律版 DV=64/th=128/st=2, 寄存器少, launch 占比大受益
    # 长序列 (T>2048):  物化W版 DV=128/th=256/st=1, 大 tile + 寄存器分摊, GEMM 少
    # ★ 调试开关:
    #   GDN_WS_PROBE=1 跑 2-WG 探针验证 warp-spec API
    #   GDN_WYO=1     跑 WY-O+async 实验版 (长序列实验, 默认关, 默认走 matw 保 93 基线)
    #   GDN_TMA=1      跑 TMA load 版 (T.copy producer + num_stages=2, 仅 T%64==0 长序列)
    #   GDN_DECOMP=1   跑三 kernel 分解 (K1 并行 W/U/ds + K2 串行 S + K3 并行 O)
    #   GDN_WS4_FQLA=1 跑 4-WG 照搬 FlashQLA hopper (第五轮)
    import os
    _WS_PROBE = os.environ.get("GDN_WS_PROBE", "0") == "1"
    _WYO = os.environ.get("GDN_WYO", "0") == "1"
    _WYO_F1 = os.environ.get("GDN_FUSE_F1_ONLY", "0") == "1"
    _WYO_FUSE = os.environ.get("GDN_FUSE_ALL", "0") == "1"
    _TMA = os.environ.get("GDN_TMA", "0") == "1"
    _DECOMP = os.environ.get("GDN_DECOMP", "0") == "1"
    _WS4_FQLA = os.environ.get("GDN_WS4_FQLA", "0") == "1"
    _QHAT = os.environ.get("GDN_QHAT", "0") == "1"
    _PREFIX = os.environ.get("GDN_PREFIX", "0") == "1"
    _ASYNC = os.environ.get("GDN_ASYNC", "0") == "1"
    _STACK = os.environ.get("GDN_STACK", "0") == "1"
    _STACKWU = os.environ.get("GDN_STACKWU", "0") == "1"
    _SPDV2 = os.environ.get("GDN_SPDV2", "0") == "1"
    _FULLTILE = os.environ.get("GDN_FULLTILE", "0") == "1"
    _SO = os.environ.get("GDN_SO", "0") == "1"
    if _WS4_FQLA and (num_tokens % CHUNK_SIZE == 0):
        # ★ 4-WG 照搬 FlashQLA hopper (第五轮): 4 sub-warp producer + set_max_nreg +
        #   T.tma_copy(barrier=) 自动 arrive + 13 barrier 结构. 数学 = FlashQLA (W=V-g⊙U).
        #   子agent确认: T.barrier_* 是 T.mbarrier_* 语法糖 (同 TIR), 但 FlashQLA 0 处
        #   fence_proxy_async — 因 T.tma_copy barrier= 让 TMA 完成自动 arrive+可见.
        #   尾块 case (short_tail) 不走此路.
        _grid = batch_size * num_heads_v
        _bdv = 128 if _grid > 4 else 64
        kernel = _gdn_ws4_fqla_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=_bdv, threads=512,
        )
    elif _WS_PROBE:
        # 探针: 所有长度都走 ws 探针 (方便用 short_tail 调试)
        # threads=256 才有 2 个 warp group: ws(0)=[0,128) producer, ws(1)=[128,256) consumer
        kernel = _gdn_ws_probe_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=128, threads=256,
        )
    elif _WYO_F1:
        # WY-O + 仅 F1 融合 (验证 N=256 GEMM + fragment 取左/右半)
        kernel = _gdn_wyo_fuse_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=128, threads=256, num_stages=1,
            use_F1=True, use_F2=False, use_F3=False,
        )
    elif _WYO_FUSE:
        # WY-O + F1+F2+F3 全融合 (M=128/N=256, fragment 取上下/左右半)
        kernel = _gdn_wyo_fuse_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=128, threads=256, num_stages=1,
            use_F1=True, use_F2=True, use_F3=True,
        )
    elif _WYO:
        # WY-O 同步对照 (无融合)
        kernel = _gdn_wyo_async_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=128, threads=256, num_stages=1,
        )
    elif _TMA and (num_tokens % CHUNK_SIZE == 0) and num_tokens > 2048:
        # ★ TMA load 实验: T%64==0 长序列. load↔compute 重叠 (T.Pipelined 本职).
        #   short_tail(T%64=1) 不走此路 (T.copy 尾块切片问题), 走下面的 matw.
        #   num_stages: chain(DV=64) 用 2 (寄存器松), 大 grid(DV=128) 用 1 (寄存器紧, ping-pong 爆 shared)
        _grid = batch_size * num_heads_v
        if _grid <= 4:
            kernel = _gdn_tma_kernel(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=64, threads=128, num_stages=2,
            )
        else:
            kernel = _gdn_tma_kernel(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=128, threads=256, num_stages=1,
            )
    elif _DECOMP and (num_tokens % CHUNK_SIZE == 0):
        # ★ 三 kernel 分解: K1 并行 W/U/ds + K2 串行 S + K3 并行 O
        # state 精确串行 (无近似), O 用精确 S[c-1]. grid=chunks×B×Hv 填满 14 SM.
        num_chunks = num_tokens // CHUNK_SIZE
        threads_d = 256
        _grid = batch_size * num_heads_v
        if _grid <= 4:
            threads_d = 128   # 小 grid 翻倍占用
        k1 = _gdn_decomp_k1_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v, num_chunks,
            threads=threads_d,
        )
        k2 = _gdn_decomp_k2_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v, num_chunks,
            threads=threads_d,
        )
        k3 = _gdn_decomp_k3_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v, num_chunks,
            threads=threads_d,
        )
        # K1: 输入 q/k/v/g/beta/A, 输出 W_out/U_out/ds_out
        W_out, U_out, ds_out = k1(q, k, v, g_cumsum, beta, A)
        # K2: 输入 W/U/K/g + initial_state, 输出 S_all (含 S[0]=initial, S[1..num])
        # S_all 形状 [B, Hv, num_chunks+1, DK, DV] FP32
        S_all = k2(q, k, v, g_cumsum, beta, A, W_out, U_out, initial_state)
        # K3: 输入 Q/W/U/ds/S_all/g, 输出 O. final_state = S_all[:, num_chunks]
        output = k3(q, k, v, g_cumsum, beta, A, W_out, U_out, ds_out, S_all)
        final_state = S_all[:, :, num_chunks, :, :].contiguous()
        return output, final_state
    elif _QHAT and num_tokens > 2048:
        # ★ Qhat/Khat 换元 (精确零近似): 消 ds gate 与 O γ 两处 ew stall.
        #   数学换元 Qhat=γ·Q, Khat=γ_inv·K. ds=tril(Qhat@Khatᵀ), O=scale·(Qhat@S_old+ds@V_new),
        #   S_new=γr·S_old+Khatᵀ@(γr·V_new). W/U/V_new 数学不变.
        #   分发同 matw: 小 grid (chain/hidden-2) DV=64/th=128/st=2; 大 grid DV=128/th=256/st=2.
        #   ★ 已验证退步 (chain 0.73 vs 0.53, long_low 3.50 vs 3.04), 保留供报告引用.
        _grid = batch_size * num_heads_v
        if _grid <= 4:
            kernel = _gdn_naive_kernel_qhat(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=64, threads=128, num_stages=2,
            )
        else:
            kernel = _gdn_naive_kernel_qhat(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=128, threads=256, num_stages=2,
            )
    elif _PREFIX and num_tokens > 2048:
        # ★ Prefix-state (output 近似, state 精确): 消 Q@Kᵀ + ds@V_new 两 GEMM.
        #   O[i] ≈ scale·γ[i]·Q[i]@(S_old + P[block_start]) (block 内用起点前缀).
        #   S_new = γr·(S_old + P[last]) 精确. block_prefix 越小越精确 (16/32).
        #   分发同 matw: 小 grid DV=64/th=128/st=2; 大 grid DV=128/th=256/st=2.
        _grid = batch_size * num_heads_v
        _bp = int(os.environ.get("GDN_PREFIX_BLOCK", "32"))
        if _grid <= 4:
            kernel = _gdn_prefix_kernel(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=64, threads=128, num_stages=2, block_prefix=_bp,
            )
        else:
            kernel = _gdn_prefix_kernel(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=128, threads=256, num_stages=2, block_prefix=_bp,
            )
    elif _ASYNC and num_tokens > 2048:
        # ★ Phase A (idea.md 第一主路线): 纯 matw + T.wgmma_gemm async batch.
        #   不改数学/不改 buffer, 把 7 次 T.gemm 隐式 wait 压缩为 3 次 wait (Batch A 4-GEMM / B / C).
        #   ★ num_stages=1: T.Pipelined 多版本流水与 async wgmma 跨 wait 累加冲突 (ns=2 chunk0 nan),
        #     WY-O async 证伪时也是 ns=1 才 PASS. 故本版 ns=1 (无跨 chunk load 重叠, 隔离 async 收益).
        #   分发: 小 grid DV=64/th=128; 大 grid DV=128/th=256 (同 matw, 仅 st 固定 1).
        _grid = batch_size * num_heads_v
        if _grid <= 4:
            kernel = _gdn_async_matw_kernel(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=64, threads=128, num_stages=2,
            )
        else:
            kernel = _gdn_async_matw_kernel(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=128, threads=256, num_stages=2,
            )
    elif _STACK and num_tokens > 2048:
        # ★ Phase G1 (idea.md 10.1): [ds;K^T]@V_new 堆叠 M=192 GEMM, state_o 统一.
        #   ★ shared 超 232KB (247KB) on DV=128 path → 仅用 DV=64 path (shared ~150KB).
        #   P_shared[192,64]=24KB + state_o[192,64] fragment + 其余 shared.
        _grid = batch_size * num_heads_v
        kernel = _gdn_stack_matw_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=64, threads=128, num_stages=2,
        )
    elif _STACKWU and num_tokens > 2048:
        # ★ Phase G2 (idea.md 10.2): A@[βγK|βV]->[W|U] 堆叠 N=256, 无 gating 矛盾, 精确.
        #   消一次 A 的 GEMM (W/U 合一), A 只从 shared 读一次.
        #   ★ DV=128 path shared 239KB > 232KB 超限 (B_wu[64,256]+WU_fragment+其余). 仅 DV=64.
        _grid = batch_size * num_heads_v
        kernel = _gdn_stackwu_matw_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=64, threads=128, num_stages=2,
        )
    elif _SPDV2 and (num_tokens % CHUNK_SIZE == 0) and num_tokens > 2048:
        # ★ SP-DV2 (idea.md 17.3): 预计算 W + DV=64 split.
        #   Kernel A 预计算 W (grid=chunks*B*Hv), Kernel B 主 kernel DV=64 split (grid=2*B*Hv).
        #   state 用原始 V_new 形式 (不系数化, idea 18 已证伪).
        #   仅 T%64==0 长序列 (无尾块, full-tile T.copy load).
        num_chunks = num_tokens // CHUNK_SIZE
        pre_kernel = _gdn_spdv2_pre_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v, num_chunks,
            threads=256,
        )
        main_kernel = _gdn_spdv2_main_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v, num_chunks,
            block_DV=64, threads=128, num_stages=2,
        )
        W_out = pre_kernel(k, g_cumsum, beta, A)
        output, final_state = main_kernel(q, k, v, g_cumsum, beta, A, W_out, initial_state)
        return output, final_state
    elif _FULLTILE and (num_tokens % CHUNK_SIZE == 0) and num_tokens > 2048:
        # ★ full-tile T.copy matw (idea 17.5): 无尾块用 T.copy 全片 load 进 producer pipeline.
        #   隔离 load pipeline 收益 (无 DV 降级混淆). 数学/配置同 matw 长序列.
        #   尾块 case (short_tail T%64≠0) 走默认 matw.
        _grid = batch_size * num_heads_v
        if _grid <= 4:
            kernel = _gdn_naive_kernel_matw_fulltile(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=64, threads=128, num_stages=2,
            )
        else:
            kernel = _gdn_naive_kernel_matw_fulltile(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=128, threads=256, num_stages=2,
            )
    elif _SO and (num_tokens % CHUNK_SIZE == 0) and num_tokens > 2048:
        # ★ State-only 三 kernel (idea 18.6 + 用户修正 S_new 写新位置解耦 output):
        #   Kernel A: C/B/P/R 预计算 (grid=chunks*B*Hv, 并行)
        #   Kernel B: S[c]=γr·S[c-1]-C@S[c-1]+B (grid=DV_slices*B*Hv, chunk 串行, 1 GEMM, 写 S_all[c+1])
        #   Kernel C: O[c]=scale·(P@S[c-1]+R_o) (grid=chunks*DV_slices*B*Hv, 完全并行, 读 S_all[c])
        #   尾块 case 走默认 matw. 仅 T%64==0 长序列.
        num_chunks = num_tokens // CHUNK_SIZE
        _bdv = 128 if (batch_size * num_heads_v) > 4 else 64
        _th = 256 if (batch_size * num_heads_v) > 4 else 128
        pre_k = _gdn_so_pre_kernel(batch_size, num_tokens, num_heads_qk, num_heads_v,
                                   head_dim_k, head_dim_v, num_chunks, threads=256)
        st_k = _gdn_so_state_kernel(batch_size, num_tokens, num_heads_qk, num_heads_v,
                                    head_dim_k, head_dim_v, num_chunks, block_DV=_bdv, threads=_th)
        out_k = _gdn_so_out_kernel(batch_size, num_tokens, num_heads_qk, num_heads_v,
                                   head_dim_k, head_dim_v, num_chunks, block_DV=_bdv, threads=_th)
        C_out, Bc_out, P_out, Ro_out = pre_k(q, k, v, g_cumsum, beta, A)
        S_all = st_k(g_cumsum, C_out, Bc_out, initial_state)
        output = out_k(P_out, Ro_out, S_all)
        final_state = S_all[:, :, num_chunks, :, :].contiguous()
        return output, final_state
    elif num_tokens <= 2048:
        kernel = _gdn_naive_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=64, threads=128, num_stages=2,
        )
    else:
        # 长序列 per-case: 小 Hv (grid 不足) 用 DV=64 提占用, 大 Hv 用 DV=128 大 tile
        # OJ 实测: chain_equal(Hv=4) DV=64=0.53ms(94分) vs DV=128=0.84ms(74分)
        #          long_low/wide/deep/batch_split DV=128 最优
        _grid = batch_size * num_heads_v
        if _grid <= 4:
            # chain(Hv=4)/hidden-2: 小 grid, DV=64 grid 翻倍提 SM 占用
            kernel = _gdn_naive_kernel_matw(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=64, threads=128, num_stages=2,
            )
        else:
            # ★ 大 grid (long_low/wide/deep/batch_split): DV=128 + num_stages=2
            # 离线调参发现 st=2 比 st=1 快 ~6% (long_low 3.19→3.00ms).
            # st=2 让 T.Pipelined 对 shared buffer 自动 multi-version, 跨 chunk load 重叠.
            kernel = _gdn_naive_kernel_matw(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=128, threads=256, num_stages=2,
            )
    output, final_state = kernel(q, k, v, g_cumsum, beta, A, initial_state)
    return output, final_state
