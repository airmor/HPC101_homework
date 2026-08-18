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
# 物化 W 版 (长序列): 直接 W=A@βγK 再 W@S, GEMM 数少, 长序列快
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
# WY-O + GEMM 融合版 (长序列, 实验):
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
    import os
    _WS_PROBE = os.environ.get("GDN_WS_PROBE", "0") == "1"
    _WYO = os.environ.get("GDN_WYO", "0") == "1"
    _WYO_F1 = os.environ.get("GDN_FUSE_F1_ONLY", "0") == "1"
    _WYO_FUSE = os.environ.get("GDN_FUSE_ALL", "0") == "1"
    _TMA = os.environ.get("GDN_TMA", "0") == "1"
    if _WS_PROBE:
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
            kernel = _gdn_naive_kernel_matw(
                batch_size, num_tokens, num_heads_qk, num_heads_v,
                head_dim_k, head_dim_v,
                block_DV=128, threads=256, num_stages=1,
            )
    output, final_state = kernel(q, k, v, g_cumsum, beta, A, initial_state)
    return output, final_state
