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
    # ★ 调试开关: WS_PROBE=1 跑 2-WG 探针验证 warp-spec API; =0 走 matw 版 (92分基线)
    import os
    _WS_PROBE = os.environ.get("GDN_WS_PROBE", "0") == "1"
    if _WS_PROBE:
        # 探针: 所有长度都走 ws 探针 (方便用 short_tail 调试)
        # threads=256 才有 2 个 warp group: ws(0)=[0,128) producer, ws(1)=[128,256) consumer
        kernel = _gdn_ws_probe_kernel(
            batch_size, num_tokens, num_heads_qk, num_heads_v,
            head_dim_k, head_dim_v,
            block_DV=128, threads=256,
        )
    elif num_tokens <= 2048:
        kernel = _gdn_naive_kernel(
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
