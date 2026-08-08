# 探测不同 block_DV / threads 的 shared mem 和 launch_bounds
# 独立运行, 不依赖 torch/student 包
import sys
sys.path.insert(0, ".")
import re
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
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)
            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)
            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                T.copy(s_fragment, s_shared)
                T.gemm(s_shared, s_shared, s_fragment)

    return kernel


for bdv in [32, 64, 128]:
    for thr in [128, 256]:
        try:
            k = _gdn_naive_kernel(1, 2048, 16, 16, 128, 128, bdv, thr, 2)
            src = k.get_kernel_source()
            lb = re.findall(r"__launch_bounds__\s*\(\s*(\d+)(?:,\s*(\d+))?\)", src)
            print(f"bdv={bdv} thr={thr}: launch_bounds={lb} srclen={len(src)}")
        except Exception as e:
            print(f"bdv={bdv} thr={thr}: ERROR {type(e).__name__}: {str(e)[:120]}")
