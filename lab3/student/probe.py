# 探测不同 block_DV 的 shared mem 和 launch_bounds (无依赖, 纯 TileLang)
import re
import tilelang
import tilelang.language as T

CHUNK_SIZE = 64


@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def probe_kernel(B, S, Hv, DK, DV, block_DV, threads, num_stages):
    block_S = CHUNK_SIZE
    num_chunks = (S + block_S - 1) // block_S

    QK_shape = (B, S, Hv, DK)
    gate_shape = (B, S, Hv)
    A_shape = (B, S, Hv, block_S)
    init_shape = (B, Hv, DK, DV)
    O_shape = (B, S, Hv, DV)
    final_shape = (B, Hv, DK, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(QK_shape, dtype=T.bfloat16),
        K: T.Tensor(QK_shape, dtype=T.bfloat16),
        V: T.Tensor(QK_shape, dtype=T.bfloat16),
        g: T.Tensor(gate_shape, dtype=T.float32),
        beta: T.Tensor(gate_shape, dtype=T.float32),
        A: T.Tensor(A_shape, dtype=T.bfloat16),
        st: T.Tensor(init_shape, dtype=T.float32),
        O: T.Tensor(O_shape, dtype=T.bfloat16),
        fs: T.Tensor(final_shape, dtype=T.float32),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * Hv, threads=threads) as (bv, bbh):
            bb = bbh // Hv
            bh = bbh % Hv
            s_shared = T.alloc_shared((DK, block_DV), dtype=T.bfloat16)
            s_fragment = T.alloc_fragment((DK, block_DV), dtype=T.float32)
            T.copy(st[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], s_shared)
            T.copy(s_shared, s_fragment)
            for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                T.copy(s_fragment, s_shared)
                T.gemm(s_shared, s_shared, s_fragment)
            T.copy(s_fragment, fs[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


for bdv in [32, 64, 128]:
    try:
        k = probe_kernel(1, 2048, 16, 128, 128, bdv, 128, 2)
        src = k.get_kernel_source()
        lb = re.findall(r"__launch_bounds__\s*\(\s*(\d+)(?:,\s*(\d+))?\)", src)
        # shared memory 大小: 找 extern __shared__ 或 dynamic shared
        smem_dyn = re.findall(r"shared_memory_size\s*=\s*(\d+)", src)
        smem_static = re.findall(r"extern\s+__shared__\s+\w+\s+(\w+)\s*\[\s*(\d+)\s*\]", src)
        print(f"bdv={bdv}: launch_bounds={lb} dyn_smem={smem_dyn[:2]} srclen={len(src)}")
    except Exception as e:
        print(f"bdv={bdv}: ERROR {type(e).__name__}: {str(e)[:150]}")
