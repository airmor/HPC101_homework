# S1b: fix per-batch for batch_split, and check if C@S_old with BF16 C + BF16 S_old snap
# accumulates too much error. Try: C FP32 (not BF16 workspace), since C drives cross-chunk state.
import torch
import sys
sys.path.insert(0, ".")
from references.torch_gdr import ref_chunk_gated_delta_rule, torch_chunk_local_cumsum, torch_kkt_solve
from evaluation.support import make_inputs, load_cases, RTOL, ATOL
import pathlib

CHUNK = 64


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def three_kernel(q, k, v, g_cumsum, beta, A, state0, dtype, c_fp32=False):
    B_, T, Hq, DK = q.shape
    _, _, Hv, DV = v.shape
    G = Hv // Hq
    num_chunks = T // CHUNK
    scale = DK ** -0.5
    ws_dt = torch.float32 if c_fp32 else dtype
    Cws = torch.zeros(B_, Hv, num_chunks, DK, DK, dtype=ws_dt, device=q.device)
    Bws = torch.zeros(B_, Hv, num_chunks, DK, DV, dtype=dtype, device=q.device)
    Pws = torch.zeros(B_, Hv, num_chunks, CHUNK, DK, dtype=dtype, device=q.device)
    Rws = torch.zeros(B_, Hv, num_chunks, CHUNK, DV, dtype=dtype, device=q.device)

    for bb in range(B_):
        for s in range(0, T, CHUNK):
            e = s + CHUNK
            ci = s // CHUNK
            for h in range(Hv):
                hg = h // G
                qh = q[bb, s:e, hg, :].to(dtype)
                kh = k[bb, s:e, hg, :].to(dtype)
                vh = v[bb, s:e, h, :].to(dtype)
                gh = g_cumsum[bb, s:e, h].to(dtype)
                bh = beta[bb, s:e, h].to(dtype)
                Ah = A[bb, s:e, h, :].to(dtype)
                eg = torch.exp(gh)
                g_last = gh[-1]
                eg_last = torch.exp(g_last)
                Rdiag = torch.exp(g_last - gh)
                bkg = kh * (bh[:, None] * eg[:, None])
                bv = vh * bh[:, None]
                W = Ah @ bkg
                U = Ah @ bv
                if dtype != torch.float64:
                    W = bf16(W); U = bf16(U)
                QK = qh @ kh.T
                eg_inv = torch.exp(-gh)
                ds = torch.tril(QK) * (eg[:, None] * eg_inv[None, :])
                if dtype != torch.float64:
                    ds = bf16(ds)
                RW = Rdiag[:, None] * W
                RU = Rdiag[:, None] * U
                C = kh.T @ RW
                Bc = kh.T @ RU
                # C: keep FP32 if c_fp32, else BF16
                if not c_fp32 and dtype != torch.float64:
                    C = bf16(C); Bc = bf16(Bc)
                Cws[bb, h, ci] = C
                Bws[bb, h, ci] = Bc
                dsW = ds @ W
                dsU = ds @ U
                P = eg[:, None] * qh - dsW
                if dtype != torch.float64:
                    P = bf16(P)
                Pws[bb, h, ci] = P
                Rws[bb, h, ci] = dsU

    S_all = torch.zeros(B_, Hv, num_chunks + 1, DK, DV, dtype=torch.float32, device=q.device)
    for bb in range(B_):
        for h in range(Hv):
            S_all[bb, h, 0] = state0[bb, h].to(torch.float32)
    for ci in range(num_chunks):
        for bb in range(B_):
            for h in range(Hv):
                C = Cws[bb, h, ci]
                Bc = Bws[bb, h, ci]
                s = ci * CHUNK
                gh = g_cumsum[bb, s:s + CHUNK, h]
                g_last = gh[-1]
                eg_last = torch.exp(g_last)
                S_old = S_all[bb, h, ci]
                S_old_snap = bf16(S_old) if dtype != torch.float64 else S_old
                CS = C @ S_old_snap
                S_all[bb, h, ci + 1] = eg_last * S_old - CS + Bc

    out = torch.zeros(B_, T, Hv, DV, dtype=dtype, device=q.device)
    for ci in range(num_chunks):
        s = ci * CHUNK
        e = s + CHUNK
        for bb in range(B_):
            for h in range(Hv):
                P = Pws[bb, h, ci]
                Ro = Rws[bb, h, ci]
                S_old = S_all[bb, h, ci]
                S_old_snap = bf16(S_old) if dtype != torch.float64 else S_old
                O = scale * (P @ S_old_snap + Ro)
                out[bb, s:e, h, :] = O
    return out, S_all[:, :, -1, :, :]


cases = load_cases(pathlib.Path("evaluation/cases.csv"))
for c in cases:
    if c.seqlen % CHUNK != 0 or c.seqlen > 8192:
        continue
    inp = make_inputs(c)
    g_cumsum = torch_chunk_local_cumsum(inp.g)
    A = torch_kkt_solve(inp.k, g_cumsum, inp.beta)
    B = c.batch_size
    st0 = torch.zeros(B, c.num_heads_v, 128, 128, dtype=torch.float32, device="cuda")
    if inp.initial_state is not None:
        st0 = inp.initial_state.clone()
    out_ref, state_ref = ref_chunk_gated_delta_rule(inp.q, inp.k, inp.v, inp.g, inp.beta, inp.initial_state)
    state_ref = state_ref.to(torch.float32)
    # BF16 workspace, C FP32 (c_fp32=True)
    out_bf, state_bf = three_kernel(inp.q, inp.k, inp.v, g_cumsum, inp.beta, A, st0.clone(), torch.float32, c_fp32=True)
    se = (state_bf - state_ref).abs().max().item()
    oe = (out_bf.float() - out_ref.float()).abs().max().item()
    srel = ((state_bf - state_ref).abs() / (state_ref.abs() + 1e-3)).max().item()
    orel = ((out_bf.float() - out_ref.float()).abs() / (out_ref.float().abs() + 1e-3)).max().item()
    print(f"{c.name:20s} T={c.seqlen:5d} B={B} | state_bf={se:.2e}(rel {srel:.2e}) "
          f"out_bf={oe:.2e}(rel {orel:.2e}) {'PASS' if se < ATOL and oe < ATOL else 'FAIL'}")
