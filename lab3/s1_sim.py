# S1: end-to-end torch simulation of the THREE-kernel idea 18.6 structure,
# including the user's correction (S_new written to S_all[c+1], S_old at S_all[c] stays for output kernel).
# Verifies: CB/PR coefficient computation + state-only recurrence + deferred output, FP32 + BF16 workspace.
# Truncation points mirror matw: W/U/ds/C/B/P/R as BF16 workspace, S in FP32 accumulator, S_old snapshot BF16 for GEMM.
import torch
import sys
sys.path.insert(0, ".")
from references.torch_gdr import ref_chunk_gated_delta_rule, torch_chunk_local_cumsum, torch_kkt_solve, _expand_qk_heads
from evaluation.support import make_inputs, load_cases, RTOL, ATOL
import pathlib

CHUNK = 64


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def three_kernel(q, k, v, g_cumsum, beta, A, state0, dtype, ws_bf16=True):
    """Simulate three-kernel structure.
    Kernel A: compute W/U/ds, then C/B/P/R per chunk, write workspace.
    Kernel B: state recurrence S[c]=eg_last*S[c-1]-C@S[c-1]+B, write S_all[c+1].
    Kernel C: output O[c]=scale*(P@S[c-1]+R), read S_all[c].
    """
    B_, T, Hq, DK = q.shape
    _, _, Hv, DV = v.shape
    G = Hv // Hq
    num_chunks = T // CHUNK
    scale = DK ** -0.5

    # Kernel A: precompute workspace (per chunk, per head)
    # C: [DK,DK], B: [DK,DV], P: [block_S,DK], R: [block_S,DV]
    Cws = torch.zeros(B_, Hv, num_chunks, DK, DK, dtype=dtype, device=q.device)
    Bws = torch.zeros(B_, Hv, num_chunks, DK, DV, dtype=dtype, device=q.device)
    Pws = torch.zeros(B_, Hv, num_chunks, CHUNK, DK, dtype=dtype, device=q.device)
    Rws = torch.zeros(B_, Hv, num_chunks, CHUNK, DV, dtype=dtype, device=q.device)

    for s in range(0, T, CHUNK):
        e = s + CHUNK
        ci = s // CHUNK
        for h in range(Hv):
            hg = h // G
            qh = q[0, s:e, hg, :].to(dtype)
            kh = k[0, s:e, hg, :].to(dtype)
            vh = v[0, s:e, h, :].to(dtype)
            gh = g_cumsum[0, s:e, h].to(dtype)
            bh = beta[0, s:e, h].to(dtype)
            Ah = A[0, s:e, h, :].to(dtype)
            eg = torch.exp(gh)
            g_last = gh[-1]
            eg_last = torch.exp(g_last)
            Rdiag = torch.exp(g_last - gh)  # [chunk]
            # W, U
            bkg = kh * (bh[:, None] * eg[:, None])
            bv = vh * bh[:, None]
            W = Ah @ bkg
            U = Ah @ bv
            if ws_bf16:
                W = bf16(W); U = bf16(U)
            # ds = tril(Q@K^T) * (eg_i * eg_inv_j)
            QK = qh @ kh.T
            eg_inv = torch.exp(-gh)
            ds = torch.tril(QK) * (eg[:, None] * eg_inv[None, :])
            if ws_bf16:
                ds = bf16(ds)
            # C = K^T @ (Rdiag * W), B = K^T @ (Rdiag * U)
            RW = Rdiag[:, None] * W
            RU = Rdiag[:, None] * U
            if ws_bf16:
                RW = bf16(RW); RU = bf16(RU)
            C = kh.T @ RW
            Bc = kh.T @ RU
            if ws_bf16:
                C = bf16(C); Bc = bf16(Bc)
            Cws[0, h, ci] = C
            Bws[0, h, ci] = Bc
            # P = eg*Q - ds@W, R_o = ds@U
            dsW = ds @ W
            dsU = ds @ U
            if ws_bf16:
                dsW = bf16(dsW); dsU = bf16(dsU)
            P = eg[:, None] * qh - dsW
            if ws_bf16:
                P = bf16(P)
            Pws[0, h, ci] = P
            Rws[0, h, ci] = dsU

    # Kernel B: state recurrence (chunk serial), S_all[0]=state0
    S_all = torch.zeros(B_, Hv, num_chunks + 1, DK, DV, dtype=torch.float32, device=q.device)
    for bb in range(B_):
        for h in range(Hv):
            S_all[bb, h, 0] = state0[bb, h].to(torch.float32)
    for ci in range(num_chunks):
        for h in range(Hv):
            C = Cws[0, h, ci]
            Bc = Bws[0, h, ci]
            s = ci * CHUNK
            gh = g_cumsum[0, s:s + CHUNK, h]
            g_last = gh[-1]
            eg_last = torch.exp(g_last)
            S_old = S_all[0, h, ci]  # FP32
            # C@S_old: C is workspace (BF16 or FP32), S_old FP32 -> GEMM; snapshot S_old to match kernel
            if ws_bf16:
                S_old_snap = bf16(S_old)
            else:
                S_old_snap = S_old
            CS = C @ S_old_snap
            S_all[0, h, ci + 1] = eg_last * S_old - CS + Bc

    # Kernel C: output (fully parallel over chunks)
    out = torch.zeros(B_, T, Hv, DV, dtype=dtype, device=q.device)
    for ci in range(num_chunks):
        s = ci * CHUNK
        e = s + CHUNK
        for h in range(Hv):
            P = Pws[0, h, ci]
            Ro = Rws[0, h, ci]
            S_old = S_all[0, h, ci]  # FP32
            if ws_bf16:
                S_old_snap = bf16(S_old)
            else:
                S_old_snap = S_old
            O = scale * (P @ S_old_snap + Ro)
            out[0, s:e, h, :] = O
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
    # FP32 workspace
    out_fp, state_fp = three_kernel(inp.q, inp.k, inp.v, g_cumsum, inp.beta, A, st0.clone(), torch.float32, ws_bf16=False)
    # BF16 workspace (kernel-matched truncation)
    out_bf, state_bf = three_kernel(inp.q, inp.k, inp.v, g_cumsum, inp.beta, A, st0.clone(), torch.float32, ws_bf16=True)
    se_fp = (state_fp - state_ref).abs().max().item()
    se_bf = (state_bf - state_ref).abs().max().item()
    oe_bf = (out_bf.float() - out_ref.float()).abs().max().item()
    orel = ((out_bf.float() - out_ref.float()).abs() / (out_ref.float().abs() + 1e-3)).max().item()
    srel = ((state_bf - state_ref).abs() / (state_ref.abs() + 1e-3)).max().item()
    print(f"{c.name:20s} T={c.seqlen:5d} | state_fp={se_fp:.2e} state_bf={se_bf:.2e}(rel {srel:.2e}) "
          f"out_bf={oe_bf:.2e}(rel {orel:.2e}) {'PASS' if se_bf < ATOL and oe_bf < ATOL else 'FAIL'}")
