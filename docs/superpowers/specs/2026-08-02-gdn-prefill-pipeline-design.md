# GDN Prefill Forward: Three-Stage Chunk Pipeline Design

**Date:** 2026-08-02
**Scope:** `lab3/student/tilelang_fwd.py`

## Objective

Replace the incomplete student stub with a TileLang-only GDN prefill-forward implementation. Preserve the established function interface and numerical semantics for normal heads, GVA, tails shorter than 64 tokens, and optional FP32 initial state.

## Per-chunk mathematical stages

For chunk `c`, define `gamma = exp(g_cumsum)` and:

- `W = A @ (beta * K * gamma)`
- `U = A @ (beta * V)`
- `D = U - W @ S_old`
- `S_new = gamma_last * S_old + K_weighted.T @ D`, where `K_weighted = K * exp(gamma_log_last - g_cumsum)`
- `output_from_state = scale * gamma * (Q @ S_old)`
- `output_in_chunk = scale * (Lower(Q @ K.T) * exp(g_i - g_j)) @ D`
- `output = output_from_state + output_in_chunk`

`D` is the reference implementation's `v_new`.

## Pipeline

Use three stages across chunks:

1. **P — input preprocessing:** compute state-independent `W`, `U`, gate factors, and causal score/decay data for a chunk.
2. **S — state critical path:** consume `S_old`, form `D`, write `S_new` immediately for the next chunk, and retain the state-output contribution before old state storage may be reused.
3. **O — output completion:** combine stored `output_from_state`, precomputed causal score/decay, and `D` to write BF16 output. This stage must not delay production of `S_new`.

Chunk-local temporary storage is bounded through ping-pong/ring buffers rather than tensors spanning the full sequence. CUDA stream/event dependencies will ensure each consumer sees finished producer data and no buffer slot is reused prematurely.

## Correctness requirements

- Inputs q/k are BF16 and may have `Hq < Hv`; map `value_head` to `qk_head = value_head // (Hv // Hq)`.
- `A[..., :length]` is transposed to the logical `[length, length]` lower-triangular inverse for a chunk. Mask tail positions where `length < 64`.
- Maintain all state and sensitive accumulations in FP32; return output in BF16 and final state in FP32.
- Treat a missing initial state as a zero FP32 matrix and do not mutate a supplied initial-state tensor.
- Do not invoke PyTorch reference implementations for the measured computation.

## Validation

Run the supplied correct-and-benchmark runner in the course GPU environment, beginning with `short_tail_state`, then all public cases. Use Nsight profiling only after correctness passes.
