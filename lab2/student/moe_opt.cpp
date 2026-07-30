// Main task: optimize the MoE forward pass.
//
// RISC-V (RVV 1.0) port of the AVX-512 implementation.
//   - Target: SpaceMiT Muse Pi Pro, VLEN = 256, ELEN = 64.
//   - Pure RVV (<riscv_vector.h>); the SpaceMiT IME (vmadot) matrix extension
//     is NOT used in this revision.
//   - The vector width is fixed to VL = 16 lanes (vfloat32m2 / vint32m2),
//     matching the original AVX-512 16-lane structure. d_model, d_ff and
//     num_experts are all multiples of 16 (in fact of 64), so 16 divides
//     every loop cleanly.
//   - Weight layout: repacked to a GEMV-friendly [k][row] order so that, for
//     a fixed K-index, the 16 weights of one 16-row tile are contiguous and
//     loadable with a unit-stride vle8. Arithmetic (the +128 -> unsigned
//     offset and the -128*sum correction) is identical to the AVX version.

#include "moe.h"
#include <cstdint>     // int8_t
#include <cstring>
#include <cmath>        // INFINITY
#include <riscv_vector.h>  // RVV intrinsics：vfloat32m2_t / vint32m2_t / __riscv_v*
#include <cassert>
#include <omp.h>
#define IS_AMX 0

#if IS_AMX
// (AMX / x86 tile config — kept verbatim from the AVX version for reference,
//  inactive on RISC-V because IS_AMX == 0.)
struct alignas(64) amx_tilecfg {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};
static inline void load_amx_gemv_config()
{
    amx_tilecfg cfg{};
    cfg.palette_id = 1;
    // tmm0: gate/down accumulator, 1 x 16 int32
    cfg.rows[0]  = 1;
    cfg.colsb[0] = 16 * sizeof(int32_t);   // 64 bytes
    // tmm1: up accumulator, 1 x 16 int32
    cfg.rows[1]  = 1;
    cfg.colsb[1] = 16 * sizeof(int32_t);   // 64 bytes
    // tmm2: activation, 1 x 64 int8
    cfg.rows[2]  = 1;
    cfg.colsb[2] = 64;
    // tmm3: gate/down weights, packed 16 x 64 bytes
    cfg.rows[3]  = 16;
    cfg.colsb[3] = 64;
    // tmm4: up weights, packed 16 x 64 bytes
    cfg.rows[4]  = 16;
    cfg.colsb[4] = 64;
    _tile_loadconfig(&cfg);
}
#endif

float* w_router_transpose;
uint8_t* w_sh_gate_transpose;
uint8_t* w_sh_up_transpose;
uint8_t* w_sh_down_transpose;
uint8_t* w_gate_transpose;
uint8_t* w_up_transpose;
uint8_t* w_down_transpose;

// Fixed vector length: 16 lanes (matches the original AVX-512 16-wide path).
// VLEN = 256 => e32m2 / e8mf2 vlmax = 16.
static constexpr size_t VL = 16;

// Max tokens per batched expert FFN call. Capped to keep RVV register pressure
// within 32 vector registers: B=4 with m2 accumulators (2 regs each) + m1
// weight regs = 4*4 + 2 = 18 regs. B=8 would need 34 (overflow).
// Stack: h_buf 4*512*4 = 8 KB per call, 4-thread = 32 KB. Fine.
static constexpr int MAX_BATCH = 4;

// Shared workspaces for the dispatch paths (RV-001 flatten + RV-002 batched).
// The flatten path uses MAX_D_MODEL stride for expert_out_workspace (each
// (token, slot) gets a fixed-stride output buffer); the batched path uses a
// runtime d_model stride inside its own indexing. xq_workspace always uses
// MAX_D_MODEL stride so a quantized token can be addressed by either path.
static int8_t  xq_workspace[MAX_NUM_TOKENS * MAX_D_MODEL];
static float   x_scale_workspace[MAX_NUM_TOKENS];
static int     topk_idx_workspace[MAX_NUM_TOKENS * MAX_TOP_K];
static float   topk_score_workspace[MAX_NUM_TOKENS * MAX_TOP_K];
static float   gate_sum_workspace[MAX_NUM_TOKENS];
static float   expert_out_workspace[MAX_NUM_TOKENS * (MAX_TOP_K + 1) * MAX_D_MODEL];
static int     task_token_workspace[MAX_NUM_TOKENS * MAX_TOP_K];
static int     task_slot_workspace[MAX_NUM_TOKENS * MAX_TOP_K];

void preprocess(MoEWeights& w) {
    //change w.router to w.router_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    assert(w.num_experts % 16 == 0);
    w_router_transpose = new float[w.num_experts * w.d_model];
    for (int e = 0; e < w.num_experts; e+=16) {
        for (int d = 0; d < w.d_model; ++d) {
            for (int i = 0; i < 16; ++i) {
                w_router_transpose[e * w.d_model + d * 16 + i] = w.w_router[e * w.d_model + i * w.d_model + d];
            }
        }
    }
    //delete[] w.w_router; // free the original w.router
    //w.w_router = w_router_transpose;

    // RVV weight packing: repack int8 [rows][cols] weights into the layout.
    // Same 16-row tile blocking as the AVX VNNI version, but the intra-tile
    // order is [k][row] (K-index outer, row inner) so that the 16 weights of a
    // single K-index are contiguous -> a unit-stride vle8 of 16 bytes.
    // Globally this is a transpose to [cols][rows] (K-major), still offset by
    // +128 to unsigned — identical arithmetic to the VNNI version.
    auto pack_rvv = [](const int8_t* src, uint8_t* dst, int rows, int cols) {
        for (int r = 0; r < rows; r += 16) {
            for (int c = 0; c < cols; c += 4) {
                for (int j = 0; j < 4; ++j) {
                    for (int i = 0; i < 16; ++i) {
                        dst[r * cols + c * 16 + j * 16 + i] =
                            static_cast<uint8_t>(static_cast<int>(src[(r + i) * cols + (c + j)]) + 128);
                    }
                }
            }
        }
    };

    // Shared expert (same shape as one routed expert)
    w_sh_gate_transpose = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_rvv(w.sh_gate, w_sh_gate_transpose, w.d_ff, w.d_model);
    w_sh_up_transpose   = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_rvv(w.sh_up,   w_sh_up_transpose,   w.d_ff, w.d_model);
    w_sh_down_transpose = new uint8_t[(size_t)w.d_model * w.d_ff];
    pack_rvv(w.sh_down, w_sh_down_transpose, w.d_model, w.d_ff);

    // Routed experts: [num_experts][...]
    const size_t gate_up_size = (size_t)w.d_ff * w.d_model;
    const size_t down_size    = (size_t)w.d_model * w.d_ff;
    w_gate_transpose = new uint8_t[(size_t)w.num_experts * gate_up_size];
    w_up_transpose   = new uint8_t[(size_t)w.num_experts * gate_up_size];
    w_down_transpose = new uint8_t[(size_t)w.num_experts * down_size];
    for (int e = 0; e < w.num_experts; ++e) {
        pack_rvv(w.w_gate + e * gate_up_size,
                  w_gate_transpose + (size_t)e * gate_up_size, w.d_ff, w.d_model);
        pack_rvv(w.w_up   + e * gate_up_size,
                  w_up_transpose   + (size_t)e * gate_up_size, w.d_ff, w.d_model);
        pack_rvv(w.w_down + e * down_size,
                  w_down_transpose + (size_t)e * down_size, w.d_model, w.d_ff);
    }
}


// approximate exp.
// Taylor expansion at x=0

// scores[i] = exp(scores[i] - max_score) / sum(exp(scores[i] - max_score))

auto exp512_approx_ps = [](vfloat32m2_t x) -> vfloat32m2_t
{
    // Clamp range to avoid overflow / underflow when constructing 2^n.
    const vfloat32m2_t max_x = __riscv_vfmv_v_f_f32m2(88.3762626647949f, VL);
    const vfloat32m2_t min_x = __riscv_vfmv_v_f_f32m2(-87.3365447505531f, VL);

    x = __riscv_vfmin_vv_f32m2(x, max_x, VL);
    x = __riscv_vfmax_vv_f32m2(x, min_x, VL);

    // exp(x) = 2^n * exp(r)
    // n = round(x / ln2)
    // r = x - n * ln2
    const vfloat32m2_t log2e  = __riscv_vfmv_v_f_f32m2(1.44269504088896341f, VL);
    const vfloat32m2_t ln2_hi = __riscv_vfmv_v_f_f32m2(0.693359375f, VL);
    const vfloat32m2_t ln2_lo = __riscv_vfmv_v_f_f32m2(-2.12194440e-4f, VL);

    const vfloat32m2_t y = __riscv_vfmul_vv_f32m2(x, log2e, VL);

    // n = round-to-nearest(y)  (vfcvt.x.f uses the current rounding mode, RNE)
    const vint32m2_t n = __riscv_vfcvt_x_f_v_i32m2(y, VL);
    const vfloat32m2_t nf = __riscv_vfcvt_f_x_v_f32m2(n, VL);

    // r = x - nf * ln2
    // vfmsac_vv(vd, vs1, vs2): vd = vs1*vs2 - vd  => (nf*ln2 - x) = -(x - nf*ln2)
    vfloat32m2_t neg_r = __riscv_vfmsac_vv_f32m2(x, nf, ln2_hi, VL);
    vfloat32m2_t r = __riscv_vfneg_v_f32m2(neg_r, VL);
    vfloat32m2_t neg_r2 = __riscv_vfmsac_vv_f32m2(r, nf, ln2_lo, VL);
    r = __riscv_vfneg_v_f32m2(neg_r2, VL);

    // exp(r) polynomial approximation.
    // 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
    // Horner: p = coeff + r*prev.  vfmacc_vv(vd, vs1, vs2): vd = vs1*vs2 + vd
    //  => vd=coeff, vs1=r, vs2=prev_p
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(1.0f / 720.0f, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 120.0f, VL), r, p, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 24.0f,  VL), r, p, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 6.0f,   VL), r, p, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 2.0f,   VL), r, p, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f,           VL), r, p, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f,           VL), r, p, VL);

    // Construct 2^n using float exponent bits.
    // float: exponent bias = 127, exponent field starts at bit 23.
    const vint32m2_t pow2_bits = __riscv_vsll_vx_i32m2(
        __riscv_vadd_vx_i32m2(n, 127, VL),
        23, VL);

    const vfloat32m2_t pow2n = __riscv_vreinterpret_v_i32m2_f32m2(pow2_bits);

    return __riscv_vfmul_vv_f32m2(p, pow2n, VL);
};

static void expert_ffn(const uint8_t* w_gate, const uint8_t* w_up,
                       const uint8_t* w_down, float s_gate, float s_up,
                       float s_down, const int8_t* xq, float s_x, float* out,
                       int d_model, int d_ff) {

    assert(d_model % 16 == 0);
    assert(d_ff % 16 == 0);
    assert(d_ff <= MAX_D_FF);

    // Gate / up projections + SwiGLU activation
    float h[MAX_D_FF];
    float h_amax = 0.0f;

    const float s_x_mul_gate = s_x * s_gate;
    const float s_x_mul_up   = s_x * s_up;


    int32_t x_sum = 0;
    {
        // sign-extend int8 -> int32 (vf4 = 4x width) then reduce-sum into int32.
        // i16 would overflow for d_model up to 1024 (sum up to ~130k > 32767).
        vint32m1_t x_sum_vec = __riscv_vmv_v_x_i32m1(0, 1);
        for (int d = 0; d < d_model; d += 16) {
            vint8mf2_t xq_v = __riscv_vle8_v_i8mf2(xq + d, VL);
            vint32m2_t xq_i32 = __riscv_vsext_vf4_i32m2(xq_v, VL);
            x_sum_vec = __riscv_vredsum_vs_i32m2_i32m1(xq_i32, x_sum_vec, VL);
        }
        x_sum = __riscv_vmv_x_s_i32m1_i32(x_sum_vec);
    }

    // correction cancels the +128 weight offset: sum_k 128 * xq[k] = 128 * x_sum
    const vint32m2_t correction = __riscv_vmv_v_x_i32m2(-128 * x_sum, VL);
    vfloat32m2_t h_amax_vec = __riscv_vfmv_v_f_f32m2(0.0f, VL);
    for (int f = 0; f < d_ff; f+=16) {

        vint32m2_t gate_acc = correction;
        vint32m2_t up_acc   = correction;
        const size_t f_offset = static_cast<size_t>(f) * d_model;

        for (int k = 0; k < d_model; k += 4) { // one time x4 (RVV: 4 K-indices)

            const size_t k4_offset = static_cast<size_t>(k / 4) * 64;

            for (int j = 0; j < 4; ++j) { // RVV: one vle8 + widen + vwmaccsu per K-index

                //read xq (scalar activation for this K-index)
                const int8_t xq_k = xq[k + j];

                // tile base for this 16-row / 4-K tile; K-index j is contiguous
                const size_t tile = f_offset + k4_offset + (size_t)j * 16;

                // read w_gate (16 unsigned weights for this K-index), widen u8->u16
                const vuint8mf2_t w_gate_u8 = __riscv_vle8_v_u8mf2(w_gate + tile, VL);
                const vuint16m1_t w_gate_vec = __riscv_vzext_vf2_u16m1(w_gate_u8, VL);
                // read w_up
                const vuint8mf2_t w_up_u8   = __riscv_vle8_v_u8mf2(w_up + tile, VL);
                const vuint16m1_t w_up_vec   = __riscv_vzext_vf2_u16m1(w_up_u8, VL);

                // calculate: acc[i] += xq_k(signed) * w[i](unsigned)
                gate_acc = __riscv_vwmaccsu_vx_i32m2(gate_acc, xq_k, w_gate_vec, VL);
                up_acc   = __riscv_vwmaccsu_vx_i32m2(up_acc,   xq_k, w_up_vec,   VL);

            }

        }

        vfloat32m2_t vg = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(gate_acc, VL), s_x_mul_gate, VL);
        vfloat32m2_t vu = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(up_acc,   VL), s_x_mul_up,   VL);
        vfloat32m2_t neg_vg = __riscv_vfneg_v_f32m2(vg, VL);
        vfloat32m2_t exp_neg_vg = exp512_approx_ps(neg_vg);
        vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_vg, 1.0f, VL);
        vfloat32m2_t silu = __riscv_vfdiv_vv_f32m2(vg, denom, VL);
        vfloat32m2_t h_vec = __riscv_vfmul_vv_f32m2(silu, vu, VL);
        h_amax_vec = __riscv_vfmax_vv_f32m2(h_amax_vec, __riscv_vfabs_v_f32m2(h_vec, VL), VL);
        __riscv_vse32_v_f32m2(&h[f], h_vec, VL); // store h_vec to h[f] array

    }
    h_amax = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(h_amax_vec, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL));

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    float r_s_h = (h_amax > 0.0f) ? 127.0f / h_amax : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; f+=16) {
        vfloat32m2_t h_vec = __riscv_vle32_v_f32m2(&h[f], VL);
        vfloat32m2_t h_scaled = __riscv_vfmul_vf_f32m2(h_vec, r_s_h, VL);
        // round-to-nearest float -> int32
        vint32m2_t h_i32 = __riscv_vfcvt_x_f_v_i32m2(h_scaled, VL);
        // saturating narrow int32 -> int16 -> int8 (two 2x narrowing steps;
        // vnclip with shift 0 saturates, matching _mm512_cvtsepi32_epi8;
        // __RISCV_VXRM_RNU = round-to-nearest-up, matches the round() intent)
        vint16m1_t h_i16 = __riscv_vnclip_wx_i16m1(h_i32, 0, __RISCV_VXRM_RNU, VL);
        vint8mf2_t v_i8 = __riscv_vnclip_wx_i8mf2(h_i16, 0, __RISCV_VXRM_RNU, VL);
        __riscv_vse8_v_i8mf2(&hq[f], v_i8, VL);
    }

    int32_t hq_sum = 0;
    {
        // sign-extend int8 -> int32, then reduce-sum (see x_sum comment)
        vint32m1_t hq_sum_vec = __riscv_vmv_v_x_i32m1(0, 1);
        for (int f = 0; f < d_ff; f += 16) {
            vint8mf2_t hq_v = __riscv_vle8_v_i8mf2(hq + f, VL);
            vint32m2_t hq_i32 = __riscv_vsext_vf4_i32m2(hq_v, VL);
            hq_sum_vec = __riscv_vredsum_vs_i32m2_i32m1(hq_i32, hq_sum_vec, VL);
        }
        hq_sum = __riscv_vmv_x_s_i32m1_i32(hq_sum_vec);
    }

    const float s_x_mul_down = s_h * s_down;

    // Down projection
    for (int d = 0; d < d_model; d += 16) {
        vint32m2_t acc = __riscv_vmv_v_x_i32m2(-128 * hq_sum, VL);
        const size_t d_offset = static_cast<size_t>(d) * d_ff;
        for (int f = 0; f < d_ff; f+=4) { // one time x4 (RVV: 4 K-indices)

            const size_t f4_offset = static_cast<size_t>(f / 4) * 64;

            for (int j = 0; j < 4; ++j) { // RVV: one vle8 + widen + vwmaccsu per K-index

                //read hq (scalar activation for this K-index)
                const int8_t hq_k = hq[f + j];

                const size_t tile = d_offset + f4_offset + (size_t)j * 16;

                // read w_down, widen u8->u16
                const vuint8mf2_t w_down_u8 = __riscv_vle8_v_u8mf2(w_down + tile, VL);
                const vuint16m1_t w_down_vec = __riscv_vzext_vf2_u16m1(w_down_u8, VL);

                // calculate: acc[i] += hq_k(signed) * w[i](unsigned)
                acc = __riscv_vwmaccsu_vx_i32m2(acc, hq_k, w_down_vec, VL);
            }
        }
        vfloat32m2_t acc_f = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(acc, VL), s_x_mul_down, VL);
        __riscv_vse32_v_f32m2(&out[d], acc_f, VL);
    }
}

// -----------------------------------------------------------------------------
// RV-002: batched expert FFN
//
// Same expert weight matrix is shared across B tokens that route to it. The
// original expert_ffn() loads weight tiles once per (token, slot). For S3
// (B≈8 tokens/expert) and S4 (B≈4 tokens/expert) this wastes weight load
// bandwidth B-fold. This batched version loads each weight vector once and
// reuses it across the B tokens before moving on.
//
// Per-token accumulators (h, hq, out) are kept on the stack as separate small
// arrays — no shared writes, no atomics, identical arithmetic to expert_ffn
// when B=1.
//
// Isolation: marked noinline + cold so the large batched body stays out of
// the S1/S2 icache path. The dispatch guard in moe_forward_optimized only
// routes total_tasks >= 32 (S3/S4) to this function.
// -----------------------------------------------------------------------------
static void __attribute__((noinline, cold))
expert_ffn_batch(const uint8_t* w_gate, const uint8_t* w_up,
                 const uint8_t* w_down, float s_gate, float s_up,
                 float s_down,
                 const int8_t* const* xq_list, const float* s_x_list,
                 float* const* out_list, int B,
                 int d_model, int d_ff) {

    assert(d_model % 16 == 0);
    assert(d_ff % 16 == 0);
    assert(d_ff <= MAX_D_FF);

    const float s_x_mul_gate_factor = s_gate;
    const float s_x_mul_up_factor   = s_up;
    const float s_x_mul_down_factor = s_down;

    // Per-token stacks (kept small enough that 4-thread stack usage is fine).
    float  h_buf[MAX_BATCH][MAX_D_FF];
    float  h_amax_buf[MAX_BATCH];
    int8_t hq_buf[MAX_BATCH][MAX_D_FF];
    float  h_amax_all = 0.0f;

    // ------------------------------------------------------------------
    // Stage A: gate / up projections + SwiGLU
    //
    // For each f-block of 16 output rows: load each weight tile ONCE per
    // K-index, then walk the B tokens and accumulate. Weights stay in vector
    // registers across the B loop.
    // ------------------------------------------------------------------

    // Per-token x_sum (for the -128 * x_sum correction).
    int32_t x_sum_buf[MAX_BATCH];
    for (int b = 0; b < B; ++b) {
        int32_t s = 0;
        vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
        for (int d = 0; d < d_model; d += 16) {
            vint8mf2_t xq_v = __riscv_vle8_v_i8mf2(xq_list[b] + d, VL);
            vint32m2_t xq_i32 = __riscv_vsext_vf4_i32m2(xq_v, VL);
            acc = __riscv_vredsum_vs_i32m2_i32m1(xq_i32, acc, VL);
        }
        s = __riscv_vmv_x_s_i32m1_i32(acc);
        x_sum_buf[b] = s;
    }

    for (int b = 0; b < B; ++b) {
        h_amax_buf[b] = 0.0f;
    }

    for (int f = 0; f < d_ff; f += 16) {
        const size_t f_offset = static_cast<size_t>(f) * d_model;

        // Per-token gate/up accumulators for this f-block.
        vint32m2_t gate_acc0 = __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[0], VL);
        vint32m2_t gate_acc1 = (B > 1) ? __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[1], VL) : gate_acc0;
        vint32m2_t gate_acc2 = (B > 2) ? __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[2], VL) : gate_acc0;
        vint32m2_t gate_acc3 = (B > 3) ? __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[3], VL) : gate_acc0;
        vint32m2_t up_acc0   = __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[0], VL);
        vint32m2_t up_acc1   = (B > 1) ? __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[1], VL) : up_acc0;
        vint32m2_t up_acc2   = (B > 2) ? __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[2], VL) : up_acc0;
        vint32m2_t up_acc3   = (B > 3) ? __riscv_vmv_v_x_i32m2(-128 * x_sum_buf[3], VL) : up_acc0;

        for (int k = 0; k < d_model; k += 4) {
            const size_t k4_offset = static_cast<size_t>(k / 4) * 64;

            // Pre-load 4 K-tiles of gate/up weights (8 m1 vector regs).
            // These stay live across the B-token sweep below.
            const vuint8mf2_t w_gate_u8_0 = __riscv_vle8_v_u8mf2(w_gate + f_offset + k4_offset + 0, VL);
            const vuint16m1_t w_gate_vec0 = __riscv_vzext_vf2_u16m1(w_gate_u8_0, VL);
            const vuint8mf2_t w_gate_u8_1 = __riscv_vle8_v_u8mf2(w_gate + f_offset + k4_offset + 16, VL);
            const vuint16m1_t w_gate_vec1 = __riscv_vzext_vf2_u16m1(w_gate_u8_1, VL);
            const vuint8mf2_t w_gate_u8_2 = __riscv_vle8_v_u8mf2(w_gate + f_offset + k4_offset + 32, VL);
            const vuint16m1_t w_gate_vec2 = __riscv_vzext_vf2_u16m1(w_gate_u8_2, VL);
            const vuint8mf2_t w_gate_u8_3 = __riscv_vle8_v_u8mf2(w_gate + f_offset + k4_offset + 48, VL);
            const vuint16m1_t w_gate_vec3 = __riscv_vzext_vf2_u16m1(w_gate_u8_3, VL);

            const vuint8mf2_t w_up_u8_0 = __riscv_vle8_v_u8mf2(w_up + f_offset + k4_offset + 0, VL);
            const vuint16m1_t w_up_vec0 = __riscv_vzext_vf2_u16m1(w_up_u8_0, VL);
            const vuint8mf2_t w_up_u8_1 = __riscv_vle8_v_u8mf2(w_up + f_offset + k4_offset + 16, VL);
            const vuint16m1_t w_up_vec1 = __riscv_vzext_vf2_u16m1(w_up_u8_1, VL);
            const vuint8mf2_t w_up_u8_2 = __riscv_vle8_v_u8mf2(w_up + f_offset + k4_offset + 32, VL);
            const vuint16m1_t w_up_vec2 = __riscv_vzext_vf2_u16m1(w_up_u8_2, VL);
            const vuint8mf2_t w_up_u8_3 = __riscv_vle8_v_u8mf2(w_up + f_offset + k4_offset + 48, VL);
            const vuint16m1_t w_up_vec3 = __riscv_vzext_vf2_u16m1(w_up_u8_3, VL);

            // B tokens share the same weight registers.
            const int8_t* xq_b0 = xq_list[0];
            gate_acc0 = __riscv_vwmaccsu_vx_i32m2(gate_acc0, xq_b0[k + 0], w_gate_vec0, VL);
            up_acc0   = __riscv_vwmaccsu_vx_i32m2(up_acc0,   xq_b0[k + 0], w_up_vec0,   VL);
            gate_acc0 = __riscv_vwmaccsu_vx_i32m2(gate_acc0, xq_b0[k + 1], w_gate_vec1, VL);
            up_acc0   = __riscv_vwmaccsu_vx_i32m2(up_acc0,   xq_b0[k + 1], w_up_vec1,   VL);
            gate_acc0 = __riscv_vwmaccsu_vx_i32m2(gate_acc0, xq_b0[k + 2], w_gate_vec2, VL);
            up_acc0   = __riscv_vwmaccsu_vx_i32m2(up_acc0,   xq_b0[k + 2], w_up_vec2,   VL);
            gate_acc0 = __riscv_vwmaccsu_vx_i32m2(gate_acc0, xq_b0[k + 3], w_gate_vec3, VL);
            up_acc0   = __riscv_vwmaccsu_vx_i32m2(up_acc0,   xq_b0[k + 3], w_up_vec3,   VL);

            if (B > 1) {
                const int8_t* xq_b1 = xq_list[1];
                gate_acc1 = __riscv_vwmaccsu_vx_i32m2(gate_acc1, xq_b1[k + 0], w_gate_vec0, VL);
                up_acc1   = __riscv_vwmaccsu_vx_i32m2(up_acc1,   xq_b1[k + 0], w_up_vec0,   VL);
                gate_acc1 = __riscv_vwmaccsu_vx_i32m2(gate_acc1, xq_b1[k + 1], w_gate_vec1, VL);
                up_acc1   = __riscv_vwmaccsu_vx_i32m2(up_acc1,   xq_b1[k + 1], w_up_vec1,   VL);
                gate_acc1 = __riscv_vwmaccsu_vx_i32m2(gate_acc1, xq_b1[k + 2], w_gate_vec2, VL);
                up_acc1   = __riscv_vwmaccsu_vx_i32m2(up_acc1,   xq_b1[k + 2], w_up_vec2,   VL);
                gate_acc1 = __riscv_vwmaccsu_vx_i32m2(gate_acc1, xq_b1[k + 3], w_gate_vec3, VL);
                up_acc1   = __riscv_vwmaccsu_vx_i32m2(up_acc1,   xq_b1[k + 3], w_up_vec3,   VL);
            }
            if (B > 2) {
                const int8_t* xq_b2 = xq_list[2];
                gate_acc2 = __riscv_vwmaccsu_vx_i32m2(gate_acc2, xq_b2[k + 0], w_gate_vec0, VL);
                up_acc2   = __riscv_vwmaccsu_vx_i32m2(up_acc2,   xq_b2[k + 0], w_up_vec0,   VL);
                gate_acc2 = __riscv_vwmaccsu_vx_i32m2(gate_acc2, xq_b2[k + 1], w_gate_vec1, VL);
                up_acc2   = __riscv_vwmaccsu_vx_i32m2(up_acc2,   xq_b2[k + 1], w_up_vec1,   VL);
                gate_acc2 = __riscv_vwmaccsu_vx_i32m2(gate_acc2, xq_b2[k + 2], w_gate_vec2, VL);
                up_acc2   = __riscv_vwmaccsu_vx_i32m2(up_acc2,   xq_b2[k + 2], w_up_vec2,   VL);
                gate_acc2 = __riscv_vwmaccsu_vx_i32m2(gate_acc2, xq_b2[k + 3], w_gate_vec3, VL);
                up_acc2   = __riscv_vwmaccsu_vx_i32m2(up_acc2,   xq_b2[k + 3], w_up_vec3,   VL);
            }
            if (B > 3) {
                const int8_t* xq_b3 = xq_list[3];
                gate_acc3 = __riscv_vwmaccsu_vx_i32m2(gate_acc3, xq_b3[k + 0], w_gate_vec0, VL);
                up_acc3   = __riscv_vwmaccsu_vx_i32m2(up_acc3,   xq_b3[k + 0], w_up_vec0,   VL);
                gate_acc3 = __riscv_vwmaccsu_vx_i32m2(gate_acc3, xq_b3[k + 1], w_gate_vec1, VL);
                up_acc3   = __riscv_vwmaccsu_vx_i32m2(up_acc3,   xq_b3[k + 1], w_up_vec1,   VL);
                gate_acc3 = __riscv_vwmaccsu_vx_i32m2(gate_acc3, xq_b3[k + 2], w_gate_vec2, VL);
                up_acc3   = __riscv_vwmaccsu_vx_i32m2(up_acc3,   xq_b3[k + 2], w_up_vec2,   VL);
                gate_acc3 = __riscv_vwmaccsu_vx_i32m2(gate_acc3, xq_b3[k + 3], w_gate_vec3, VL);
                up_acc3   = __riscv_vwmaccsu_vx_i32m2(up_acc3,   xq_b3[k + 3], w_up_vec3,   VL);
            }
        }

        // Finalize: SwiGLU + store h, accumulate amax per token.
        // Token 0
        {
            const float s_x_mul_gate = s_x_list[0] * s_x_mul_gate_factor;
            const float s_x_mul_up   = s_x_list[0] * s_x_mul_up_factor;
            vfloat32m2_t vg = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(gate_acc0, VL), s_x_mul_gate, VL);
            vfloat32m2_t vu = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(up_acc0,   VL), s_x_mul_up,   VL);
            vfloat32m2_t neg_vg = __riscv_vfneg_v_f32m2(vg, VL);
            vfloat32m2_t exp_neg_vg = exp512_approx_ps(neg_vg);
            vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_vg, 1.0f, VL);
            vfloat32m2_t silu = __riscv_vfdiv_vv_f32m2(vg, denom, VL);
            vfloat32m2_t h_vec = __riscv_vfmul_vv_f32m2(silu, vu, VL);
            h_amax_buf[0] = fmaxf(h_amax_buf[0],
                                  __riscv_vfmv_f_s_f32m1_f32(
                                      __riscv_vfredmax_vs_f32m2_f32m1(
                                          __riscv_vfabs_v_f32m2(h_vec, VL),
                                          __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL)));
            __riscv_vse32_v_f32m2(&h_buf[0][f], h_vec, VL);
        }
        if (B > 1) {
            const float s_x_mul_gate = s_x_list[1] * s_x_mul_gate_factor;
            const float s_x_mul_up   = s_x_list[1] * s_x_mul_up_factor;
            vfloat32m2_t vg = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(gate_acc1, VL), s_x_mul_gate, VL);
            vfloat32m2_t vu = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(up_acc1,   VL), s_x_mul_up,   VL);
            vfloat32m2_t neg_vg = __riscv_vfneg_v_f32m2(vg, VL);
            vfloat32m2_t exp_neg_vg = exp512_approx_ps(neg_vg);
            vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_vg, 1.0f, VL);
            vfloat32m2_t silu = __riscv_vfdiv_vv_f32m2(vg, denom, VL);
            vfloat32m2_t h_vec = __riscv_vfmul_vv_f32m2(silu, vu, VL);
            h_amax_buf[1] = fmaxf(h_amax_buf[1],
                                  __riscv_vfmv_f_s_f32m1_f32(
                                      __riscv_vfredmax_vs_f32m2_f32m1(
                                          __riscv_vfabs_v_f32m2(h_vec, VL),
                                          __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL)));
            __riscv_vse32_v_f32m2(&h_buf[1][f], h_vec, VL);
        }
        if (B > 2) {
            const float s_x_mul_gate = s_x_list[2] * s_x_mul_gate_factor;
            const float s_x_mul_up   = s_x_list[2] * s_x_mul_up_factor;
            vfloat32m2_t vg = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(gate_acc2, VL), s_x_mul_gate, VL);
            vfloat32m2_t vu = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(up_acc2,   VL), s_x_mul_up,   VL);
            vfloat32m2_t neg_vg = __riscv_vfneg_v_f32m2(vg, VL);
            vfloat32m2_t exp_neg_vg = exp512_approx_ps(neg_vg);
            vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_vg, 1.0f, VL);
            vfloat32m2_t silu = __riscv_vfdiv_vv_f32m2(vg, denom, VL);
            vfloat32m2_t h_vec = __riscv_vfmul_vv_f32m2(silu, vu, VL);
            h_amax_buf[2] = fmaxf(h_amax_buf[2],
                                  __riscv_vfmv_f_s_f32m1_f32(
                                      __riscv_vfredmax_vs_f32m2_f32m1(
                                          __riscv_vfabs_v_f32m2(h_vec, VL),
                                          __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL)));
            __riscv_vse32_v_f32m2(&h_buf[2][f], h_vec, VL);
        }
        if (B > 3) {
            const float s_x_mul_gate = s_x_list[3] * s_x_mul_gate_factor;
            const float s_x_mul_up   = s_x_list[3] * s_x_mul_up_factor;
            vfloat32m2_t vg = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(gate_acc3, VL), s_x_mul_gate, VL);
            vfloat32m2_t vu = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(up_acc3,   VL), s_x_mul_up,   VL);
            vfloat32m2_t neg_vg = __riscv_vfneg_v_f32m2(vg, VL);
            vfloat32m2_t exp_neg_vg = exp512_approx_ps(neg_vg);
            vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_vg, 1.0f, VL);
            vfloat32m2_t silu = __riscv_vfdiv_vv_f32m2(vg, denom, VL);
            vfloat32m2_t h_vec = __riscv_vfmul_vv_f32m2(silu, vu, VL);
            h_amax_buf[3] = fmaxf(h_amax_buf[3],
                                  __riscv_vfmv_f_s_f32m1_f32(
                                      __riscv_vfredmax_vs_f32m2_f32m1(
                                          __riscv_vfabs_v_f32m2(h_vec, VL),
                                          __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL)));
            __riscv_vse32_v_f32m2(&h_buf[3][f], h_vec, VL);
        }
    }

    for (int b = 0; b < B; ++b) {
        if (h_amax_buf[b] > h_amax_all) h_amax_all = h_amax_buf[b];
    }

    // ------------------------------------------------------------------
    // Stage B: per-token requantize h -> hq.
    // Each token uses its own scale (independent amax), matching expert_ffn.
    // ------------------------------------------------------------------
    for (int b = 0; b < B; ++b) {
        const float s_h = (h_amax_buf[b] > 0.0f) ? h_amax_buf[b] / 127.0f : 1.0f;
        const float r_s_h = (h_amax_buf[b] > 0.0f) ? 127.0f / h_amax_buf[b] : 1.0f;
        for (int f = 0; f < d_ff; f += 16) {
            vfloat32m2_t h_vec = __riscv_vle32_v_f32m2(&h_buf[b][f], VL);
            vfloat32m2_t h_scaled = __riscv_vfmul_vf_f32m2(h_vec, r_s_h, VL);
            vint32m2_t h_i32 = __riscv_vfcvt_x_f_v_i32m2(h_scaled, VL);
            vint16m1_t h_i16 = __riscv_vnclip_wx_i16m1(h_i32, 0, __RISCV_VXRM_RNU, VL);
            vint8mf2_t v_i8 = __riscv_vnclip_wx_i8mf2(h_i16, 0, __RISCV_VXRM_RNU, VL);
            __riscv_vse8_v_i8mf2(&hq_buf[b][f], v_i8, VL);
        }
    }

    // ------------------------------------------------------------------
    // Stage C: down projection. Reuse w_down tile across B tokens.
    // ------------------------------------------------------------------
    int32_t hq_sum_buf[MAX_BATCH];
    for (int b = 0; b < B; ++b) {
        int32_t s = 0;
        vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
        for (int f = 0; f < d_ff; f += 16) {
            vint8mf2_t hq_v = __riscv_vle8_v_i8mf2(&hq_buf[b][f], VL);
            vint32m2_t hq_i32 = __riscv_vsext_vf4_i32m2(hq_v, VL);
            acc = __riscv_vredsum_vs_i32m2_i32m1(hq_i32, acc, VL);
        }
        s = __riscv_vmv_x_s_i32m1_i32(acc);
        hq_sum_buf[b] = s;
    }

    for (int d = 0; d < d_model; d += 16) {
        const size_t d_offset = static_cast<size_t>(d) * d_ff;

        // Per-token down accumulators for this d-block (live across f-loop).
        vint32m2_t down_acc0 = __riscv_vmv_v_x_i32m2(-128 * hq_sum_buf[0], VL);
        vint32m2_t down_acc1 = (B > 1) ? __riscv_vmv_v_x_i32m2(-128 * hq_sum_buf[1], VL) : down_acc0;
        vint32m2_t down_acc2 = (B > 2) ? __riscv_vmv_v_x_i32m2(-128 * hq_sum_buf[2], VL) : down_acc0;
        vint32m2_t down_acc3 = (B > 3) ? __riscv_vmv_v_x_i32m2(-128 * hq_sum_buf[3], VL) : down_acc0;

        for (int f = 0; f < d_ff; f += 4) {
            const size_t f4_offset = static_cast<size_t>(f / 4) * 64;
            const vuint8mf2_t w_down_u8_0 = __riscv_vle8_v_u8mf2(w_down + d_offset + f4_offset + 0,  VL);
            const vuint16m1_t w_down_vec0 = __riscv_vzext_vf2_u16m1(w_down_u8_0, VL);
            const vuint8mf2_t w_down_u8_1 = __riscv_vle8_v_u8mf2(w_down + d_offset + f4_offset + 16, VL);
            const vuint16m1_t w_down_vec1 = __riscv_vzext_vf2_u16m1(w_down_u8_1, VL);
            const vuint8mf2_t w_down_u8_2 = __riscv_vle8_v_u8mf2(w_down + d_offset + f4_offset + 32, VL);
            const vuint16m1_t w_down_vec2 = __riscv_vzext_vf2_u16m1(w_down_u8_2, VL);
            const vuint8mf2_t w_down_u8_3 = __riscv_vle8_v_u8mf2(w_down + d_offset + f4_offset + 48, VL);
            const vuint16m1_t w_down_vec3 = __riscv_vzext_vf2_u16m1(w_down_u8_3, VL);

            const int8_t* hq_b0 = hq_buf[0];
            down_acc0 = __riscv_vwmaccsu_vx_i32m2(down_acc0, hq_b0[f + 0], w_down_vec0, VL);
            down_acc0 = __riscv_vwmaccsu_vx_i32m2(down_acc0, hq_b0[f + 1], w_down_vec1, VL);
            down_acc0 = __riscv_vwmaccsu_vx_i32m2(down_acc0, hq_b0[f + 2], w_down_vec2, VL);
            down_acc0 = __riscv_vwmaccsu_vx_i32m2(down_acc0, hq_b0[f + 3], w_down_vec3, VL);

            if (B > 1) {
                const int8_t* hq_b1 = hq_buf[1];
                down_acc1 = __riscv_vwmaccsu_vx_i32m2(down_acc1, hq_b1[f + 0], w_down_vec0, VL);
                down_acc1 = __riscv_vwmaccsu_vx_i32m2(down_acc1, hq_b1[f + 1], w_down_vec1, VL);
                down_acc1 = __riscv_vwmaccsu_vx_i32m2(down_acc1, hq_b1[f + 2], w_down_vec2, VL);
                down_acc1 = __riscv_vwmaccsu_vx_i32m2(down_acc1, hq_b1[f + 3], w_down_vec3, VL);
            }
            if (B > 2) {
                const int8_t* hq_b2 = hq_buf[2];
                down_acc2 = __riscv_vwmaccsu_vx_i32m2(down_acc2, hq_b2[f + 0], w_down_vec0, VL);
                down_acc2 = __riscv_vwmaccsu_vx_i32m2(down_acc2, hq_b2[f + 1], w_down_vec1, VL);
                down_acc2 = __riscv_vwmaccsu_vx_i32m2(down_acc2, hq_b2[f + 2], w_down_vec2, VL);
                down_acc2 = __riscv_vwmaccsu_vx_i32m2(down_acc2, hq_b2[f + 3], w_down_vec3, VL);
            }
            if (B > 3) {
                const int8_t* hq_b3 = hq_buf[3];
                down_acc3 = __riscv_vwmaccsu_vx_i32m2(down_acc3, hq_b3[f + 0], w_down_vec0, VL);
                down_acc3 = __riscv_vwmaccsu_vx_i32m2(down_acc3, hq_b3[f + 1], w_down_vec1, VL);
                down_acc3 = __riscv_vwmaccsu_vx_i32m2(down_acc3, hq_b3[f + 2], w_down_vec2, VL);
                down_acc3 = __riscv_vwmaccsu_vx_i32m2(down_acc3, hq_b3[f + 3], w_down_vec3, VL);
            }
        }

        // Convert down accumulators to float output (per-token scale).
        {
            const float s_h_b = (h_amax_buf[0] > 0.0f) ? h_amax_buf[0] / 127.0f : 1.0f;
            const float s_x_mul_down_b = s_h_b * s_x_mul_down_factor;
            vfloat32m2_t acc_f = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(down_acc0, VL), s_x_mul_down_b, VL);
            __riscv_vse32_v_f32m2(&out_list[0][d], acc_f, VL);
        }
        if (B > 1) {
            const float s_h_b = (h_amax_buf[1] > 0.0f) ? h_amax_buf[1] / 127.0f : 1.0f;
            const float s_x_mul_down_b = s_h_b * s_x_mul_down_factor;
            vfloat32m2_t acc_f = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(down_acc1, VL), s_x_mul_down_b, VL);
            __riscv_vse32_v_f32m2(&out_list[1][d], acc_f, VL);
        }
        if (B > 2) {
            const float s_h_b = (h_amax_buf[2] > 0.0f) ? h_amax_buf[2] / 127.0f : 1.0f;
            const float s_x_mul_down_b = s_h_b * s_x_mul_down_factor;
            vfloat32m2_t acc_f = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(down_acc2, VL), s_x_mul_down_b, VL);
            __riscv_vse32_v_f32m2(&out_list[2][d], acc_f, VL);
        }
        if (B > 3) {
            const float s_h_b = (h_amax_buf[3] > 0.0f) ? h_amax_buf[3] / 127.0f : 1.0f;
            const float s_x_mul_down_b = s_h_b * s_x_mul_down_factor;
            vfloat32m2_t acc_f = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(down_acc3, VL), s_x_mul_down_b, VL);
            __riscv_vse32_v_f32m2(&out_list[3][d], acc_f, VL);
        }
    }
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    // set up OpenMP for parallel processing of tokens
    omp_set_dynamic(0);
    omp_set_num_threads(4);

    // RV-002 dispatch guard: only S3/S4 (large N) benefit from batched FFN.
    // S1 (N=1, 5 tasks) and S2 (N=1, 5 tasks) lack cross-token weight reuse,
    // so batched FFN gives them nothing — instead they take the RV-001 flatten
    // path above, which parallelizes the 5 (token, slot) FFN calls across 4
    // threads (measured S1 -36%, S2 -21% vs serial). Each scenario takes its
    // own best route; the guard decides by problem size.
    const int total_tasks = num_tokens * (top_k + 1);
    const bool use_batched = (num_tokens >= 4) && (total_tasks >= 32)
                             && (d_ff >= 64) && (d_model >= 256);

    if (!use_batched) {
        // -----------------------------------------------------------------
        // S1/S2 path (small N): (token, slot) flatten parallel — RV-001.
        // N=1 has only 1 token but top_k+1=5 FFN slots; the original per-token
        // path ran those 5 slots serially on a single thread (3 threads idle).
        // Flattening the (token, slot) pairs into one task space lets all 4
        // threads work on 4 of the 5 slots in parallel. Measured on the real
        // cluster (round-001): S1 -36%, S2 -21% vs the serial per-token path.
        // S3/S4 stay on the batched path below — flattening regresses them
        // (large-N cache density), and batched FFN beats flatten there anyway.
        // -----------------------------------------------------------------
        const int n_slots = top_k + 1;

        // Stage 1: per-token router sigmoid + Top-K + int8 quantization.
        #pragma omp parallel for schedule(static) if(num_tokens >= 4)
        for (int t = 0; t < num_tokens; ++t) {
            const float* xt = x + (size_t)t * d_model;
            int8_t* xq_t = xq_workspace + (size_t)t * d_model;
            int* topk_idx_t = topk_idx_workspace + (size_t)t * top_k;
            float* s_t = topk_score_workspace + (size_t)t * top_k;

            float s[MAX_NUM_EXPERTS];
            for (int e = 0; e < num_experts; e += 16) {
                vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, VL);
                for (int d = 0; d < d_model; ++d) {
                    vfloat32m2_t w_router_vec = __riscv_vle32_v_f32m2(&w_router_transpose[e * d_model + d * 16], VL);
                    acc = __riscv_vfmacc_vf_f32m2(acc, xt[d], w_router_vec, VL);
                }
                vfloat32m2_t neg_acc = __riscv_vfneg_v_f32m2(acc, VL);
                vfloat32m2_t exp_neg_acc = exp512_approx_ps(neg_acc);
                vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_acc, 1.0f, VL);
                vfloat32m2_t s_vec = __riscv_vfdiv_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, VL), denom, VL);
                __riscv_vse32_v_f32m2(&s[e], s_vec, VL);
            }

            bool used[MAX_NUM_EXPERTS] = {};
            float gate_sum = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                int best = -1;
                for (int e = 0; e < num_experts; ++e) {
                    if (used[e]) continue;
                    if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                        best = e;
                    }
                }
                used[best] = true;
                topk_idx_t[k] = best;
                s_t[k] = s[best];
                gate_sum += s[best];
            }
            gate_sum_workspace[t] = gate_sum;

            float x_amax = 0.0f;
            vfloat32m2_t xt_vec_max = __riscv_vfmv_v_f_f32m2(0.0f, VL);
            for (int d = 0; d < d_model; d += 16) {
                vfloat32m2_t xt_vec_now = __riscv_vle32_v_f32m2(&xt[d], VL);
                xt_vec_max = __riscv_vfmax_vv_f32m2(xt_vec_max, __riscv_vfabs_v_f32m2(xt_vec_now, VL), VL);
            }
            x_amax = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredmax_vs_f32m2_f32m1(xt_vec_max, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL));
            float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
            float r_s_x = (x_amax > 0.0f) ? 127.0f / x_amax : 1.0f;
            x_scale_workspace[t] = s_x;
            for (int d = 0; d < d_model; d += 16) {
                vfloat32m2_t xt_vec_now = __riscv_vle32_v_f32m2(&xt[d], VL);
                vfloat32m2_t xt_vec_now_scaled = __riscv_vfmul_vf_f32m2(xt_vec_now, r_s_x, VL);
                vint32m2_t v_i32 = __riscv_vfcvt_x_f_v_i32m2(xt_vec_now_scaled, VL);
                vint16m1_t v_i16 = __riscv_vnclip_wx_i16m1(v_i32, 0, __RISCV_VXRM_RNU, VL);
                vint8mf2_t v_i8 = __riscv_vnclip_wx_i8mf2(v_i16, 0, __RISCV_VXRM_RNU, VL);
                __riscv_vse8_v_i8mf2(&xq_t[d], v_i8, VL);
            }
        }
        // implicit barrier — topk/xq/score visible to Stage 2

        // Stage 2: (token, slot) flatten FFN.
        // slot 0 = shared expert; slot 1..top_k = routed experts (topk_idx_t[k]).
        #pragma omp parallel for schedule(static) if(total_tasks >= 4)
        for (int task = 0; task < total_tasks; ++task) {
            int t = task / n_slots;
            int slot = task % n_slots;
            const int8_t* xq_t = xq_workspace + (size_t)t * d_model;
            float s_x = x_scale_workspace[t];
            float* o_slot = expert_out_workspace
                            + (size_t)t * (MAX_TOP_K + 1) * MAX_D_MODEL
                            + (size_t)slot * MAX_D_MODEL;

            if (slot == 0) {
                expert_ffn(w_sh_gate_transpose, w_sh_up_transpose, w_sh_down_transpose,
                           w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                           xq_t, s_x, o_slot, d_model, d_ff);
            } else {
                int e = topk_idx_workspace[t * top_k + (slot - 1)];
                expert_ffn(w_gate_transpose + (size_t)e * d_ff * d_model,
                           w_up_transpose   + (size_t)e * d_ff * d_model,
                           w_down_transpose + (size_t)e * d_model * d_ff,
                           w.s_gate[e], w.s_up[e], w.s_down[e],
                           xq_t, s_x, o_slot, d_model, d_ff);
            }
        }
        // implicit barrier — all expert outputs visible to Stage 3

        // Stage 3: residual combine. y = x + o_shared + Σ gate_k * o_k.
        #pragma omp parallel for schedule(static) if(num_tokens >= 4)
        for (int t = 0; t < num_tokens; ++t) {
            const float* xt = x + (size_t)t * d_model;
            float* yt = y + (size_t)t * d_model;
            const float* o_shared = expert_out_workspace
                                    + (size_t)t * (MAX_TOP_K + 1) * MAX_D_MODEL;
            const float* s_t = topk_score_workspace + (size_t)t * top_k;
            const float gate_sum = gate_sum_workspace[t];

            for (int d = 0; d < d_model; d += 16) {
                vfloat32m2_t o_vec = __riscv_vle32_v_f32m2(&o_shared[d], VL);
                vfloat32m2_t xt_vec = __riscv_vle32_v_f32m2(&xt[d], VL);
                __riscv_vse32_v_f32m2(&yt[d], __riscv_vfadd_vv_f32m2(xt_vec, o_vec, VL), VL);
            }
            for (int k = 0; k < top_k; ++k) {
                const float gate = s_t[k] / gate_sum;
                const float* o_k = expert_out_workspace
                                   + (size_t)t * (MAX_TOP_K + 1) * MAX_D_MODEL
                                   + (size_t)(k + 1) * MAX_D_MODEL;
                for (int d = 0; d < d_model; d += 16) {
                    vfloat32m2_t o_vec = __riscv_vle32_v_f32m2(&o_k[d], VL);
                    vfloat32m2_t yt_vec = __riscv_vle32_v_f32m2(&yt[d], VL);
                    vfloat32m2_t yt_new = __riscv_vfadd_vv_f32m2(
                        yt_vec, __riscv_vfmul_vf_f32m2(o_vec, gate, VL), VL);
                    __riscv_vse32_v_f32m2(&yt[d], yt_new, VL);
                }
            }
        }
        return;
    }

    // -----------------------------------------------------------------------
    // Batched path for S3/S4 (large N).
    //
    // Stage 1 (per-token, parallel): router + Top-K + int8 quantization.
    // Stage 2 (per-expert, parallel): group tokens by selected expert and call
    //         expert_ffn_batch once per (expert, slot) group. Weights are loaded
    //         once per batch and reused across B tokens.
    // Stage 3 (per-token, parallel): combine residual + shared FFN output +
    //         Σ gate_k * routed FFN output.
    // -----------------------------------------------------------------------

    // Stage 1: router + top-K + quantization.
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* xt = x + (size_t)t * d_model;
        int8_t* xq_t = xq_workspace + (size_t)t * d_model;
        int* topk_idx_t = topk_idx_workspace + (size_t)t * top_k;
        float* s_t = topk_score_workspace + (size_t)t * top_k;

        float s[MAX_NUM_EXPERTS];
        for (int e = 0; e < num_experts; e += 16) {
            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, VL);
            for (int d = 0; d < d_model; ++d) {
                vfloat32m2_t w_router_vec = __riscv_vle32_v_f32m2(&w_router_transpose[e * d_model + d * 16], VL);
                acc = __riscv_vfmacc_vf_f32m2(acc, xt[d], w_router_vec, VL);
            }
            vfloat32m2_t neg_acc = __riscv_vfneg_v_f32m2(acc, VL);
            vfloat32m2_t exp_neg_acc = exp512_approx_ps(neg_acc);
            vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_acc, 1.0f, VL);
            vfloat32m2_t s_vec = __riscv_vfdiv_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, VL), denom, VL);
            __riscv_vse32_v_f32m2(&s[e], s_vec, VL);
        }

        bool used[MAX_NUM_EXPERTS] = {};
        float gate_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            for (int e = 0; e < num_experts; ++e) {
                if (used[e]) continue;
                if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx_t[k] = best;
            s_t[k] = s[best];
            gate_sum += s[best];
        }
        gate_sum_workspace[t] = gate_sum;

        float x_amax = 0.0f;
        vfloat32m2_t xt_vec_max = __riscv_vfmv_v_f_f32m2(0.0f, VL);
        for (int d = 0; d < d_model; d += 16) {
            vfloat32m2_t xt_vec_now = __riscv_vle32_v_f32m2(&xt[d], VL);
            xt_vec_max = __riscv_vfmax_vv_f32m2(xt_vec_max, __riscv_vfabs_v_f32m2(xt_vec_now, VL), VL);
        }
        x_amax = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m2_f32m1(xt_vec_max, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL));
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
        float r_s_x = (x_amax > 0.0f) ? 127.0f / x_amax : 1.0f;
        x_scale_workspace[t] = s_x;
        for (int d = 0; d < d_model; d += 16) {
            vfloat32m2_t xt_vec_now = __riscv_vle32_v_f32m2(&xt[d], VL);
            vfloat32m2_t xt_vec_now_scaled = __riscv_vfmul_vf_f32m2(xt_vec_now, r_s_x, VL);
            vint32m2_t v_i32 = __riscv_vfcvt_x_f_v_i32m2(xt_vec_now_scaled, VL);
            vint16m1_t v_i16 = __riscv_vnclip_wx_i16m1(v_i32, 0, __RISCV_VXRM_RNU, VL);
            vint8mf2_t v_i8 = __riscv_vnclip_wx_i8mf2(v_i16, 0, __RISCV_VXRM_RNU, VL);
            __riscv_vse8_v_i8mf2(&xq_t[d], v_i8, VL);
        }
    }

    // -----------------------------------------------------------------------
    // Stage 2a: Shared expert FFN — batched across tokens. All tokens share
    // the same weight matrix, so this is a perfect batch candidate (B up to
    // MAX_BATCH=4 per chunk). For S4, shared is a large fraction of FFN time.
    // -----------------------------------------------------------------------
    #pragma omp parallel for schedule(dynamic, 4)
    for (int chunk_start = 0; chunk_start < num_tokens; chunk_start += MAX_BATCH) {
        const int B = (num_tokens - chunk_start < MAX_BATCH) ? (num_tokens - chunk_start) : MAX_BATCH;
        const int8_t* xq_ptrs[MAX_BATCH];
        float s_x_list[MAX_BATCH];
        float* out_ptrs[MAX_BATCH];
        for (int b = 0; b < B; ++b) {
            int t = chunk_start + b;
            xq_ptrs[b] = xq_workspace + (size_t)t * d_model;
            s_x_list[b] = x_scale_workspace[t];
            out_ptrs[b] = expert_out_workspace
                          + (size_t)t * (MAX_TOP_K + 1) * MAX_D_MODEL;
        }
        expert_ffn_batch(w_sh_gate_transpose, w_sh_up_transpose, w_sh_down_transpose,
                          w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                          xq_ptrs, s_x_list, out_ptrs, B, d_model, d_ff);
    }

    // -----------------------------------------------------------------------
    // Stage 2b: Routed expert FFN — batched across tokens that picked the
    // same expert. Count-sort grouping: for each expert e, walk all tokens
    // and collect those that selected e in any slot. Then process each
    // expert's token list in chunks of up to MAX_BATCH=4.
    //
    // Per-expert work is independent -> parallelize across experts.
    // -----------------------------------------------------------------------
    const size_t gate_up_size = (size_t)d_ff * d_model;
    const size_t down_size    = (size_t)d_model * d_ff;

    // Build flat task list of (token, slot, expert) for all routed selections.
    // Group by expert: tokens_for_expert[e] is a list of (token, slot) pairs.
    // We use a count + offset scheme (count-sort).
    int expert_counts[MAX_NUM_EXPERTS] = {};
    for (int t = 0; t < num_tokens; ++t) {
        const int* topk_idx_t = topk_idx_workspace + (size_t)t * top_k;
        for (int k = 0; k < top_k; ++k) {
            int e = topk_idx_t[k];
            ++expert_counts[e];
        }
    }
    int expert_offsets_local[MAX_NUM_EXPERTS + 1] = {};
    for (int e = 0; e < num_experts; ++e) {
        expert_offsets_local[e + 1] = expert_offsets_local[e] + expert_counts[e];
    }
    // Use task_token_workspace / task_slot_workspace as the per-task lists.
    int* expert_token_list = task_token_workspace;
    int* expert_slot_list  = task_slot_workspace;
    int expert_write_pos[MAX_NUM_EXPERTS];
    for (int e = 0; e < num_experts; ++e) expert_write_pos[e] = expert_offsets_local[e];
    for (int t = 0; t < num_tokens; ++t) {
        const int* topk_idx_t = topk_idx_workspace + (size_t)t * top_k;
        for (int k = 0; k < top_k; ++k) {
            int e = topk_idx_t[k];
            const int idx = expert_write_pos[e]++;
            expert_token_list[idx] = t;
            expert_slot_list[idx]  = k;
        }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int e = 0; e < num_experts; ++e) {
        if (expert_counts[e] == 0) continue;
        const int base = expert_offsets_local[e];
        const int cnt = expert_counts[e];
        const uint8_t* w_gate_e = w_gate_transpose + (size_t)e * gate_up_size;
        const uint8_t* w_up_e   = w_up_transpose   + (size_t)e * gate_up_size;
        const uint8_t* w_down_e = w_down_transpose + (size_t)e * down_size;
        const float s_gate_e = w.s_gate[e];
        const float s_up_e   = w.s_up[e];
        const float s_down_e = w.s_down[e];

        // Process this expert's token list in chunks of MAX_BATCH=4.
        for (int chunk_start = 0; chunk_start < cnt; chunk_start += MAX_BATCH) {
            const int B = (cnt - chunk_start < MAX_BATCH) ? (cnt - chunk_start) : MAX_BATCH;
            const int8_t* xq_ptrs[MAX_BATCH];
            float s_x_list[MAX_BATCH];
            float* out_ptrs[MAX_BATCH];
            for (int b = 0; b < B; ++b) {
                int t = expert_token_list[base + chunk_start + b];
                int slot = expert_slot_list[base + chunk_start + b];
                xq_ptrs[b] = xq_workspace + (size_t)t * d_model;
                s_x_list[b] = x_scale_workspace[t];
                out_ptrs[b] = expert_out_workspace
                              + (size_t)t * (MAX_TOP_K + 1) * MAX_D_MODEL
                              + (size_t)(slot + 1) * MAX_D_MODEL;
            }
            expert_ffn_batch(w_gate_e, w_up_e, w_down_e,
                              s_gate_e, s_up_e, s_down_e,
                              xq_ptrs, s_x_list, out_ptrs, B,
                              d_model, d_ff);
        }
    }

    // -----------------------------------------------------------------------
    // Stage 3: combine. y = x + o_shared + Σ_k gate_k * o_routed_k
    // -----------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* xt = x + (size_t)t * d_model;
        float* yt = y + (size_t)t * d_model;
        const float* o_shared = expert_out_workspace + (size_t)t * (MAX_TOP_K + 1) * MAX_D_MODEL;
        const float* s_t = topk_score_workspace + (size_t)t * top_k;
        const float gate_sum = gate_sum_workspace[t];

        // yt = xt + o_shared
        for (int d = 0; d < d_model; d += 16) {
            vfloat32m2_t o_vec = __riscv_vle32_v_f32m2(&o_shared[d], VL);
            vfloat32m2_t xt_vec = __riscv_vle32_v_f32m2(&xt[d], VL);
            __riscv_vse32_v_f32m2(&yt[d], __riscv_vfadd_vv_f32m2(xt_vec, o_vec, VL), VL);
        }
        // yt += Σ gate_k * o_k (slot k+1 of token t)
        for (int k = 0; k < top_k; ++k) {
            const float gate = s_t[k] / gate_sum;
            const float* o_k = expert_out_workspace
                               + (size_t)t * (MAX_TOP_K + 1) * MAX_D_MODEL
                               + (size_t)(k + 1) * MAX_D_MODEL;
            for (int d = 0; d < d_model; d += 16) {
                vfloat32m2_t o_vec = __riscv_vle32_v_f32m2(&o_k[d], VL);
                vfloat32m2_t yt_vec = __riscv_vle32_v_f32m2(&yt[d], VL);
                vfloat32m2_t yt_new = __riscv_vfadd_vv_f32m2(
                    yt_vec, __riscv_vfmul_vf_f32m2(o_vec, gate, VL), VL);
                __riscv_vse32_v_f32m2(&yt[d], yt_new, VL);
            }
        }
    }
}
