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

    // r = x - nf * ln2  (fnmacc: vd = -(vs1*vs2) + vd)
    vfloat32m2_t r = __riscv_vfnmacc_vv_f32m2(x, nf, ln2_hi, VL);
    r = __riscv_vfnmacc_vv_f32m2(r, nf, ln2_lo, VL);

    // exp(r) polynomial approximation.
    // 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
    // fmacc vv: vd = vs1*vs2 + vd  ->  p = coeff + p*r
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(1.0f / 720.0f, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 120.0f, VL), p, r, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 24.0f,  VL), p, r, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 6.0f,   VL), p, r, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f / 2.0f,   VL), p, r, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f,           VL), p, r, VL);
    p = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f,           VL), p, r, VL);

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

            for (int j = 0; j < 4; ++j) { // RVV: one vle8 + scalar broadcast + vwmaccus per K-index

                //read xq (scalar activation for this K-index)
                const int8_t xq_k = xq[k + j];

                // tile base for this 16-row / 4-K tile; K-index j is contiguous
                const size_t tile = f_offset + k4_offset + (size_t)j * 16;

                // read w_gate (16 unsigned weights for this K-index)
                const vuint8mf2_t w_gate_vec = __riscv_vle8_v_u8mf2(w_gate + tile, VL);
                // read w_up
                const vuint8mf2_t w_up_vec   = __riscv_vle8_v_u8mf2(w_up + tile, VL);

                // calculate: acc[i] += xq_k * w[i]  (signed activation * unsigned weight)
                gate_acc = __riscv_vwmaccus_vx_i32m2(gate_acc, xq_k, w_gate_vec, VL);
                up_acc   = __riscv_vwmaccus_vx_i32m2(up_acc,   xq_k, w_up_vec,   VL);

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
        // vnclip with shift 0 saturates, matching _mm512_cvtsepi32_epi8)
        vint16m1_t h_i16 = __riscv_vnclip_wx_i16m1(h_i32, 0, VL);
        vint8mf2_t v_i8 = __riscv_vnclip_wx_i8mf2(h_i16, 0, VL);
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

            for (int j = 0; j < 4; ++j) { // RVV: one vle8 + scalar broadcast + vwmaccus per K-index

                //read hq (scalar activation for this K-index)
                const int8_t hq_k = hq[f + j];

                const size_t tile = d_offset + f4_offset + (size_t)j * 16;

                // read w_down
                const vuint8mf2_t w_down_vec = __riscv_vle8_v_u8mf2(w_down + tile, VL);

                // calculate: acc[i] += hq_k * w[i]  (signed activation * unsigned weight)
                acc = __riscv_vwmaccus_vx_i32m2(acc, hq_k, w_down_vec, VL);
            }
        }
        vfloat32m2_t acc_f = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(acc, VL), s_x_mul_down, VL);
        __riscv_vse32_v_f32m2(&out[d], acc_f, VL);
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

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* xt = x + (size_t)t * d_model;
        float* yt = y + (size_t)t * d_model;

        // 1. Affinity scores
        /*
        * acc[e] = <w.router[e], xt>
        * s[e] = 1/1+exp(-acc[e])
        * change w.router
        * | --/ |    | |// |
        * | /-/ | -> | ||| |
        * | /-- |    | //| |
        */

        float s[MAX_NUM_EXPERTS];
        //float s_add_bias[MAX_NUM_EXPERTS];
        for (int e = 0; e < num_experts; e+=16) {
            // acc[e] = <w.router[e], xt>
            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, VL);
            for (int d = 0; d < d_model; ++d) {
                vfloat32m2_t w_router_vec = __riscv_vle32_v_f32m2(&w_router_transpose[e * d_model + d * 16], VL);
                // acc = w_router * xt[d] + acc   (vfmacc vf: vd = rs1*vs2 + vd)
                acc = __riscv_vfmacc_vf_f32m2(acc, xt[d], w_router_vec, VL);
            }
            // s[e] = 1/1+exp(-acc[e])
            // Compute 1 / (1 + exp(-acc[e]))
            vfloat32m2_t neg_acc = __riscv_vfneg_v_f32m2(acc, VL);
            vfloat32m2_t exp_neg_acc = exp512_approx_ps(neg_acc);
            vfloat32m2_t denom = __riscv_vfadd_vf_f32m2(exp_neg_acc, 1.0f, VL);
            vfloat32m2_t s_vec = __riscv_vfdiv_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, VL), denom, VL);
            __riscv_vse32_v_f32m2(&s[e], s_vec, VL);
            // s_add_bias[e] = w.bias[e] + s[e]
            /*
            vfloat32m2_t bias_vec = __riscv_vle32_v_f32m2(&w.bias[e], VL);
            vfloat32m2_t s_add_bias_vec = __riscv_vfadd_vv_f32m2(bias_vec, s_vec, VL);
            __riscv_vse32_v_f32m2(&s_add_bias[e], s_add_bias_vec, VL);
            */
        }

        // 2. Top-K selection by biased score (ties broken by smaller index)
        /*
        * w.bias[e] + s[e] -> topk_idx[k]
        * num_of_divide = sqrt(num_experts) is most efficient
        */

        int topk_idx[MAX_TOP_K];
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
            topk_idx[k] = best;
            gate_sum += s[best];
        }

        // 3. Gate values: normalize the ORIGINAL affinities of the selected
        //    experts (the bias never enters the gate values)
        /*
        * g[e] = s[e] / sum_{k=0}^{top_k-1} s[topk_idx[k]]
        * may combine with step 2
        */

        // 4. Quantize the token to int8 (symmetric, per-token scale)
        /*
        * Convert the token to int8 using a symmetric quantization scheme
        * with a per-token scale factor.
        * x_amax = max(|xt[t]|)
        * s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f
        * xq[d] = (int8_t)lrintf(xt[d] / s_x)
        *
        * may change:
        * r_s_x = 1.0f / s_x
        * xq[d] = (int8_t)lrintf(xt[d] * r_s_x)
        */

        // x_amax = max(|xt[t]|)
        float x_amax = 0.0f;
        vfloat32m2_t xt_vec_max = __riscv_vfmv_v_f_f32m2(0.0f, VL);
        for (int d = 0; d < d_model; d+=16) {
            vfloat32m2_t xt_vec_now = __riscv_vle32_v_f32m2(&xt[d], VL);
            xt_vec_max = __riscv_vfmax_vv_f32m2(xt_vec_max, __riscv_vfabs_v_f32m2(xt_vec_now, VL), VL);
        }
        // reduce max
        x_amax = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m2_f32m1(xt_vec_max, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), VL));

        // s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;

        // change to r_s_x = 1.0f / s_x
        float r_s_x = (x_amax > 0.0f) ? 127.0f / x_amax : 1.0f;

        // xq[d] = (int8_t)lrintf(xt[d] * r_s_x)
        int8_t xq[MAX_D_MODEL];
        for (int d = 0; d < d_model; d+=16) {
            vfloat32m2_t xt_vec_now = __riscv_vle32_v_f32m2(&xt[d], VL);
            vfloat32m2_t xt_vec_now_scaled = __riscv_vfmul_vf_f32m2(xt_vec_now, r_s_x, VL);
            // round-to-nearest float -> int32
            vint32m2_t v_i32 = __riscv_vfcvt_x_f_v_i32m2(xt_vec_now_scaled, VL);
            // saturating narrow int32 -> int16 -> int8 (two 2x narrowing steps)
            vint16m1_t v_i16 = __riscv_vnclip_wx_i16m1(v_i32, 0, VL);
            vint8mf2_t v_i8 = __riscv_vnclip_wx_i8mf2(v_i16, 0, VL);
            __riscv_vse8_v_i8mf2(&xq[d], v_i8, VL);
        }

        // 5+6. Shared expert (always on), then selected routed experts,
        //      combined on top of the residual connection
        /*
        * o_s = ffn_shared
        * ffn_shared:
        *   - w_down = w.sh_down
        *   - w_up = w.sh_up
        *   - w_gate = w.sh_gate
        * yt = xt + o_s
        * o_e = ffn_routed
        * ffn_routed:
        *   - w_down = w.down[e]
        *   - w_up = w.up[e]
        *   - w_gate = w.gate[e]
        * yt += sum(g_e * o_e) for e in topk_idx
        *
        */

        #if IS_AMX
            float o[MAX_D_MODEL];
            for (int k = 0; k < top_k; ++k) {
                int e = topk_idx[k];
                float gate = s[e] / gate_sum;
                expert_ffn(w_gate_transpose + (size_t)e * d_ff * d_model,
                        w_up_transpose + (size_t)e * d_ff * d_model,
                        w_down_transpose + (size_t)e * d_model * d_ff, w.s_gate[e],
                        w.s_up[e], w.s_down[e], xq, s_x, o, d_model, d_ff);
                for (int d = 0; d < d_model; d+=16) {
                    vfloat32m2_t o_vec = __riscv_vle32_v_f32m2(&o[d], VL);
                    vfloat32m2_t yt_vec = __riscv_vle32_v_f32m2(&yt[d], VL);
                    vfloat32m2_t yt_vec_updated = __riscv_vfadd_vv_f32m2(yt_vec, __riscv_vfmul_vf_f32m2(o_vec, gate, VL), VL);
                    __riscv_vse32_v_f32m2(&yt[d], yt_vec_updated, VL);
                }
            }
        #else
            float o[MAX_D_MODEL];
            expert_ffn(w_sh_gate_transpose, w_sh_up_transpose, w_sh_down_transpose, w.sh_s_gate, w.sh_s_up,
                    w.sh_s_down, xq, s_x, o, d_model, d_ff);
            for (int d = 0; d < d_model; d+=16) {
                vfloat32m2_t o_vec = __riscv_vle32_v_f32m2(&o[d], VL);
                vfloat32m2_t xt_vec = __riscv_vle32_v_f32m2(&xt[d], VL);
                vfloat32m2_t yt_vec = __riscv_vfadd_vv_f32m2(xt_vec, o_vec, VL);
                __riscv_vse32_v_f32m2(&yt[d], yt_vec, VL);
            }
            for (int k = 0; k < top_k; ++k) {
                int e = topk_idx[k];
                float gate = s[e] / gate_sum;
                expert_ffn(w_gate_transpose + (size_t)e * d_ff * d_model,
                        w_up_transpose + (size_t)e * d_ff * d_model,
                        w_down_transpose + (size_t)e * d_model * d_ff, w.s_gate[e],
                        w.s_up[e], w.s_down[e], xq, s_x, o, d_model, d_ff);
                for (int d = 0; d < d_model; d+=16) {
                    vfloat32m2_t o_vec = __riscv_vle32_v_f32m2(&o[d], VL);
                    vfloat32m2_t yt_vec = __riscv_vle32_v_f32m2(&yt[d], VL);
                    vfloat32m2_t yt_vec_updated = __riscv_vfadd_vv_f32m2(yt_vec, __riscv_vfmul_vf_f32m2(o_vec, gate, VL), VL);
                    __riscv_vse32_v_f32m2(&yt[d], yt_vec_updated, VL);
                }
            }
        #endif
    }
}
