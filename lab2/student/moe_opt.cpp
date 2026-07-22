// Main task: optimize the MoE forward pass.

#include "moe.h"
#include <cstdint>     // int8_t
#include <cstring>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h> // AVX-512 intrinsic：__m512 / _mm512_*
#include <cassert>
#include <omp.h>
#define IS_AMX 0

float* w_router_transpose;
uint8_t* w_sh_gate_transpose;
uint8_t* w_sh_up_transpose;
uint8_t* w_sh_down_transpose;
uint8_t* w_gate_transpose;
uint8_t* w_up_transpose;
uint8_t* w_down_transpose;

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

    // VNNI weight packing: repack int8 [rows][cols] weights into the layout
    auto pack_vnni = [](const int8_t* src, uint8_t* dst, int rows, int cols) {
        for (int r = 0; r < rows; r += 16) {
            for (int c = 0; c < cols; c += 4) {
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        dst[r * cols + c * 16 + i * 4 + j] =
                            static_cast<uint8_t>(static_cast<int>(src[(r + i) * cols + (c + j)]) + 128);
                    }
                }
            }
        }
    };

    // Shared expert (same shape as one routed expert)
    w_sh_gate_transpose = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_vnni(w.sh_gate, w_sh_gate_transpose, w.d_ff, w.d_model);
    w_sh_up_transpose   = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_vnni(w.sh_up,   w_sh_up_transpose,   w.d_ff, w.d_model);
    w_sh_down_transpose = new uint8_t[(size_t)w.d_model * w.d_ff];
    pack_vnni(w.sh_down, w_sh_down_transpose, w.d_model, w.d_ff);

    // Routed experts: [num_experts][...]
    const size_t gate_up_size = (size_t)w.d_ff * w.d_model;
    const size_t down_size    = (size_t)w.d_model * w.d_ff;
    w_gate_transpose = new uint8_t[(size_t)w.num_experts * gate_up_size];
    w_up_transpose   = new uint8_t[(size_t)w.num_experts * gate_up_size];
    w_down_transpose = new uint8_t[(size_t)w.num_experts * down_size];
    for (int e = 0; e < w.num_experts; ++e) {
        pack_vnni(w.w_gate + e * gate_up_size,
                  w_gate_transpose + (size_t)e * gate_up_size, w.d_ff, w.d_model);
        pack_vnni(w.w_up   + e * gate_up_size,
                  w_up_transpose   + (size_t)e * gate_up_size, w.d_ff, w.d_model);
        pack_vnni(w.w_down + e * down_size,
                  w_down_transpose + (size_t)e * down_size, w.d_model, w.d_ff);
    }
}


// approximate exp.
// Taylor expansion at x=0

// scores[i] = exp(scores[i] - max_score) / sum(exp(scores[i] - max_score))

auto exp512_approx_ps = [](__m512 x) -> __m512
{
    // Clamp range to avoid overflow / underflow when constructing 2^n.
    const __m512 max_x = _mm512_set1_ps(88.3762626647949f);
    const __m512 min_x = _mm512_set1_ps(-87.3365447505531f);

    x = _mm512_min_ps(x, max_x);
    x = _mm512_max_ps(x, min_x);

    // exp(x) = 2^n * exp(r)
    // n = round(x / ln2)
    // r = x - n * ln2
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
    const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);

    const __m512 y = _mm512_mul_ps(x, log2e);

    const __m512i n = _mm512_cvt_roundps_epi32(
        y,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    const __m512 nf = _mm512_cvtepi32_ps(n);

    __m512 r = _mm512_fnmadd_ps(nf, ln2_hi, x);
    r = _mm512_fnmadd_ps(nf, ln2_lo, r);

    // exp(r) polynomial approximation.
    // 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
    __m512 p = _mm512_set1_ps(1.0f / 720.0f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 2.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));

    // Construct 2^n using float exponent bits.
    // float: exponent bias = 127, exponent field starts at bit 23.
    const __m512i pow2_bits = _mm512_slli_epi32(
        _mm512_add_epi32(n, _mm512_set1_epi32(127)),
        23);

    const __m512 pow2n = _mm512_castsi512_ps(pow2_bits);

    return _mm512_mul_ps(p, pow2n);
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

    const __m512 s_x_mul_gate = _mm512_set1_ps(s_x * s_gate);
    const __m512 s_x_mul_up = _mm512_set1_ps(s_x * s_up);
    

    int32_t x_sum = 0;
    __m512i x_sum_vec = _mm512_setzero_si512();
    for (int d = 0; d < d_model; d += 16) {
        __m128i xq_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xq + d));
        __m512i xq_i32 = _mm512_cvtepi8_epi32(xq_128);
        x_sum_vec = _mm512_add_epi32(x_sum_vec, xq_i32);
    }
    x_sum = _mm512_reduce_add_epi32(x_sum_vec);

    const __m512i correction = _mm512_set1_epi32(-128 * x_sum);
    __m512 h_amax_vec = _mm512_setzero_ps();
    for (int f = 0; f < d_ff; f+=16) {
        
        __m512i gate_acc = correction;
        __m512i up_acc = correction;
        const size_t f_offset = static_cast<size_t>(f) * d_model;

        for (int k = 0; k < d_model; k += 4) { // one time x4

            //read xq
            uint32_t x4;
            memcpy(&x4, xq + k, sizeof(x4));

            const __m512i x4_i32 = _mm512_set1_epi32(static_cast<int32_t>(x4));
            const size_t k4_offset = static_cast<size_t>(k / 4) * 64;

            // read w_gate
            const __m512i w_gate_vec_16x4 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(f_offset + k4_offset + w_gate));

            // read w_up
            const __m512i w_up_vec_16x4 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(f_offset + k4_offset + w_up));

            // calculate
            gate_acc = _mm512_dpbusd_epi32(gate_acc, w_gate_vec_16x4, x4_i32);
            up_acc = _mm512_dpbusd_epi32(up_acc, w_up_vec_16x4, x4_i32);


        }

        __m512 vg = _mm512_mul_ps(s_x_mul_gate, _mm512_cvtepi32_ps(gate_acc));
        __m512 vu = _mm512_mul_ps(s_x_mul_up, _mm512_cvtepi32_ps(up_acc));
        __m512 neg_vg = _mm512_sub_ps(_mm512_setzero_ps(), vg);
        __m512 exp_neg_vg = exp512_approx_ps(neg_vg);
        __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_vg);
        __m512 silu = _mm512_div_ps(vg, denom);
        __m512 h_vec = _mm512_mul_ps(silu, vu);
        h_amax_vec = _mm512_max_ps(h_amax_vec, _mm512_abs_ps(h_vec));
        _mm512_storeu_ps(&h[f], h_vec); // store h_vec to h[f] array
        
    }
    h_amax = _mm512_reduce_max_ps(h_amax_vec);

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    float r_s_h = (h_amax > 0.0f) ? 127.0f / h_amax : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; f+=16) {
        __m512 h_vec = _mm512_loadu_ps(&h[f]);
        __m512 h_scaled = _mm512_mul_ps(h_vec, _mm512_set1_ps(r_s_h));
        __m512i h_i32 = _mm512_cvt_roundps_epi32(h_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i v_i8 = _mm512_cvtsepi32_epi8(h_i32);
        _mm_storeu_si128((__m128i*)&hq[f], v_i8);
    }

    int32_t hq_sum = 0;
    {
        __m512i hq_sum_vec = _mm512_setzero_si512();
        for (int f = 0; f < d_ff; f += 16) {
            __m128i hq_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hq + f));
            __m512i hq_i32 = _mm512_cvtepi8_epi32(hq_128);
            hq_sum_vec = _mm512_add_epi32(hq_sum_vec, hq_i32);
        }
        hq_sum = _mm512_reduce_add_epi32(hq_sum_vec);
    }

    const __m512 s_x_mul_down = _mm512_set1_ps(s_h * s_down);

    // Down projection
    for (int d = 0; d < d_model; d += 16) {
        __m512i acc = _mm512_set1_epi32(-128 * hq_sum);
        const size_t d_offset = static_cast<size_t>(d) * d_ff;
        for (int f = 0; f < d_ff; f+=4) { // one time x4

            //read hq
            uint32_t hq4;
            memcpy(&hq4, hq + f, sizeof(hq4));

            // convert hq4 to __m512i
            const __m512i hq4_i32 = _mm512_set1_epi32(static_cast<int32_t>(hq4));
            const size_t f4_offset = static_cast<size_t>(f / 4) * 64;

            // read w_down
            const __m512i w_down_vec_16x4 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(d_offset + f4_offset + w_down));

            // calculate
            acc = _mm512_dpbusd_epi32(acc, w_down_vec_16x4, hq4_i32);
        }
        __m512 acc_f = _mm512_mul_ps(_mm512_cvtepi32_ps(acc), s_x_mul_down);
        _mm512_storeu_ps(&out[d], acc_f);
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
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < d_model; ++d) {
                __m512 w_router_vec = _mm512_loadu_ps(&w_router_transpose[e * d_model + d * 16]);
                __m512 xt_vec = _mm512_set1_ps(xt[d]);
                acc = _mm512_fmadd_ps(w_router_vec, xt_vec, acc);
            }
            // s[e] = 1/1+exp(-acc[e])
            // Compute 1 / (1 + exp(-acc[e]))
            __m512 neg_acc = _mm512_sub_ps(_mm512_setzero_ps(), acc);
            __m512 exp_neg_acc = exp512_approx_ps(neg_acc);
            __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_acc);
            __m512 s_vec = _mm512_div_ps(_mm512_set1_ps(1.0f), denom);
            _mm512_storeu_ps(&s[e], s_vec);
            // s_add_bias[e] = w.bias[e] + s[e]
            /*
            __m512 bias_vec = _mm512_loadu_ps(&w.bias[e]);
            __m512 s_add_bias_vec = _mm512_add_ps(bias_vec, s_vec);
            _mm512_storeu_ps(&s_add_bias[e], s_add_bias_vec);
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
        __m512 xt_vec_max = _mm512_setzero_ps();
        for (int d = 0; d < d_model; d+=16) {
            __m512 xt_vec_now = _mm512_loadu_ps(&xt[d]);
            xt_vec_max = _mm512_max_ps(xt_vec_max, _mm512_abs_ps(xt_vec_now));
        }
        // reduce max
        x_amax = _mm512_reduce_max_ps(xt_vec_max);

        // s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;

        // change to r_s_x = 1.0f / s_x
        float r_s_x = (x_amax > 0.0f) ? 127.0f / x_amax : 1.0f;

        // xq[d] = (int8_t)lrintf(xt[d] * r_s_x)
        int8_t xq[MAX_D_MODEL];
        for (int d = 0; d < d_model; d+=16) {
            __m512 xt_vec_now = _mm512_loadu_ps(&xt[d]);
            __m512 xt_vec_now_scaled = _mm512_mul_ps(xt_vec_now, _mm512_set1_ps(r_s_x));
            __m512i v_i32 = _mm512_cvt_roundps_epi32(xt_vec_now_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i v_i8 = _mm512_cvtsepi32_epi8(v_i32);
            _mm_storeu_si128((__m128i*)(void*)&xq[d], v_i8);
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
                    __m512 o_vec = _mm512_loadu_ps(&o[d]);
                    __m512 yt_vec = _mm512_loadu_ps(&yt[d]);
                    __m512 gate_vec = _mm512_set1_ps(gate);
                    __m512 yt_vec_updated = _mm512_add_ps(yt_vec, _mm512_mul_ps(gate_vec, o_vec));
                    _mm512_storeu_ps(&yt[d], yt_vec_updated);
                }
            }
        #else
            float o[MAX_D_MODEL];
            expert_ffn(w_sh_gate_transpose, w_sh_up_transpose, w_sh_down_transpose, w.sh_s_gate, w.sh_s_up,
                    w.sh_s_down, xq, s_x, o, d_model, d_ff);
            for (int d = 0; d < d_model; d+=16) {
                __m512 o_vec = _mm512_loadu_ps(&o[d]);
                __m512 xt_vec = _mm512_loadu_ps(&xt[d]);
                __m512 yt_vec = _mm512_add_ps(xt_vec, o_vec);
                _mm512_storeu_ps(&yt[d], yt_vec);
            }
            for (int k = 0; k < top_k; ++k) {
                int e = topk_idx[k];
                float gate = s[e] / gate_sum;
                expert_ffn(w_gate_transpose + (size_t)e * d_ff * d_model,
                        w_up_transpose + (size_t)e * d_ff * d_model,
                        w_down_transpose + (size_t)e * d_model * d_ff, w.s_gate[e],
                        w.s_up[e], w.s_down[e], xq, s_x, o, d_model, d_ff);
                for (int d = 0; d < d_model; d+=16) {
                    __m512 o_vec = _mm512_loadu_ps(&o[d]);
                    __m512 yt_vec = _mm512_loadu_ps(&yt[d]);
                    __m512 gate_vec = _mm512_set1_ps(gate);
                    __m512 yt_vec_updated = _mm512_add_ps(yt_vec, _mm512_mul_ps(gate_vec, o_vec));
                    _mm512_storeu_ps(&yt[d], yt_vec_updated);
                }
            }
        #endif
    }
}