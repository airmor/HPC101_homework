// Main task: optimize the MoE forward pass.

#include "moe.h"
#include <cstdint>     // int8_t
#include <cstring>
#include <cmath>
#include <cassert>
#include <omp.h>

// #include <immintrin.h> // AVX-512 intrinsic：__m512 / _mm512_*
// The original AVX-512 note is retained above.  Scheme C uses RVV instead.
#if defined(__riscv_vector)
#include <riscv_vector.h>
#define HAVE_RVV 1
#else
#define HAVE_RVV 0
#endif

float* w_router_transpose;
int8_t* w_sh_gate_transpose;
int8_t* w_sh_up_transpose;
int8_t* w_sh_down_transpose;
int8_t* w_gate_transpose;
int8_t* w_up_transpose;
int8_t* w_down_transpose;

void preprocess(MoEWeights& w) {
    //change w.router to w.router_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    // RVV consumes a runtime number of contiguous experts, so the new
    // router layout is [d_model][num_experts] instead of fixed 16-wide rows.
    w_router_transpose = new float[(size_t)w.num_experts * w.d_model];
    for (int d = 0; d < w.d_model; ++d) {
        for (int e = 0; e < w.num_experts; ++e) {
            w_router_transpose[(size_t)d * w.num_experts + e] =
                w.w_router[(size_t)e * w.d_model + d];
        }
    }
    //delete[] w.w_router; // free the original w.router
    //w.w_router = w_router_transpose;

    // VNNI weight packing: repack int8 [rows][cols] weights into the layout
    // Scheme C keeps the comment but changes the layout to RVV-friendly
    // [input_dimension][output_dimension] signed-int8 storage.
    auto pack_vnni = [](const int8_t* src, int8_t* dst, int rows, int cols) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
            }
        }
    };

    // Shared expert (same shape as one routed expert)
    w_sh_gate_transpose = new int8_t[(size_t)w.d_ff * w.d_model];
    pack_vnni(w.sh_gate, w_sh_gate_transpose, w.d_ff, w.d_model);
    w_sh_up_transpose   = new int8_t[(size_t)w.d_ff * w.d_model];
    pack_vnni(w.sh_up,   w_sh_up_transpose,   w.d_ff, w.d_model);
    w_sh_down_transpose = new int8_t[(size_t)w.d_model * w.d_ff];
    pack_vnni(w.sh_down, w_sh_down_transpose, w.d_model, w.d_ff);

    // Routed experts: [num_experts][...]
    const size_t gate_up_size = (size_t)w.d_ff * w.d_model;
    const size_t down_size    = (size_t)w.d_model * w.d_ff;
    w_gate_transpose = new int8_t[(size_t)w.num_experts * gate_up_size];
    w_up_transpose   = new int8_t[(size_t)w.num_experts * gate_up_size];
    w_down_transpose = new int8_t[(size_t)w.num_experts * down_size];
    for (int e = 0; e < w.num_experts; ++e) {
        pack_vnni(w.w_gate + (size_t)e * gate_up_size,
                  w_gate_transpose + (size_t)e * gate_up_size, w.d_ff, w.d_model);
        pack_vnni(w.w_up   + (size_t)e * gate_up_size,
                  w_up_transpose   + (size_t)e * gate_up_size, w.d_ff, w.d_model);
        pack_vnni(w.w_down + (size_t)e * down_size,
                  w_down_transpose + (size_t)e * down_size, w.d_model, w.d_ff);
    }
}


// approximate exp.
// Taylor expansion at x=0

// scores[i] = exp(scores[i] - max_score) / sum(exp(scores[i] - max_score))
// Legacy AMX/exp implementation notes retained verbatim; Scheme C does not
// compile that x86 path.
// Clamp range to avoid overflow / underflow when constructing 2^n.
// exp(x) = 2^n * exp(r)
// n = round(x / ln2)
// r = x - n * ln2
// exp(r) polynomial approximation.
// 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
// Construct 2^n using float exponent bits.
// float: exponent bias = 127, exponent field starts at bit 23.
// tmm0: gate/down accumulator, 1 x 16 int32
// tmm1: up accumulator, 1 x 16 int32
// tmm2: activation, 1 x 64 int8
// tmm3: gate/down weights, packed 16 x 64 bytes
// tmm4: up weights, packed 16 x 64 bytes
// Scheme C intentionally uses expf below rather than the former x86-only
// polynomial approximation: routing choices and quantization boundaries must
// match the scalar reference as closely as possible.
static inline float sigmoid_exact(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Compute rows of a signed-int8 matrix times a signed-int8 vector.  The
// matrix is stored as [cols][rows], which lets RVV process output channels in
// parallel while retaining the scalar reference's K-dimension accumulation
// order for every lane.
static void matvec_i8_transposed(const int8_t* weights, const int8_t* input,
                                 int32_t* output, int rows, int cols) {
#if HAVE_RVV
    for (int r = 0; r < rows;) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(rows - r));
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
        for (int c = 0; c < cols; ++c) {
            const vint8m1_t weight_i8 =
                __riscv_vle8_v_i8m1(weights + (size_t)c * rows + r, vl);
            const vint32m4_t weight_i32 =
                __riscv_vsext_vf4_i32m4(weight_i8, vl);
            acc = __riscv_vmacc_vx_i32m4(acc, (int32_t)input[c],
                                          weight_i32, vl);
        }
        __riscv_vse32_v_i32m4(output + r, acc, vl);
        r += (int)vl;
    }
#else
    for (int r = 0; r < rows; ++r) {
        int32_t acc = 0;
        for (int c = 0; c < cols; ++c) {
            acc += (int32_t)weights[(size_t)c * rows + r] * (int32_t)input[c];
        }
        output[r] = acc;
    }
#endif
}

static void router_scores(const float* xt, const MoEWeights& w, float* s) {
#if HAVE_RVV
    for (int e = 0; e < w.num_experts;) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(w.num_experts - e));
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (int d = 0; d < w.d_model; ++d) {
            const vfloat32m4_t router = __riscv_vle32_v_f32m4(
                w_router_transpose + (size_t)d * w.num_experts + e, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, xt[d], router, vl);
        }
        __riscv_vse32_v_f32m4(s + e, acc, vl);
        e += (int)vl;
    }
#else
    for (int e = 0; e < w.num_experts; ++e) {
        float acc = 0.0f;
        for (int d = 0; d < w.d_model; ++d) {
            acc += w_router_transpose[(size_t)d * w.num_experts + e] * xt[d];
        }
        s[e] = acc;
    }
#endif
    for (int e = 0; e < w.num_experts; ++e) {
        s[e] = sigmoid_exact(s[e]);
    }
}

static void add_residual(const float* xt, const float* o, float* yt,
                         int d_model) {
#if HAVE_RVV
    for (int d = 0; d < d_model;) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(d_model - d));
        const vfloat32m4_t x_vec = __riscv_vle32_v_f32m4(xt + d, vl);
        const vfloat32m4_t o_vec = __riscv_vle32_v_f32m4(o + d, vl);
        __riscv_vse32_v_f32m4(yt + d, __riscv_vfadd_vv_f32m4(x_vec, o_vec, vl), vl);
        d += (int)vl;
    }
#else
    for (int d = 0; d < d_model; ++d) {
        yt[d] = xt[d] + o[d];
    }
#endif
}

static void add_scaled(float* yt, const float* o, float scale, int d_model) {
#if HAVE_RVV
    for (int d = 0; d < d_model;) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(d_model - d));
        vfloat32m4_t y_vec = __riscv_vle32_v_f32m4(yt + d, vl);
        const vfloat32m4_t o_vec = __riscv_vle32_v_f32m4(o + d, vl);
        y_vec = __riscv_vfmacc_vf_f32m4(y_vec, scale, o_vec, vl);
        __riscv_vse32_v_f32m4(yt + d, y_vec, vl);
        d += (int)vl;
    }
#else
    for (int d = 0; d < d_model; ++d) {
        yt[d] += scale * o[d];
    }
#endif
}

static void expert_ffn(const int8_t* w_gate, const int8_t* w_up,
                       const int8_t* w_down, float s_gate, float s_up,
                       float s_down, const int8_t* xq, float s_x, float* out,
                       int d_model, int d_ff) {

    assert(d_ff <= MAX_D_FF);

    // Gate / up projections + SwiGLU activation
    float h[MAX_D_FF];
    float h_amax = 0.0f;
    int32_t gate_acc[MAX_D_FF];
    int32_t up_acc[MAX_D_FF];

    //read xq
    // read w_gate
    // read w_up
    // calculate
    matvec_i8_transposed(w_gate, xq, gate_acc, d_ff, d_model);
    matvec_i8_transposed(w_up, xq, up_acc, d_ff, d_model);
    for (int f = 0; f < d_ff; ++f) {
        const float vg = (float)gate_acc[f] * (s_x * s_gate);
        const float vu = (float)up_acc[f] * (s_x * s_up);
        const float silu = vg / (1.0f + expf(-vg));
        h[f] = silu * vu;
        const float a = fabsf(h[f]);
        if (a > h_amax) h_amax = a;
    }

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    float r_s_h = (h_amax > 0.0f) ? 127.0f / h_amax : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; ++f) {
        hq[f] = (int8_t)lrintf(h[f] * r_s_h);
    }

    // Down projection
    //read hq
    // convert hq4 to __m512i
    // read w_down
    // calculate
    int32_t down_acc[MAX_D_MODEL];
    matvec_i8_transposed(w_down, hq, down_acc, d_model, d_ff);
    for (int d = 0; d < d_model; ++d) {
        out[d] = (float)down_acc[d] * (s_h * s_down);
    }
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    // set up OpenMP for parallel processing of tokens
    // The caller/runtime controls thread count and affinity.  Do not change
    // global OpenMP configuration in the timed forward path.
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
        // acc[e] = <w.router[e], xt>
        // s[e] = 1/1+exp(-acc[e])
        // Compute 1 / (1 + exp(-acc[e]))
        router_scores(xt, w, s);
        // s_add_bias[e] = w.bias[e] + s[e]
        /*
        __m512 bias_vec = _mm512_loadu_ps(&w.bias[e]);
        __m512 s_add_bias_vec = _mm512_add_ps(bias_vec, s_vec);
        _mm512_storeu_ps(&s_add_bias[e], s_add_bias_vec);
        */

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
        for (int d = 0; d < d_model; ++d) {
            const float a = fabsf(xt[d]);
            if (a > x_amax) x_amax = a;
        }

        // reduce max
        // s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;

        // change to r_s_x = 1.0f / s_x
        float r_s_x = (x_amax > 0.0f) ? 127.0f / x_amax : 1.0f;

        // xq[d] = (int8_t)lrintf(xt[d] * r_s_x)
        int8_t xq[MAX_D_MODEL];
        for (int d = 0; d < d_model; ++d) {
            xq[d] = (int8_t)lrintf(xt[d] * r_s_x);
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

        float o[MAX_D_MODEL];
        expert_ffn(w_sh_gate_transpose, w_sh_up_transpose, w_sh_down_transpose,
                   w.sh_s_gate, w.sh_s_up, w.sh_s_down, xq, s_x, o,
                   d_model, d_ff);
        add_residual(xt, o, yt, d_model);

        for (int k = 0; k < top_k; ++k) {
            int e = topk_idx[k];
            float gate = s[e] / gate_sum;
            expert_ffn(w_gate_transpose + (size_t)e * d_ff * d_model,
                       w_up_transpose + (size_t)e * d_ff * d_model,
                       w_down_transpose + (size_t)e * d_model * d_ff,
                       w.s_gate[e], w.s_up[e], w.s_down[e], xq, s_x, o,
                       d_model, d_ff);
            add_scaled(yt, o, gate, d_model);
        }
    }
}
