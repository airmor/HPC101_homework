// Main task: optimize the MoE forward pass.

#include "moe.h"
#include <cstdint>     // int8_t
#include <immintrin.h> // AVX-512 intrinsic：__m512 / _mm512_*

void preprocess(MoEWeights& w) {
    //change w.router to w.router_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    float* w_router_transpose = new float[w.num_experts * w.d_model];
    for (int e = 0; e < w.num_experts; e+=16) {
        for (int d = 0; d < w.d_model; ++d) {
            for (int i = 0; i < 16; ++i) {
                w_router_transpose[e * w.d_model + d * 16 + i] = w.w_router[e * w.d_model + i * w.d_model + d];
            }
        }
    }
    delete[] w.w_router; // free the original w.router
    w.w_router = w_router_transpose;

    //change w.sh_gate to w.sh_gate_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    int8_t* w_sh_gate_transpose = new int8_t[w.d_ff * w.d_model];
    for (int f = 0; f < w.d_ff; f+=16) {
        for (int d = 0; d < w.d_model; ++d) {
            for (int i = 0; i < 16; ++i) {
                w_sh_gate_transpose[f * w.d_model + d * 16 + i] = w.sh_gate[f * w.d_model + i * w.d_model + d];
            }
        }
    }
    delete[] w.sh_gate; // free the original w.sh_gate
    w.sh_gate = w_sh_gate_transpose;

    //change w.sh_up to w.sh_up_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    int8_t* w_sh_up_transpose = new int8_t[w.d_ff * w.d_model];
    for (int f = 0; f < w.d_ff; f+=16) {
        for (int d = 0; d < w.d_model; ++d) {
            for (int i = 0; i < 16; ++i) {
                w_sh_up_transpose[f * w.d_model + d * 16 + i] = w.sh_up[f * w.d_model + i * w.d_model + d];
            }
        }
    }
    delete[] w.sh_up; // free the original w.sh_up
    w.sh_up = w_sh_up_transpose;

    //change w.sh_down to w.sh_down_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    int8_t* w_sh_down_transpose = new int8_t[w.d_model * w.d_ff];
    for (int d = 0; d < w.d_model; d+=16) {
        for (int f = 0; f < w.d_ff; ++f) {
            for (int i = 0; i < 16; ++i) {
                w_sh_down_transpose[d * w.d_ff + f * 16 + i] = w.sh_down[d * w.d_ff + i * w.d_ff + f];
            }
        }
    }
    delete[] w.sh_down; // free the original w.sh_down
    w.sh_down = w_sh_down_transpose;

    //change w.w_gate to w.w_gate_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    int8_t* w_gate_transpose = new int8_t[(size_t)w.num_experts * w.d_ff * w.d_model];
    for (int e = 0; e < w.num_experts; ++e) {
        for (int f = 0; f < w.d_ff; f+=16) {
            for (int d = 0; d < w.d_model; ++d) {
                for (int i = 0; i < 16; ++i) {
                    w_gate_transpose[(size_t)e * w.d_ff * w.d_model + (size_t)f * w.d_model + d * 16 + i] = w.w_gate[(size_t)e * w.d_ff * w.d_model + (size_t)(f + i) * w.d_model + d];
                }
            }
        }
    }
    delete[] w.w_gate; // free the original w.w_gate
    w.w_gate = w_gate_transpose;

    //change w.w_up to w.w_up_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    int8_t* w_up_transpose = new int8_t[(size_t)w.num_experts * w.d_ff * w.d_model];
    for (int e = 0; e < w.num_experts; ++e) {
        for (int f = 0; f < w.d_ff; f+=16) {
            for (int d = 0; d < w.d_model; ++d) {
                for (int i = 0; i < 16; ++i) {
                    w_up_transpose[(size_t)e * w.d_ff * w.d_model + (size_t)f * w.d_model + d * 16 + i] = w.w_up[(size_t)e * w.d_ff * w.d_model + (size_t)(f + i) * w.d_model + d];
                }
            }
        }
    }
    delete[] w.w_up; // free the original w.w_up
    w.w_up = w_up_transpose;

    //change w.w_down to w.w_down_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    int8_t* w_down_transpose = new int8_t[(size_t)w.num_experts * w.d_model * w.d_ff];
    for (int e = 0; e < w.num_experts; ++e) {
        for (int d = 0; d < w.d_model; d+=16) {
            for (int f = 0; f < w.d_ff; ++f) {
                for (int i = 0; i < 16; ++i) {
                    w_down_transpose[(size_t)e * w.d_model * w.d_ff + (size_t)d * w.d_ff + f * 16 + i] = w.w_down[(size_t)e * w.d_model * w.d_ff + (size_t)(d + i) * w.d_ff + f];
                }
            }
        }
    }
    delete[] w.w_down; // free the original w.w_down
    w.w_down = w_down_transpose;
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

static void expert_ffn(const int8_t* w_gate, const int8_t* w_up,
                       const int8_t* w_down, float s_gate, float s_up,
                       float s_down, const int8_t* xq, float s_x, float* out,
                       int d_model, int d_ff) {
    // Gate / up projections + SwiGLU activation
    float h[MAX_D_FF];
    float h_amax = 0.0f;
    __m512 h_amax_vec = _mm512_setzero_ps();
    for (int f = 0; f < d_ff; f+=16) {
        __m512i acc_g = _mm512_setzero_epi32();
        __m512i acc_u = _mm512_setzero_epi32();
        for (int d = 0; d < d_model; ++d) {
            // Load 16 int8 weights, sign-extend to int32
            __m128i w_gate_16 = _mm_loadu_si128((const __m128i*)&w_gate[f * d_model + d * 16]);
            __m128i w_up_16   = _mm_loadu_si128((const __m128i*)&w_up  [f * d_model + d * 16]);
            __m512i w_gate_i32 = _mm512_cvtepi8_epi32(w_gate_16);
            __m512i w_up_i32   = _mm512_cvtepi8_epi32(w_up_16);
            __m512i xq_i32 = _mm512_set1_epi32((int32_t)xq[d]);
            acc_g = _mm512_add_epi32(acc_g, _mm512_mullo_epi32(w_gate_i32, xq_i32));
            acc_u = _mm512_add_epi32(acc_u, _mm512_mullo_epi32(w_up_i32,   xq_i32));
        }
        __m512 s_x_mul_gate = _mm512_set1_ps(s_x * s_gate);
        __m512 s_x_mul_up = _mm512_set1_ps(s_x * s_up);
        __m512 vg = _mm512_mul_ps(s_x_mul_gate, _mm512_cvtepi32_ps(acc_g));
        __m512 vu = _mm512_mul_ps(s_x_mul_up, _mm512_cvtepi32_ps(acc_u));
        __m512 neg_vg = _mm512_sub_ps(_mm512_setzero_ps(), vg);
        __m512 exp_neg_vg = exp512_approx_ps(neg_vg);
        __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_vg);
        __m512 silu = _mm512_div_ps(vg, denom);
        __m512 h_vec = _mm512_mul_ps(silu, vu);
        h_amax_vec = _mm512_max_ps(h_amax_vec, _mm512_abs_ps(h_vec));
        _mm512_storeu_ps(&h[f], h_vec);
    }
    h_amax = _mm512_reduce_max_ps(h_amax_vec);

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    float r_s_h = (h_amax > 0.0f) ? 127.0f / h_amax : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; f+=16) {
        __m512 h_vec = _mm512_loadu_ps(&h[f]);
        __m512 h_scaled = _mm512_mul_ps(h_vec, _mm512_set1_ps(r_s_h));
        __m512i h_i32 = _mm512_cvt_roundps_epi32(h_scaled, _MM_FROUND_NINT);
        __m128i v_i8 = _mm512_cvtsepi32_epi8(h_i32);
        _mm_storeu_si128((__m128i*)&hq[f], v_i8);
    }

    // Down projection
    for (int d = 0; d < d_model; d += 16) {
        __m512i acc = _mm512_setzero_epi32();
        for (int f = 0; f < d_ff; ++f) {
            __m128i wd = _mm_loadu_si128((const __m128i*)&w_down[d * d_ff + f * 16]);
            __m512i wd_i32 = _mm512_cvtepi8_epi32(wd);
            __m512i hq_i32 = _mm512_set1_epi32((int32_t)hq[f]);
            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(wd_i32, hq_i32));
        }
        __m512 acc_f = _mm512_mul_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(s_h * s_down));
        _mm512_storeu_ps(&out[d], acc_f);
    }
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

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
                __m512 w_router_vec = _mm512_loadu_ps(&w.w_router[e * d_model + d * 16]);
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
            __m512i v_i32 = _mm512_cvt_roundps_epi32(xt_vec_now_scaled, _MM_FROUND_NINT);
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

        #if 0
            float o_s[MAX_D_MODEL];
            float o_e[MAX_TOP_K][MAX_D_MODEL];
            expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
                    w.sh_s_down, xq, s_x, o_s, d_model, d_ff);
            for (int e = 0; e < MAX_TOP_K; ++e) {
                            expert_ffn(w.w_gate + (size_t)e * d_ff * d_model,
                        w.w_up + (size_t)e * d_ff * d_model,
                        w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
                        w.s_up[e], w.s_down[e], xq, s_x, o_e[e], d_model, d_ff);
            }
            for (int d = 0; d < d_model; d+=16) {
                __m512 o_s_vec = _mm512_loadu_ps(&o_s[d]);
                __m512 xt_vec = _mm512_loadu_ps(&xt[d]);
                __m512 yt_vec = _mm512_add_ps(xt_vec, o_s_vec);
                for (int k = 0; k < top_k; ++k) {
                    int e = topk_idx[k];
                    __m512 o_e_vec = _mm512_loadu_ps(&o_e[e][d]);
                    __m512 gate_vec = _mm512_set1_ps(s[e] / gate_sum);
                    __m512 yt_vec = _mm512_add_ps(yt_vec, _mm512_mul_ps(gate_vec, o_e_vec));
                }
                _mm512_storeu_ps(&yt[d], yt_vec);
            }
        #else
            float o[MAX_D_MODEL];
            expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
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
                expert_ffn(w.w_gate + (size_t)e * d_ff * d_model,
                        w.w_up + (size_t)e * d_ff * d_model,
                        w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
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
        // ffn
        /*
        * ffn:
        *   - w_down
        *   - w_up
        *   - w_gate
        * 
        *
        */
    }
}