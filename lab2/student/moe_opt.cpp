// Main task: optimize the MoE forward pass.

#include "moe.h"
#include <algorithm>   // std::max：求最大值、最大误差
#include <chrono>      // std::chrono：计时
#include <cmath>       // std::sqrt / std::exp / std::abs
#include <cstdint>     // 固定宽度整数类型；当前文件保留但基本未用
#include <cstdlib>     // std::atoi：解析命令行参数
#include <iostream>    // std::cout：打印结果
#include <limits>      // std::numeric_limits：拿到 float 的无穷大
#include <random>      // 随机数生成器
#include <vector>      // std::vector：动态数组
#include <immintrin.h> // AVX-512 intrinsic：__m512 / _mm512_* 等

void preprocess(MoEWeights& w) {
    //change w.router to w.router_transpose
    // | --/ | -> | |// |
    // | /-/ | -> | ||| |
    // | /-- | -> | //| |

    float* w_router_transpose = new float[w.num_experts * w.d_model];
    for (int e = 0; e < w.num_experts; e+=16) {
        for (int d = 0; d < w.d_model; d++) {
            for (int i = 0; i < 16; i++) {
                w_router_transpose[e * w.d_model + d * 16 + i] = w.w_router[e * w.d_model + i * w.d_model + d];
            }
        }
    }
    free(w.w_router); // free the original w.router
    w.w_router = w_router_transpose;
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    for (int t = 0; t < num_tokens; t++) {
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
        for (int e = 0; e < num_experts; e+=16) {
            // acc[e] = <w.router[e], xt>
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < d_model; d++) {
                __m512 w_router_vec = _mm512_loadu_ps(&w.w_router[e * d_model + d * 16]);
                __m512 xt_vec = _mm512_set1_ps(xt[d]);
                acc = _mm512_fmadd_ps(w_router_vec, xt_vec, acc);
            }
            // s[e] = 1/1+exp(-acc[e])
            // Taylor expansion at x=0
            // s[e] = 1/2 + acc[e]/4 - acc[e]^3/48 + acc[e]^5/480 - 17*acc[e]^7/80640 + 31*acc[e]^9/1451520
            // sq = sq_acc[e] = acc[e]^2
            // s[e] = 1/2 + acc[e]*(1/4 + sq*(-1/48 + sq*(1/480 + sq*(-17/80640 + sq*31/1451520))))
                    
            // Clamp range to avoid overflow / underflow when constructing 2^n.
            const __m512 max_x = _mm512_set1_ps(88.3762626647949f);
            const __m512 min_x = _mm512_set1_ps(-87.3365447505531f);

            acc = _mm512_max_ps(acc, max_x);
            acc = _mm512_min_ps(acc, min_x);

            // exp(x) = 2^n * exp(r)
            // n = round(x / ln2)
            // r = x - n * ln2
            const __m512 log2e  = _mm512_set1_ps(1.44269504088896341f);
            const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
            const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);

            const __m512 acc_y = _mm512_mul_ps(acc, log2e);

            const __m512i acc_n = _mm512_cvt_roundps_epi32(
                acc_y,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
            );

            const __m512 acc_nf = _mm512_cvtepi32_ps(acc_n);

            __m512 acc_r = _mm512_fnmadd_ps(acc_nf, ln2_hi, acc);
            acc_r = _mm512_fnmadd_ps(acc_nf, ln2_lo, acc_r);

            // sq = sq_acc[e] = acc[e]^2
            // s[e] = 1/2 + acc[e]*(1/4 + sq*(-1/48 + sq*(1/480 + sq*(-17/80640 + sq*31/1451520))))
            const __m512 sq = _mm512_mul_ps(acc_r, acc_r);
            __m512 s_temp = _mm512_set1_ps(31.0f / 1451520.0f);
            s_temp = _mm512_fnmadd_ps(sq, s_temp, _mm512_set1_ps(-17.0f / 80640.0f));
            s_temp = _mm512_fnmadd_ps(sq, s_temp, _mm512_set1_ps(1.0f / 480.0f));
            s_temp = _mm512_fnmadd_ps(sq, s_temp, _mm512_set1_ps(1.0f / 48.0f));
            s_temp = _mm512_fnmadd_ps(sq, s_temp, _mm512_set1_ps(1.0f / 4.0f));
            s_temp = _mm512_fnmadd_ps(acc_r, s_temp, _mm512_set1_ps(0.5f));

            // Construct 2^n using float exponent bits.
            // float: exponent bias = 127, exponent field starts at bit 23.
            const __m512i pow2_bits = _mm512_slli_epi32(
                _mm512_add_epi32(acc_n, _mm512_set1_epi32(127)),
                23
            );

            const __m512 pow2n = _mm512_castsi512_ps(pow2_bits);

            s_temp = _mm512_mul_ps(s_temp, pow2n);
            _mm512_storeu_ps(&s[e], s_temp);
        }


        // 2. Top-K selection by biased score (ties broken by smaller index)
        /*
        * w.bias[e] + s[e] -> topk_idx[k]
        */
        

        
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