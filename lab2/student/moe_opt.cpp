// Main task: optimize the MoE forward pass.

#include "moe.h"
#include <cstdint>     // int8_t
#include <cstring>     // memset
#include <immintrin.h> // AVX-512 / AMX intrinsics

// approximate exp via 2^n * exp(r) with a degree-6 minimax polynomial in r.
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

// ===========================================================================
// Intel AMX (AMX-INT8 + AMX-TILE) support
//
// When num_tokens > 16 the per-token AVX path leaves the shared expert (which
// runs on EVERY token) under-utilizing the hardware: each token's gate/up/down
// int8 dot-product is computed independently. AMX instead batches a group of
// up to 16 tokens into one matrix and runs three large int8 GEMMs:
//
//   gate : C[16, d_ff] = xq[16, d_model] · W_gate[d_ff, d_model]^T   (int8->int32)
//   up   : C[16, d_ff] = xq[16, d_model] · W_up  [d_ff, d_model]^T   (int8->int32)
//   down : C[16,d_model] = hq[16, d_ff] · W_down[d_model, d_ff]^T    (int8->int32)
//
// `_tile_dpbssd(dst, src0, src1)` computes C[i][j] += sum_k src0[i][k]*src1[j][k],
// i.e. BOTH operands are [row][k] with k contiguous. init_data already stores
// sh_gate/sh_up as [d_ff][d_model] (d_model contiguous) and sh_down as
// [d_model][d_ff] (d_ff contiguous) — exactly the N×K layout AMX wants, so no
// transpose is needed for the AMX path. We snapshot the ORIGINAL shared weights
// in preprocess() before the AVX 16-block transpose frees them.
//
// The fp32 stages (dequant, SwiGLU, requant) are identical element-wise to the
// scalar/AVX reference, and the int8 GEMMs are bit-exact, so the AMX output is
// bit-identical to the reference (check_result passes by construction).
//
// Routed experts (top-k, a different expert set per token) stay on the existing
// per-token AVX path: their per-token expert choice defeats batching and the
// shared expert is the single largest always-on matmul anyway.
// ===========================================================================

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)

// Tile register assignment (palette 0, uniform rows=16, colsb=64):
//   gate/up GEMM: T_A=xq, T_BG=W_gate, T_CG=acc_gate, T_BU=W_up, T_CU=acc_up
//   down  GEMM : T_A=hq,  T_BD=W_down, T_CD=acc_down
#define T_A   0
#define T_BG  1
#define T_CG  2
#define T_BU  3
#define T_CU  4
#define T_BD  1
#define T_CD  2

// 64-byte tile configuration blob (LDTILECFG). All 8 tiles: 16 rows, 64 bytes/row
// => operand tiles hold 16×64 int8, result tiles hold 16×16 int32. Unused tiles
// zeroed so the config is valid (palette 0 requires colsb a multiple of 64).
struct amx_tile_config {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved_0[14];
    uint16_t rows[8];
    uint16_t colsb[8];
    uint8_t  reserved_1[16];
};
static_assert(sizeof(amx_tile_config) == 64, "tile config must be 64 bytes");

static void amx_configure() {
    amx_tile_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 0;
    cfg.start_row = 0;
    for (int i = 0; i < 8; i++) {
        cfg.rows[i] = 16;
        cfg.colsb[i] = 64;
    }
    _tile_loadconfig(&cfg);
}

// Batched shared-expert FFN for one group of up to 16 tokens (zero-padded to 16).
//   xq  : [16][d_model] int8  (token rows beyond M are zero)
//   s_x : [16] float          (per-token input scale)
// out: o   : [16][d_model] float
static void expert_ffn_amx(const int8_t* sh_gate, const int8_t* sh_up,
                           const int8_t* sh_down, float sh_s_gate,
                           float sh_s_up, float sh_s_down,
                           const int8_t* xq, const float* s_x, float* o,
                           int d_model, int d_ff) {
    static int32_t acc_g[16 * MAX_D_FF];
    static int32_t acc_u[16 * MAX_D_FF];
    static float    h_buf[16 * MAX_D_FF];
    static int8_t   hq_buf[16 * MAX_D_FF];
    static int32_t  acc_d[16 * MAX_D_MODEL];

    const int A_stride = d_model;   // xq row pitch (bytes)
    const int B_stride = d_model;   // sh_gate/up row pitch (bytes)
    const int Cg_stride = d_ff * 4; // acc_g row pitch (int32)
    // ---- gate + up projections (fused: reuse xq tile across both) ----
    for (int n = 0; n < d_ff; n += 16) {
        _tile_zero(T_CG);
        _tile_zero(T_CU);
        for (int k = 0; k < d_model; k += 64) {
            _tile_loadd(T_A, &xq[k], A_stride);                 // 16 rows x 64
            _tile_loadd(T_BG, &sh_gate[(size_t)n * d_model + k], B_stride);
            _tile_dpbssd(T_CG, T_A, T_BG);
            _tile_loadd(T_BU, &sh_up[(size_t)n * d_model + k], B_stride);
            _tile_dpbssd(T_CU, T_A, T_BU);
        }
        _tile_stored(T_CG, &acc_g[n], Cg_stride);
        _tile_stored(T_CU, &acc_u[n], Cg_stride);
    }

    // ---- dequant + SwiGLU + per-token requant ----
    // h[m][f] = silu(vg) * vu,  vg = acc_g * s_x[m]*s_gate, vu = acc_u * s_x[m]*s_up
    float s_h[16];
    for (int m = 0; m < 16; m++) {
        const float sg = s_x[m] * sh_s_gate;
        const float su = s_x[m] * sh_s_up;
        __m512 h_amax_vec = _mm512_setzero_ps();
        for (int f = 0; f < d_ff; f += 16) {
            __m512 ag = _mm512_cvtepi32_ps(_mm512_loadu_epi32(&acc_g[m * d_ff + f]));
            __m512 au = _mm512_cvtepi32_ps(_mm512_loadu_epi32(&acc_u[m * d_ff + f]));
            __m512 vg = _mm512_mul_ps(ag, _mm512_set1_ps(sg));
            __m512 vu = _mm512_mul_ps(au, _mm512_set1_ps(su));
            __m512 exp_neg = exp512_approx_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg));
            __m512 silu = _mm512_div_ps(vg, _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg));
            __m512 h = _mm512_mul_ps(silu, vu);
            h_amax_vec = _mm512_max_ps(h_amax_vec, _mm512_abs_ps(h));
            _mm512_storeu_ps(&h_buf[m * d_ff + f], h);
        }
        float h_amax = _mm512_reduce_max_ps(h_amax_vec);
        float sh = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
        float rsh = (h_amax > 0.0f) ? 127.0f / h_amax : 1.0f;
        s_h[m] = sh;
        for (int f = 0; f < d_ff; f += 16) {
            __m512 h = _mm512_loadu_ps(&h_buf[m * d_ff + f]);
            __m512 hs = _mm512_mul_ps(h, _mm512_set1_ps(rsh));
            __m512i hi = _mm512_cvt_roundps_epi32(hs, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i v8 = _mm512_cvtsepi32_epi8(hi);
            _mm_storeu_si128((__m128i*)(void*)&hq_buf[m * d_ff + f], v8);
        }
    }

    // ---- down projection: o[16, d_model] = hq[16, d_ff] · sh_down[d_model, d_ff]^T ----
    const int Hd_stride = d_ff;     // hq row pitch (bytes)
    const int Bd_stride = d_ff;     // sh_down row pitch (bytes)
    const int Cd_stride = d_model * 4; // acc_d row pitch (int32)
    for (int n = 0; n < d_model; n += 16) {
        _tile_zero(T_CD);
        for (int k = 0; k < d_ff; k += 64) {
            _tile_loadd(T_A, &hq_buf[k], Hd_stride);            // 16 rows x 64
            _tile_loadd(T_BD, &sh_down[(size_t)n * d_ff + k], Bd_stride);
            _tile_dpbssd(T_CD, T_A, T_BD);
        }
        _tile_stored(T_CD, &acc_d[n], Cd_stride);
    }

    // ---- dequant down ----
    for (int m = 0; m < 16; m++) {
        float sd = s_h[m] * sh_s_down;
        for (int d = 0; d < d_model; d += 16) {
            __m512 ad = _mm512_cvtepi32_ps(_mm512_loadu_epi32(&acc_d[m * d_model + d]));
            __m512 ov = _mm512_mul_ps(ad, _mm512_set1_ps(sd));
            _mm512_storeu_ps(&o[m * d_model + d], ov);
        }
    }
}

#endif // __AMX_INT8__ && __AMX_TILE__

// ---------------------------------------------------------------------------
// Original-layout shared weights for AMX (snapshot taken in preprocess before
// the AVX 16-block transpose frees the originals). moe.h is framework code and
// cannot be extended, so these live as file-scope statics.
// ---------------------------------------------------------------------------
// Original-layout shared weights for AMX (snapshot taken in preprocess before
// the AVX 16-block transpose frees the originals). moe.h is framework code and
// cannot be extended, so these live as file-scope statics.
// ---------------------------------------------------------------------------
int8_t* g_amx_sh_gate = nullptr;
int8_t* g_amx_sh_up   = nullptr;
int8_t* g_amx_sh_down = nullptr;

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
    delete[] w.w_router; // free the original w.router (allocated with new[])
    w.w_router = w_router_transpose;

    // AMX needs the ORIGINAL [d_ff][d_model] / [d_model][d_ff] layouts (k
    // contiguous). Snapshot the shared weights BEFORE the AVX transpose below
    // frees them.
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    {
        size_t ng = (size_t)w.d_ff * w.d_model;
        size_t nd = (size_t)w.d_model * w.d_ff;
        g_amx_sh_gate = new int8_t[ng]; std::memcpy(g_amx_sh_gate, w.sh_gate, ng);
        g_amx_sh_up   = new int8_t[ng]; std::memcpy(g_amx_sh_up,   w.sh_up,   ng);
        g_amx_sh_down = new int8_t[nd]; std::memcpy(g_amx_sh_down, w.sh_down, nd);
    }
#endif

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
    delete[] w.sh_gate; // free the original w.sh_gate (allocated with new[])
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
    delete[] w.sh_up; // free the original w.sh_up (allocated with new[])
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
    delete[] w.sh_down; // free the original w.sh_down (allocated with new[])
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
    delete[] w.w_gate; // free the original w.w_gate (allocated with new[])
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
    delete[] w.w_up; // free the original w.w_up (allocated with new[])
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
    delete[] w.w_down; // free the original w.w_down (allocated with new[])
    w.w_down = w_down_transpose;
}


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
        __m512i h_i32 = _mm512_cvt_roundps_epi32(h_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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

// Per-token light work: affinity scores, top-K selection, gate sum, and
// per-token int8 quantization. Shared by the AVX and AMX dispatch paths.
struct TokenState {
    float s[MAX_NUM_EXPERTS];
    int topk_idx[MAX_TOP_K];
    float gate_sum;
    int8_t xq[MAX_D_MODEL];
    float s_x;
};

static inline void compute_routing(const float* x, const MoEWeights& w, int t,
                                   TokenState& st) {
    const int d_model = w.d_model;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;
    const float* xt = x + (size_t)t * d_model;

    // 1. Affinity scores s[e] = sigmoid(w_router[e] . xt)
    for (int e = 0; e < num_experts; e += 16) {
        __m512 acc = _mm512_setzero_ps();
        for (int d = 0; d < d_model; ++d) {
            __m512 w_router_vec = _mm512_loadu_ps(&w.w_router[e * d_model + d * 16]);
            __m512 xt_vec = _mm512_set1_ps(xt[d]);
            acc = _mm512_fmadd_ps(w_router_vec, xt_vec, acc);
        }
        __m512 neg_acc = _mm512_sub_ps(_mm512_setzero_ps(), acc);
        __m512 exp_neg_acc = exp512_approx_ps(neg_acc);
        __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_acc);
        __m512 s_vec = _mm512_div_ps(_mm512_set1_ps(1.0f), denom);
        _mm512_storeu_ps(&st.s[e], s_vec);
    }

    // 2. Top-K selection by biased score (ties broken by smaller index)
    bool used[MAX_NUM_EXPERTS] = {};
    st.gate_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int best = -1;
        for (int e = 0; e < num_experts; ++e) {
            if (used[e]) continue;
            if (best < 0 || st.s[e] + w.bias[e] > st.s[best] + w.bias[best]) {
                best = e;
            }
        }
        used[best] = true;
        st.topk_idx[k] = best;
        st.gate_sum += st.s[best];
    }

    // 3. Quantize the token to int8 (symmetric, per-token scale)
    __m512 xt_vec_max = _mm512_setzero_ps();
    for (int d = 0; d < d_model; d += 16) {
        __m512 xt_vec_now = _mm512_loadu_ps(&xt[d]);
        xt_vec_max = _mm512_max_ps(xt_vec_max, _mm512_abs_ps(xt_vec_now));
    }
    float x_amax = _mm512_reduce_max_ps(xt_vec_max);
    st.s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
    float r_s_x = (x_amax > 0.0f) ? 127.0f / x_amax : 1.0f;
    for (int d = 0; d < d_model; d += 16) {
        __m512 xt_vec_now = _mm512_loadu_ps(&xt[d]);
        __m512 scaled = _mm512_mul_ps(xt_vec_now, _mm512_set1_ps(r_s_x));
        __m512i v_i32 = _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i v_i8 = _mm512_cvtsepi32_epi8(v_i32);
        _mm_storeu_si128((__m128i*)(void*)&st.xq[d], v_i8);
    }
}

// Per-token light work + full FFN combine, on the per-token AVX path.
static inline void run_token_avx(const float* x, const MoEWeights& w, int t,
                                float* y) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int top_k = w.top_k;
    const float* xt = x + (size_t)t * d_model;
    float* yt = y + (size_t)t * d_model;

    TokenState st;
    compute_routing(x, w, t, st);

    // Shared expert + residual
    float o[MAX_D_MODEL];
    expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
               w.sh_s_down, st.xq, st.s_x, o, d_model, d_ff);
    for (int d = 0; d < d_model; d += 16) {
        __m512 o_vec = _mm512_loadu_ps(&o[d]);
        __m512 xt_vec = _mm512_loadu_ps(&xt[d]);
        _mm512_storeu_ps(&yt[d], _mm512_add_ps(xt_vec, o_vec));
    }
    // Routed experts
    for (int k = 0; k < top_k; ++k) {
        int e = st.topk_idx[k];
        float gate = st.s[e] / st.gate_sum;
        expert_ffn(w.w_gate + (size_t)e * d_ff * d_model,
                   w.w_up + (size_t)e * d_ff * d_model,
                   w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
                   w.s_up[e], w.s_down[e], st.xq, st.s_x, o, d_model, d_ff);
        for (int d = 0; d < d_model; d += 16) {
            __m512 o_vec = _mm512_loadu_ps(&o[d]);
            __m512 yt_vec = _mm512_loadu_ps(&yt[d]);
            __m512 gv = _mm512_set1_ps(gate);
            _mm512_storeu_ps(&yt[d], _mm512_add_ps(yt_vec, _mm512_mul_ps(gv, o_vec)));
        }
    }
}

// Per-token AVX path (used directly when num_tokens <= 16, and as the fallback
// when AMX is unavailable).
static void moe_forward_avx(const float* x, const MoEWeights& w, float* y,
                            int num_tokens) {
    for (int t = 0; t < num_tokens; ++t) {
        run_token_avx(x, w, t, y);
    }
}

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
// AMX path: process tokens in groups of 16. Per group:
//   - run the light per-token work (routing + quant) to fill xq_batch / s_x_batch
//     and cache the routing (topk_idx, s, gate_sum) for the combine step
//   - run the shared expert as ONE batched AMX FFN over all 16 tokens
//   - run the routed experts per-token (AVX) and combine everything into yt
static void moe_forward_amx(const float* x, const MoEWeights& w, float* y,
                             int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int top_k = w.top_k;

    amx_configure();

    static int8_t xq_batch[16 * MAX_D_MODEL];
    static float  s_x_batch[16];
    static float  o_shared[16 * MAX_D_MODEL];
    // Cached routing for the group: topk indices, affinity scores, gate sums.
    static int   grp_topk[16][MAX_TOP_K];
    static float grp_s   [16][MAX_NUM_EXPERTS];
    static float grp_gsum[16];

    for (int g = 0; g < num_tokens; g += 16) {
        const int M = (num_tokens - g < 16) ? (num_tokens - g) : 16;

        // Light per-token work; pack xq into the 16-row batch and cache routing.
        // Pad rows (m >= M) are left zeroed so the AMX batch is well-defined.
        std::memset(xq_batch, 0, (size_t)16 * d_model);
        std::memset(s_x_batch, 0, sizeof(s_x_batch));
        for (int m = 0; m < M; ++m) {
            int t = g + m;
            TokenState st;
            compute_routing(x, w, t, st);
            std::memcpy(&xq_batch[m * d_model], st.xq, d_model);
            s_x_batch[m] = st.s_x;
            for (int k = 0; k < top_k; ++k) grp_topk[m][k] = st.topk_idx[k];
            std::memcpy(grp_s[m], st.s, sizeof(float) * w.num_experts);
            grp_gsum[m] = st.gate_sum;
        }

        // Batched shared expert via AMX (all 16 rows; pad rows are zero).
        expert_ffn_amx(g_amx_sh_gate, g_amx_sh_up, g_amx_sh_down,
                       w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                       xq_batch, s_x_batch, o_shared, d_model, d_ff);

        // Per-token: residual + shared (from AMX), then routed experts (AVX).
        for (int m = 0; m < M; ++m) {
            int t = g + m;
            const float* xt = x + (size_t)t * d_model;
            float* yt = y + (size_t)t * d_model;

            for (int d = 0; d < d_model; d += 16) {
                __m512 xt_vec = _mm512_loadu_ps(&xt[d]);
                __m512 os_vec = _mm512_loadu_ps(&o_shared[m * d_model + d]);
                _mm512_storeu_ps(&yt[d], _mm512_add_ps(xt_vec, os_vec));
            }
            float o[MAX_D_MODEL];
            for (int k = 0; k < top_k; ++k) {
                int e = grp_topk[m][k];
                float gate = grp_s[m][e] / grp_gsum[m];
                expert_ffn(w.w_gate + (size_t)e * d_ff * d_model,
                           w.w_up + (size_t)e * d_ff * d_model,
                           w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
                           w.s_up[e], w.s_down[e], &xq_batch[m * d_model],
                           s_x_batch[m], o, d_model, d_ff);
                for (int d = 0; d < d_model; d += 16) {
                    __m512 o_vec = _mm512_loadu_ps(&o[d]);
                    __m512 yt_vec = _mm512_loadu_ps(&yt[d]);
                    __m512 gv = _mm512_set1_ps(gate);
                    _mm512_storeu_ps(&yt[d], _mm512_add_ps(yt_vec, _mm512_mul_ps(gv, o_vec)));
                }
            }
        }
    }

    _tile_release();
}
#endif // __AMX_INT8__ && __AMX_TILE__

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (num_tokens <= 16) {
        moe_forward_avx(x, w, y, num_tokens);
    } else {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
        moe_forward_amx(x, w, y, num_tokens);
#else
        moe_forward_avx(x, w, y, num_tokens);
#endif
    }
}
