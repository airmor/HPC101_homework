// Main task: optimize the MoE forward pass.

#include "moe.h"
#include <cstdint>     // int8_t
#include <cstring>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h> // AVX-512 intrinsic：__m512 / _mm512_*
#include <cassert>
#include <omp.h>
#define IS_AMX 1

float* w_router_transpose;
uint8_t* w_sh_gate_transpose;
uint8_t* w_sh_up_transpose;
uint8_t* w_sh_down_transpose;
uint8_t* w_gate_transpose;
uint8_t* w_up_transpose;
uint8_t* w_down_transpose;

#if IS_AMX

struct alignas(64) amx_tilecfg {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

static inline void load_amx_shared_config(int B) {
    assert(B >= 1 && B <= 16);
    amx_tilecfg cfg{};
    cfg.palette_id = 1;
    cfg.rows[0] = cfg.rows[1] = 16;
    cfg.colsb[0] = cfg.colsb[1] = static_cast<uint16_t>(B * 4);
    cfg.rows[2] = cfg.rows[3] = 16;
    cfg.colsb[2] = cfg.colsb[3] = 64;
    cfg.rows[4] = 16;
    cfg.colsb[4] = static_cast<uint16_t>(B * 4);
    _tile_loadconfig(&cfg);
}

// AMX requires each tile to be laid out as [16 output rows][64 K bytes].
// This differs from the AVX-512 VNNI packing used by routed experts below.
static void pack_amx_shared_weights(const int8_t* src, uint8_t* dst,
                                    int rows, int cols) {
    assert(rows % 16 == 0);
    assert(cols % 64 == 0);
    for (int r = 0; r < rows; r += 16) {
        for (int k0 = 0; k0 < cols; k0 += 64) {
            uint8_t* tile = dst + static_cast<size_t>(r) * cols +
                            static_cast<size_t>(k0) * 16;
            for (int i = 0; i < 16; ++i) {
                for (int k = 0; k < 64; ++k) {
                    tile[i * 64 + k] = static_cast<uint8_t>(
                        static_cast<int>(src[static_cast<size_t>(r + i) * cols + k0 + k]) + 128);
                }
            }
        }
    }
}

// Kept local to the AMX branch because the existing lambda with the same
// approximation is declared later for the routed-expert AVX-512 path.
static inline __m512 exp512_approx_ps_amx(__m512 x) {
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.3365447505531f));
    const __m512 y = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f));
    const __m512i n = _mm512_cvt_roundps_epi32(
        y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 nf = _mm512_cvtepi32_ps(n);
    __m512 r = _mm512_fnmadd_ps(nf, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(nf, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.0f / 720.0f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 2.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    const __m512i pow2_bits = _mm512_slli_epi32(
        _mm512_add_epi32(n, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(pow2_bits));
}

// Execute all three shared-expert projections for B contiguous tokens.
// Activations use [K/4][B * 4] rows: every token lane occupies four bytes.
static void shared_expert_amx_batch(const float* x, float* y,
                                    const uint8_t* w_gate,
                                    const uint8_t* w_up,
                                    const uint8_t* w_down,
                                    float s_gate, float s_up, float s_down,
                                    const int8_t* xq_total,
                                    const float* s_x_total,
                                    int block_begin, int B,
                                    int d_model, int d_ff) {
    assert(B >= 1 && B <= 16);
    assert(d_model % 64 == 0 && d_ff % 64 == 0);

    alignas(64) int8_t x_tile[(MAX_D_MODEL / 4) * 16 * 4];
    alignas(64) int8_t h_tile[(MAX_D_FF / 4) * 16 * 4];
    alignas(64) float hidden[16 * MAX_D_FF];
    alignas(64) int8_t hq[16 * MAX_D_FF];
    alignas(64) int32_t gate_acc[16 * 16];
    alignas(64) int32_t up_acc[16 * 16];
    alignas(64) int32_t down_acc[16 * 16];
    int32_t x_sum[16] = {};
    int32_t hq_sum[16] = {};
    float h_amax[16] = {};
    float s_h[16] = {};

    for (int b = 0; b < B; ++b) {
        const int8_t* xq = xq_total + static_cast<size_t>(block_begin + b) * d_model;
        for (int d = 0; d < d_model; ++d) x_sum[b] += xq[d];
    }
    for (int k0 = 0; k0 < d_model; k0 += 64) {
        int8_t* dst_block = x_tile + static_cast<size_t>(k0 / 4) * B * 4;
        for (int k4 = 0; k4 < 16; ++k4) {
            int8_t* dst_row = dst_block + static_cast<size_t>(k4) * B * 4;
            for (int b = 0; b < B; ++b) {
                const int8_t* src = xq_total + static_cast<size_t>(block_begin + b) * d_model + k0 + k4 * 4;
                memcpy(dst_row + b * 4, src, 4);
            }
        }
    }

    // Tile state is architectural per OpenMP worker, so load it here.
    load_amx_shared_config(B);
    for (int f = 0; f < d_ff; f += 16) {
        _tile_zero(0);
        _tile_zero(1);
        for (int k0 = 0; k0 < d_model; k0 += 64) {
            _tile_loadd(2, w_gate + static_cast<size_t>(f) * d_model + static_cast<size_t>(k0) * 16, 64);
            _tile_loadd(3, w_up + static_cast<size_t>(f) * d_model + static_cast<size_t>(k0) * 16, 64);
            _tile_loadd(4, x_tile + static_cast<size_t>(k0 / 4) * B * 4, B * 4);
            _tile_dpbusd(0, 2, 4);
            _tile_dpbusd(1, 3, 4);
        }
        _tile_stored(0, gate_acc, B * 4);
        _tile_stored(1, up_acc, B * 4);

        for (int i = 0; i < 16; ++i) {
            alignas(64) float gate_values[16] = {};
            alignas(64) float up_values[16] = {};
            for (int b = 0; b < B; ++b) {
                const int offset = i * B + b;
                const int32_t correction = -128 * x_sum[b];
                gate_values[b] = static_cast<float>(gate_acc[offset] + correction) *
                                 (s_x_total[block_begin + b] * s_gate);
                up_values[b] = static_cast<float>(up_acc[offset] + correction) *
                               (s_x_total[block_begin + b] * s_up);
            }
            const __m512 vg = _mm512_load_ps(gate_values);
            const __m512 vu = _mm512_load_ps(up_values);
            const __m512 exp_neg_vg = exp512_approx_ps_amx(_mm512_sub_ps(_mm512_setzero_ps(), vg));
            const __m512 silu = _mm512_div_ps(vg, _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_vg));
            alignas(64) float h_values[16];
            _mm512_store_ps(h_values, _mm512_mul_ps(silu, vu));
            for (int b = 0; b < B; ++b) {
                const float value = h_values[b];
                hidden[static_cast<size_t>(b) * d_ff + f + i] = value;
                const float abs_value = value < 0.0f ? -value : value;
                if (abs_value > h_amax[b]) h_amax[b] = abs_value;
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        s_h[b] = h_amax[b] > 0.0f ? h_amax[b] / 127.0f : 1.0f;
        const __m512 inv_s_h = _mm512_set1_ps(h_amax[b] > 0.0f ? 127.0f / h_amax[b] : 1.0f);
        int8_t* hq_token = hq + static_cast<size_t>(b) * d_ff;
        const float* h_token = hidden + static_cast<size_t>(b) * d_ff;
        for (int f = 0; f < d_ff; f += 16) {
            const __m512i h_i32 = _mm512_cvt_roundps_epi32(
                _mm512_mul_ps(_mm512_loadu_ps(h_token + f), inv_s_h),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(hq_token + f), _mm512_cvtsepi32_epi8(h_i32));
        }
        for (int f = 0; f < d_ff; ++f) hq_sum[b] += hq_token[f];
    }

    for (int f0 = 0; f0 < d_ff; f0 += 64) {
        int8_t* dst_block = h_tile + static_cast<size_t>(f0 / 4) * B * 4;
        for (int f4 = 0; f4 < 16; ++f4) {
            int8_t* dst_row = dst_block + static_cast<size_t>(f4) * B * 4;
            for (int b = 0; b < B; ++b) {
                memcpy(dst_row + b * 4,
                       hq + static_cast<size_t>(b) * d_ff + f0 + f4 * 4, 4);
            }
        }
    }

    for (int d = 0; d < d_model; d += 16) {
        _tile_zero(0);
        for (int f0 = 0; f0 < d_ff; f0 += 64) {
            _tile_loadd(2, w_down + static_cast<size_t>(d) * d_ff + static_cast<size_t>(f0) * 16, 64);
            _tile_loadd(4, h_tile + static_cast<size_t>(f0 / 4) * B * 4, B * 4);
            _tile_dpbusd(0, 2, 4);
        }
        _tile_stored(0, down_acc, B * 4);
        for (int i = 0; i < 16; ++i) {
            for (int b = 0; b < B; ++b) {
                const int32_t acc = down_acc[i * B + b] - 128 * hq_sum[b];
                const float shared = static_cast<float>(acc) * (s_h[b] * s_down);
                y[static_cast<size_t>(block_begin + b) * d_model + d + i] =
                    x[static_cast<size_t>(block_begin + b) * d_model + d + i] + shared;
            }
        }
    }
    _tile_release();
}

#endif

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

    #if IS_AMX

    // Shared matrices use AMX tile-native packing. Routed matrices below
    // keep their existing AVX-512 VNNI packing for expert_ffn.
    w_sh_gate_transpose = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_amx_shared_weights(w.sh_gate, w_sh_gate_transpose, w.d_ff, w.d_model);
    w_sh_up_transpose = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_amx_shared_weights(w.sh_up, w_sh_up_transpose, w.d_ff, w.d_model);
    w_sh_down_transpose = new uint8_t[(size_t)w.d_model * w.d_ff];
    pack_amx_shared_weights(w.sh_down, w_sh_down_transpose, w.d_model, w.d_ff);

    #else

    // Shared expert (same shape as one routed expert)
    w_sh_gate_transpose = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_vnni(w.sh_gate, w_sh_gate_transpose, w.d_ff, w.d_model);
    w_sh_up_transpose   = new uint8_t[(size_t)w.d_ff * w.d_model];
    pack_vnni(w.sh_up,   w_sh_up_transpose,   w.d_ff, w.d_model);
    w_sh_down_transpose = new uint8_t[(size_t)w.d_model * w.d_ff];
    pack_vnni(w.sh_down, w_sh_down_transpose, w.d_model, w.d_ff);

    #endif

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

static void expert_ffn_amx(const uint8_t* w_gate, const uint8_t* w_up,
                      const uint8_t* w_down, float s_gate, float s_up,
                      float s_down, const int8_t* xq, float s_x, float* out,
                      int d_model, int d_ff) {


}

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

    int8_t* xq_total = new int8_t[MAX_NUM_TOKENS * MAX_D_MODEL] {0};
    int** topk_idx_total = new int*[MAX_NUM_TOKENS] {0};
    float** s_total = new float*[MAX_NUM_TOKENS] {0};
    float* gate_sum_total = new float[MAX_NUM_TOKENS] {0.0f};
    float* s_x_total = new float[MAX_NUM_TOKENS] {0.0f};

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
        
        float* s = s_total[t] = new float[num_experts] {0.0f};//s_add_bias[MAX_NUM_EXPERTS] = new float[num_experts] {0.0f};//s_add_bias[MAX_NUM_EXPERTS] = new float[num_experts] {0.0f};//s_add_bias[MAX_NUM_EXPERTS] = new float[num_experts] {0.0f};//s_add_bias[MAX_NUM_EXPERTS] = new float[num_exp
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
        
        int* topk_idx = topk_idx_total[t] = new int[top_k] {0};
        bool used[MAX_NUM_EXPERTS] = {};
        float &gate_sum = gate_sum_total[t];
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
        float &s_x = s_x_total[t] = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;

        // change to r_s_x = 1.0f / s_x
        float r_s_x = (x_amax > 0.0f) ? 127.0f / x_amax : 1.0f;

        // xq[d] = (int8_t)lrintf(xt[d] * r_s_x)
        int8_t* xq = xq_total + (size_t)t * d_model;
        for (int d = 0; d < d_model; d+=16) {
            __m512 xt_vec_now = _mm512_loadu_ps(&xt[d]);
            __m512 xt_vec_now_scaled = _mm512_mul_ps(xt_vec_now, _mm512_set1_ps(r_s_x));
            __m512i v_i32 = _mm512_cvt_roundps_epi32(xt_vec_now_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i v_i8 = _mm512_cvtsepi32_epi8(v_i32);
            _mm_storeu_si128((__m128i*)(void*)&xq[d], v_i8);
        }
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
    // shared expert

    #if IS_AMX

    // The common weights make the shared FFN a small GEMM: AMX columns are
    // token lanes, so one tile operation handles up to 16 tokens. Every
    // block writes a disjoint range of y and safely supports a short tail.
    #pragma omp parallel for schedule(static)
    for (int block_begin = 0; block_begin < num_tokens; block_begin += 16) {
        const int B = (num_tokens - block_begin < 16) ?
                      (num_tokens - block_begin) : 16;
        shared_expert_amx_batch(x, y, w_sh_gate_transpose, w_sh_up_transpose,
                                w_sh_down_transpose, w.sh_s_gate, w.sh_s_up,
                                w.sh_s_down, xq_total, s_x_total, block_begin,
                                B, d_model, d_ff);
    }

    #endif
    // route expert
    for (int t = 0; t < num_tokens; ++t) {
    
        int* topk_idx = topk_idx_total[t];
        int8_t* xq = xq_total + (size_t)t * d_model;
        float* s = s_total[t];
        const float* xt = x + (size_t)t * d_model;
        float* yt = y + (size_t)t * d_model;
        float &gate_sum = gate_sum_total[t];
        float &s_x = s_x_total[t];

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