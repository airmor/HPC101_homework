// Main task: optimize the MoE forward pass.

#include "moe.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

// AMX code is compiled only when the compiler enabled the AMX ISA.  This keeps
// the AVX-512 fallback buildable on machines/toolchains without AMX support.
#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
#define USE_AMX 1
#else
#define USE_AMX 0
#endif

float* w_router_transpose;
uint8_t* w_sh_gate_transpose;
uint8_t* w_sh_up_transpose;
uint8_t* w_sh_down_transpose;
uint8_t* w_gate_transpose;
uint8_t* w_up_transpose;
uint8_t* w_down_transpose;

static int g_router_padded_experts = 0;

// The benchmark invokes moe_forward_optimized many times.  Allocating and
// leaking per-token buffers in that hot path dominated the AMX speedup.  The
// problem bounds are fixed by moe.h, so reuse one cache-aligned workspace.
alignas(64) static int8_t xq_workspace[MAX_NUM_TOKENS * MAX_D_MODEL];
alignas(64) static float score_workspace[MAX_NUM_TOKENS * MAX_NUM_EXPERTS];
alignas(64) static int topk_workspace[MAX_NUM_TOKENS * MAX_TOP_K];
alignas(64) static float gate_sum_workspace[MAX_NUM_TOKENS];
alignas(64) static float x_scale_workspace[MAX_NUM_TOKENS];

static inline __m512 exp512_approx_ps(__m512 x) {
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

static inline int32_t sum_int8(const int8_t* values, int count) {
    __m512i sum = _mm512_setzero_si512();
    for (int i = 0; i < count; i += 16) {
        const __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(values + i));
        sum = _mm512_add_epi32(sum, _mm512_cvtepi8_epi32(v));
    }
    return _mm512_reduce_add_epi32(sum);
}

static void pack_vnni_weights(const int8_t* src, uint8_t* dst,
                              int rows, int cols) {
    assert(rows % 16 == 0);
    assert(cols % 4 == 0);
    for (int r = 0; r < rows; r += 16) {
        for (int c = 0; c < cols; c += 4) {
            uint8_t* packed = dst + static_cast<size_t>(r) * cols +
                              static_cast<size_t>(c) * 16;
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 4; ++j) {
                    packed[i * 4 + j] = static_cast<uint8_t>(
                        static_cast<int>(src[static_cast<size_t>(r + i) * cols + c + j]) + 128);
                }
            }
        }
    }
}

#if USE_AMX

struct alignas(64) amx_tilecfg {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

static inline void load_amx_shared_config(int tokens_in_block) {
    assert(tokens_in_block >= 1 && tokens_in_block <= 16);
    amx_tilecfg cfg{};
    cfg.palette_id = 1;
    cfg.rows[0] = cfg.rows[1] = 16;
    cfg.colsb[0] = cfg.colsb[1] = static_cast<uint16_t>(tokens_in_block * 4);
    cfg.rows[2] = cfg.rows[3] = 16;
    cfg.colsb[2] = cfg.colsb[3] = 64;
    cfg.rows[4] = 16;
    cfg.colsb[4] = static_cast<uint16_t>(tokens_in_block * 4);
    _tile_loadconfig(&cfg);
}

// AMX consumes one 16x64 byte tile at a time.  The layout is different from
// VNNI's 16-output-channel x 4-K packing used for routed experts.
static void pack_amx_weights(const int8_t* src, uint8_t* dst,
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

// Compute the shared FFN for up to 16 tokens.  Packing the activation once
// lets gate and up share it, and AMX turns all three projections into GEMMs.
static void shared_expert_amx_batch(const float* __restrict x,
                                    float* __restrict y,
                                    const uint8_t* __restrict w_gate,
                                    const uint8_t* __restrict w_up,
                                    const uint8_t* __restrict w_down,
                                    float s_gate, float s_up, float s_down,
                                    const int8_t* __restrict xq_total,
                                    const float* __restrict s_x_total,
                                    int block_begin, int B,
                                    int d_model, int d_ff) {
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
        x_sum[b] = sum_int8(xq_total + static_cast<size_t>(block_begin + b) * d_model,
                             d_model);
    }
    for (int k0 = 0; k0 < d_model; k0 += 64) {
        int8_t* const dst_block = x_tile + static_cast<size_t>(k0 / 4) * B * 4;
        for (int k4 = 0; k4 < 16; ++k4) {
            int8_t* const dst_row = dst_block + static_cast<size_t>(k4) * B * 4;
            for (int b = 0; b < B; ++b) {
                const int8_t* src = xq_total + static_cast<size_t>(block_begin + b) * d_model + k0 + k4 * 4;
                std::memcpy(dst_row + b * 4, src, 4);
            }
        }
    }

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
            const __m512 silu = _mm512_div_ps(
                vg, _mm512_add_ps(_mm512_set1_ps(1.0f),
                                  exp512_approx_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg))));
            alignas(64) float h_values[16];
            _mm512_store_ps(h_values, _mm512_mul_ps(silu, vu));
            for (int b = 0; b < B; ++b) {
                const float value = h_values[b];
                hidden[static_cast<size_t>(b) * d_ff + f + i] = value;
                h_amax[b] = (std::fabs(value) > h_amax[b]) ? std::fabs(value) : h_amax[b];
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        const float inv_s_h = h_amax[b] > 0.0f ? 127.0f / h_amax[b] : 1.0f;
        s_h[b] = h_amax[b] > 0.0f ? h_amax[b] / 127.0f : 1.0f;
        int8_t* const hq_token = hq + static_cast<size_t>(b) * d_ff;
        const float* const h_token = hidden + static_cast<size_t>(b) * d_ff;
        const __m512 inv = _mm512_set1_ps(inv_s_h);
        for (int f = 0; f < d_ff; f += 16) {
            const __m512i h_i32 = _mm512_cvt_roundps_epi32(
                _mm512_mul_ps(_mm512_loadu_ps(h_token + f), inv),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(hq_token + f),
                             _mm512_cvtsepi32_epi8(h_i32));
        }
        hq_sum[b] = sum_int8(hq_token, d_ff);
    }

    for (int f0 = 0; f0 < d_ff; f0 += 64) {
        int8_t* const dst_block = h_tile + static_cast<size_t>(f0 / 4) * B * 4;
        for (int f4 = 0; f4 < 16; ++f4) {
            int8_t* const dst_row = dst_block + static_cast<size_t>(f4) * B * 4;
            for (int b = 0; b < B; ++b) {
                std::memcpy(dst_row + b * 4,
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
                y[static_cast<size_t>(block_begin + b) * d_model + d + i] =
                    x[static_cast<size_t>(block_begin + b) * d_model + d + i] +
                    static_cast<float>(acc) * (s_h[b] * s_down);
            }
        }
    }
    _tile_release();
}

#endif  // USE_AMX

static void expert_ffn(const uint8_t* __restrict w_gate,
                       const uint8_t* __restrict w_up,
                       const uint8_t* __restrict w_down,
                       float s_gate, float s_up, float s_down,
                       const int8_t* __restrict xq, float s_x,
                       float* __restrict out, int d_model, int d_ff) {
    alignas(64) float h[MAX_D_FF];
    alignas(64) int8_t hq[MAX_D_FF];

    const int32_t x_sum = sum_int8(xq, d_model);
    const __m512i x_correction = _mm512_set1_epi32(-128 * x_sum);
    const __m512 gate_scale = _mm512_set1_ps(s_x * s_gate);
    const __m512 up_scale = _mm512_set1_ps(s_x * s_up);
    __m512 h_amax_vec = _mm512_setzero_ps();

    for (int f = 0; f < d_ff; f += 16) {
        __m512i gate_acc = x_correction;
        __m512i up_acc = x_correction;
        const size_t f_offset = static_cast<size_t>(f) * d_model;
        for (int k = 0; k < d_model; k += 4) {
            uint32_t x4;
            std::memcpy(&x4, xq + k, sizeof(x4));
            const __m512i x4_i32 = _mm512_set1_epi32(static_cast<int32_t>(x4));
            const size_t k4_offset = static_cast<size_t>(k / 4) * 64;
            gate_acc = _mm512_dpbusd_epi32(
                gate_acc,
                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + k4_offset)),
                x4_i32);
            up_acc = _mm512_dpbusd_epi32(
                up_acc,
                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + k4_offset)),
                x4_i32);
        }
        const __m512 vg = _mm512_mul_ps(gate_scale, _mm512_cvtepi32_ps(gate_acc));
        const __m512 vu = _mm512_mul_ps(up_scale, _mm512_cvtepi32_ps(up_acc));
        const __m512 silu = _mm512_div_ps(
            vg, _mm512_add_ps(_mm512_set1_ps(1.0f),
                              exp512_approx_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg))));
        const __m512 h_vec = _mm512_mul_ps(silu, vu);
        h_amax_vec = _mm512_max_ps(h_amax_vec, _mm512_abs_ps(h_vec));
        _mm512_storeu_ps(h + f, h_vec);
    }

    const float h_amax = _mm512_reduce_max_ps(h_amax_vec);
    const float s_h = h_amax > 0.0f ? h_amax / 127.0f : 1.0f;
    const __m512 inv_s_h = _mm512_set1_ps(h_amax > 0.0f ? 127.0f / h_amax : 1.0f);
    for (int f = 0; f < d_ff; f += 16) {
        const __m512i h_i32 = _mm512_cvt_roundps_epi32(
            _mm512_mul_ps(_mm512_loadu_ps(h + f), inv_s_h),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(hq + f),
                         _mm512_cvtsepi32_epi8(h_i32));
    }

    const int32_t hq_sum = sum_int8(hq, d_ff);
    const __m512i h_correction = _mm512_set1_epi32(-128 * hq_sum);
    const __m512 down_scale = _mm512_set1_ps(s_h * s_down);
    for (int d = 0; d < d_model; d += 16) {
        __m512i acc = h_correction;
        const size_t d_offset = static_cast<size_t>(d) * d_ff;
        for (int f = 0; f < d_ff; f += 4) {
            uint32_t hq4;
            std::memcpy(&hq4, hq + f, sizeof(hq4));
            const __m512i hq4_i32 = _mm512_set1_epi32(static_cast<int32_t>(hq4));
            const size_t f4_offset = static_cast<size_t>(f / 4) * 64;
            acc = _mm512_dpbusd_epi32(
                acc,
                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + f4_offset)),
                hq4_i32);
        }
        _mm512_storeu_ps(out + d,
                         _mm512_mul_ps(_mm512_cvtepi32_ps(acc), down_scale));
    }
}

void preprocess(MoEWeights& w) {
    g_router_padded_experts = (w.num_experts + 15) & ~15;
    w_router_transpose = new float[static_cast<size_t>(g_router_padded_experts) * w.d_model]();
    for (int e = 0; e < w.num_experts; ++e) {
        const int block = e & ~15;
        const int lane = e & 15;
        for (int d = 0; d < w.d_model; ++d) {
            w_router_transpose[static_cast<size_t>(block) * w.d_model +
                               static_cast<size_t>(d) * 16 + lane] =
                w.w_router[static_cast<size_t>(e) * w.d_model + d];
        }
    }

    const size_t gate_up_size = static_cast<size_t>(w.d_ff) * w.d_model;
    const size_t down_size = static_cast<size_t>(w.d_model) * w.d_ff;
    w_sh_gate_transpose = new uint8_t[gate_up_size];
    w_sh_up_transpose = new uint8_t[gate_up_size];
    w_sh_down_transpose = new uint8_t[down_size];
#if USE_AMX
    pack_amx_weights(w.sh_gate, w_sh_gate_transpose, w.d_ff, w.d_model);
    pack_amx_weights(w.sh_up, w_sh_up_transpose, w.d_ff, w.d_model);
    pack_amx_weights(w.sh_down, w_sh_down_transpose, w.d_model, w.d_ff);
#else
    pack_vnni_weights(w.sh_gate, w_sh_gate_transpose, w.d_ff, w.d_model);
    pack_vnni_weights(w.sh_up, w_sh_up_transpose, w.d_ff, w.d_model);
    pack_vnni_weights(w.sh_down, w_sh_down_transpose, w.d_model, w.d_ff);
#endif

    w_gate_transpose = new uint8_t[static_cast<size_t>(w.num_experts) * gate_up_size];
    w_up_transpose = new uint8_t[static_cast<size_t>(w.num_experts) * gate_up_size];
    w_down_transpose = new uint8_t[static_cast<size_t>(w.num_experts) * down_size];
    for (int e = 0; e < w.num_experts; ++e) {
        pack_vnni_weights(w.w_gate + static_cast<size_t>(e) * gate_up_size,
                          w_gate_transpose + static_cast<size_t>(e) * gate_up_size,
                          w.d_ff, w.d_model);
        pack_vnni_weights(w.w_up + static_cast<size_t>(e) * gate_up_size,
                          w_up_transpose + static_cast<size_t>(e) * gate_up_size,
                          w.d_ff, w.d_model);
        pack_vnni_weights(w.w_down + static_cast<size_t>(e) * down_size,
                          w_down_transpose + static_cast<size_t>(e) * down_size,
                          w.d_model, w.d_ff);
    }
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    // Stage 1: router, Top-K metadata, and quantization.  All output arrays
    // are indexed by token, hence this is race-free and allocation-free.
#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;

        for (int e = 0; e < g_router_padded_experts; e += 16) {
            __m512 acc = _mm512_setzero_ps();
            const float* const router_block = w_router_transpose + static_cast<size_t>(e) * d_model;
            for (int d = 0; d < d_model; ++d) {
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(router_block + static_cast<size_t>(d) * 16),
                                      _mm512_set1_ps(xt[d]), acc);
            }
            const __m512 s = _mm512_div_ps(
                _mm512_set1_ps(1.0f),
                _mm512_add_ps(_mm512_set1_ps(1.0f),
                              exp512_approx_ps(_mm512_sub_ps(_mm512_setzero_ps(), acc))));
            _mm512_storeu_ps(scores + e, s);
        }

        int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        bool used[MAX_NUM_EXPERTS] = {};
        float gate_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            for (int e = 0; e < num_experts; ++e) {
                if (!used[e] && (best < 0 || scores[e] + w.bias[e] > scores[best] + w.bias[best])) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
            gate_sum += scores[best];
        }
        gate_sum_workspace[t] = gate_sum;

        __m512 x_amax_vec = _mm512_setzero_ps();
        for (int d = 0; d < d_model; d += 16) {
            x_amax_vec = _mm512_max_ps(x_amax_vec, _mm512_abs_ps(_mm512_loadu_ps(xt + d)));
        }
        const float x_amax = _mm512_reduce_max_ps(x_amax_vec);
        const float s_x = x_amax > 0.0f ? x_amax / 127.0f : 1.0f;
        x_scale_workspace[t] = s_x;
        const __m512 inv_s_x = _mm512_set1_ps(x_amax > 0.0f ? 127.0f / x_amax : 1.0f);
        for (int d = 0; d < d_model; d += 16) {
            const __m512i q = _mm512_cvt_roundps_epi32(
                _mm512_mul_ps(_mm512_loadu_ps(xt + d), inv_s_x),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(xq + d),
                             _mm512_cvtsepi32_epi8(q));
        }
    }

    // Stage 2: the shared expert is common to all tokens.  Use AMX across a
    // 16-token batch; the fallback preserves the same VNNI implementation.
#if USE_AMX
    const int blocks = (num_tokens + 15) / 16;
#pragma omp parallel for if (blocks > 1) schedule(static)
    for (int block = 0; block < blocks; ++block) {
        const int block_begin = block * 16;
        const int B = (num_tokens - block_begin < 16) ? num_tokens - block_begin : 16;
        shared_expert_amx_batch(x, y, w_sh_gate_transpose, w_sh_up_transpose,
                                w_sh_down_transpose, w.sh_s_gate, w.sh_s_up,
                                w.sh_s_down, xq_workspace, x_scale_workspace,
                                block_begin, B, d_model, d_ff);
    }
#else
#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        alignas(64) float out[MAX_D_MODEL];
        const int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
        expert_ffn(w_sh_gate_transpose, w_sh_up_transpose, w_sh_down_transpose,
                   w.sh_s_gate, w.sh_s_up, w.sh_s_down, xq,
                   x_scale_workspace[t], out, d_model, d_ff);
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const yt = y + static_cast<size_t>(t) * d_model;
        for (int d = 0; d < d_model; d += 16) {
            _mm512_storeu_ps(yt + d, _mm512_add_ps(_mm512_loadu_ps(xt + d),
                                                    _mm512_loadu_ps(out + d)));
        }
    }
#endif

    // Stage 3: routed experts were previously serial, leaving most CPU cores
    // idle after the AMX shared pass.  Tokens are independent, so parallelize
    // this expensive top-k FFN work as well.
#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        alignas(64) float out[MAX_D_MODEL];
        const int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        const float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        float* const yt = y + static_cast<size_t>(t) * d_model;
        const float inv_gate_sum = 1.0f / gate_sum_workspace[t];

        for (int k = 0; k < top_k; ++k) {
            const int e = topk_idx[k];
            expert_ffn(w_gate_transpose + static_cast<size_t>(e) * d_ff * d_model,
                       w_up_transpose + static_cast<size_t>(e) * d_ff * d_model,
                       w_down_transpose + static_cast<size_t>(e) * d_model * d_ff,
                       w.s_gate[e], w.s_up[e], w.s_down[e], xq,
                       x_scale_workspace[t], out, d_model, d_ff);
            const __m512 gate = _mm512_set1_ps(scores[e] * inv_gate_sum);
            for (int d = 0; d < d_model; d += 16) {
                _mm512_storeu_ps(yt + d, _mm512_fmadd_ps(gate, _mm512_loadu_ps(out + d),
                                                          _mm512_loadu_ps(yt + d)));
            }
        }
    }
}
