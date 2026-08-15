// Main task: optimize the MoE forward pass.

#include "moe.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILE_DATA
#define XFEATURE_XTILE_DATA 18
#endif
#endif

// AMX code is compiled only when the compiler enabled the AMX ISA.  This keeps
// the AVX-512 fallback buildable on machines/toolchains without AMX support.
#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
#define USE_AMX 1
#else
#define USE_AMX 0
#endif

#if USE_AMX && defined(__linux__)
// AMX tile data (XFEATURE_XTILE_DATA) is permission-gated by the kernel on
// Sapphire Rapids.  The first execution of an AMX instruction other than
// LDTILECFG/STTILECFG/TILERELEASE raises SIGILL unless the process has
// previously requested the feature via arch_prctl(ARCH_REQ_XCOMP_PERM).
// Request it once per process (thread-safe via the std::call_once guard).
#include <mutex>
static std::once_flag g_amx_perm_flag;
static inline void amx_request_permission() {
    std::call_once(g_amx_perm_flag, [] {
        syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILE_DATA);
    });
}
#endif

float* w_router_transpose;
// Router weights quantized and packed as 16 output rows x 4-K-byte groups
// for VNNI.  The +128 encoding is corrected with the token xq sum in the
// kernel, exactly as in the expert W8A8 projections.
uint8_t* w_router_vnni;
// AMX-tiled router weights for the S4 batched-GEMM path.  Layout matches
// pack_amx_weights: [16 expert rows][64 K-byte cols] per tile, +128 encoding.
// Only the router GEMV changes; s_router is shared with the VNNI path.
uint8_t* w_router_amx;
float* s_router;
uint8_t* w_sh_gate_transpose;
uint8_t* w_sh_up_transpose;
uint8_t* w_sh_down_transpose;
// VNNI-packed shared expert, used by the small-batch path that folds the
// shared expert into the same parallel pool as the routed experts (AMX
// batching only pays off when a full 16-token tile can be filled).
uint8_t* w_sh_gate_vnni;
uint8_t* w_sh_up_vnni;
uint8_t* w_sh_down_vnni;
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
// Per-token, per-expert-slot FFN outputs for the small-batch combine path.
// Slot 0 is the shared expert (gate = 1); slots 1..K are the routed experts.
alignas(64) static float expert_out_workspace[MAX_NUM_TOKENS *
                                               (MAX_TOP_K + 1) * MAX_D_MODEL];

// Per-token sum of quantized activations (for the int8 x_correction term
// used by the VNNI gate/up dot products).  Shared across all expert slots
// of a token because xq is per-token.
alignas(64) static int32_t x_sum_workspace[MAX_NUM_TOKENS];

// Batched expert FFN: process B tokens of the SAME expert in one call so
// each weight ZMM is loaded once and reused across the inner B-token loop.
// Static workspace (NOT stack) so the large h/hq buffers do not bloat the
// stack frame of moe_forward_optimized — round-004 showed stack bloat from
// per-call alignas(64) arrays caused a catastrophic S2 binary-level
// regression (+11472%) via GCC stack-frame rearrangement of the intra-FFN
// path.  Using file-scope statics keeps each thread's frame small.
constexpr int EXPERT_FFN_BATCH_B_MAX = 8;

// Count-sort + flat-batch workspaces for the batched-FFN dispatch path.
alignas(64) static int expert_token_count[MAX_NUM_EXPERTS + 2];
alignas(64) static int expert_token_offset[MAX_NUM_EXPERTS + 2];
alignas(64) static int sorted_token_ids[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];
alignas(64) static int sorted_slot_ids[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];
alignas(64) static int batch_expert_id[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];
alignas(64) static int batch_B[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];
alignas(64) static int batch_token_start[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];
alignas(64) static int batch_is_shared[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];

// Intra-FFN workspaces: when N is tiny (S1/S2: one token, 5 (token,slot)
// tasks), the (token, slot) task space is too small to fill 16 threads.  We
// split each expert_ffn internally across threads — gate/up by 16-wide
// f-block, down by 16-wide d-block — which requires per-(token, slot)
// hidden-state buffers instead of stack arrays.  The three-stage pipeline
// (gate/up → reduce+requant → down) has two barriers but turns 5 tasks into
// 5×(d_ff/16 + d_model/16) fine-grained tasks.
alignas(64) static float intra_h[MAX_NUM_TOKENS * (MAX_TOP_K + 1) * MAX_D_FF];
alignas(64) static int8_t intra_hq[MAX_NUM_TOKENS * (MAX_TOP_K + 1) * MAX_D_FF];
alignas(64) static float intra_fblock_amax[MAX_NUM_TOKENS * (MAX_TOP_K + 1) *
                                           (MAX_D_FF / 16)];
alignas(64) static float intra_s_h[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];
alignas(64) static int32_t intra_hq_sum[MAX_NUM_TOKENS * (MAX_TOP_K + 1)];

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
// Quantize fp32 router weights per expert row and pack them in the VNNI
// layout: [16 rows][d_model / 4 groups][4 bytes].  The stored uint8 values
// are q + 128; router GEMV subtracts 128 * sum(xq) from every accumulator.
static void pack_router_vnni(const float* src, uint8_t* dst, float* dst_scale,
                             int rows, int cols) {
    assert(rows % 16 == 0);
    assert(cols % 4 == 0);
    for (int r = 0; r < rows; r += 16) {
        float amax[16] = {};
        for (int i = 0; i < 16; ++i) {
            float a = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const float v = src[static_cast<size_t>(r + i) * cols + c];
                const float av = v < 0.0f ? -v : v;
                if (av > a) a = av;
            }
            amax[i] = a;
            dst_scale[r + i] = a > 0.0f ? a / 127.0f : 1.0f;
        }
        for (int c = 0; c < cols; c += 4) {
            uint8_t* const packed = dst + static_cast<size_t>(r) * cols +
                                    static_cast<size_t>(c) * 16;
            for (int i = 0; i < 16; ++i) {
                const float inv = amax[i] > 0.0f ? 127.0f / amax[i] : 1.0f;
                for (int j = 0; j < 4; ++j) {
                    int q = lrintf(src[static_cast<size_t>(r + i) * cols + c + j] * inv);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    packed[i * 4 + j] = static_cast<uint8_t>(q + 128);
                }
            }
        }
    }
}
// Quantize fp32 router weights per expert row and pack them in the AMX tile
// layout: [16 expert rows][64-byte K-tiles].  Reuses the SAME per-row amax and
// scale (s_router) as pack_router_vnni so the AMX router path produces scores
// numerically identical to the VNNI router path (both recover the signed int8
// dot product via the +128 / -128*sum(xq) correction).  cols must be a
// multiple of 64 (the AMX tile width); d_model=512 satisfies this for S4.
static void pack_router_amx(const float* src, uint8_t* dst, const float* src_scale,
                            int rows, int cols) {
#if USE_AMX
    assert(rows % 16 == 0);
    assert(cols % 64 == 0);
    for (int r = 0; r < rows; r += 16) {
        float amax[16] = {};
        for (int i = 0; i < 16; ++i) {
            float a = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const float v = src[static_cast<size_t>(r + i) * cols + c];
                const float av = v < 0.0f ? -v : v;
                if (av > a) a = av;
            }
            amax[i] = a;
            // s_router is shared with the VNNI path; verify consistency.
            const float expect = a > 0.0f ? a / 127.0f : 1.0f;
            (void)expect; (void)src_scale;
        }
        for (int k0 = 0; k0 < cols; k0 += 64) {
            uint8_t* tile = dst + static_cast<size_t>(r) * cols +
                            static_cast<size_t>(k0) * 16;
            for (int i = 0; i < 16; ++i) {
                const float inv = amax[i] > 0.0f ? 127.0f / amax[i] : 1.0f;
                for (int k = 0; k < 64; ++k) {
                    int q = lrintf(src[static_cast<size_t>(r + i) * cols + k0 + k] * inv);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    tile[i * 64 + k] = static_cast<uint8_t>(q + 128);
                }
            }
        }
    }
#else
    (void)src; (void)dst; (void)src_scale; (void)rows; (void)cols;
#endif
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

// AMX tile configuration for the batched router GEMM.  C[M,N] += A[M,K]*B[K,N]
// where M=16 experts, N=tokens_in_block, K=d_model (in groups of 64).
//   tile 0 = C (16 experts x tokens, 4 bytes per token cell)
//   tile 2 = B = router weights (16 rows x 64 K-cols, byte)
//   tile 4 = A = packed token activations (16 K-rows x tokens*4, byte)
// The token activation tile rows step through d_model in chunks of 64
// (each K-row holds 64 bytes); rows fixed at 16 so the tile spans 1024 K bytes.
static inline void load_amx_router_config(int tokens_in_block) {
    assert(tokens_in_block >= 1 && tokens_in_block <= 16);
    amx_tilecfg cfg{};
    cfg.palette_id = 1;
    cfg.rows[0] = 16;
    cfg.colsb[0] = static_cast<uint16_t>(tokens_in_block * 4);
    cfg.rows[2] = 16;
    cfg.colsb[2] = 64;
    cfg.rows[4] = 16;
    cfg.colsb[4] = static_cast<uint16_t>(tokens_in_block * 4);
    _tile_loadconfig(&cfg);
}
#else

struct amx_tilecfg;
static inline void load_amx_shared_config(int) {}
static inline void load_amx_router_config(int) {}
#endif  // USE_AMX (tilecfg helpers)

// Batched router GEMM for the S4 large-N path.  Computes coarse W8A8 router
// scores for a 16-token block: score[expert][token] = sum_k w[expert][k] *
// xq[token][k], using AMX _tile_dpbusd to process 16 experts x up to 16 tokens
// x 64 K-bytes per tile.  Mirrors shared_expert_amx_batch's packing and
// +128-correction handling so the int8 dot product is recovered exactly.
//
// Output is written to `scores` at the same layout the VNNI router path uses:
// scores[t * MAX_NUM_EXPERTS + e] (token-major, padded-experts stride), so the
// downstream Top-M heap and FP32 rescore need no changes.
//
// xq_total uses the MAX_D_MODEL-per-token stride fixed by the quantization
// stage (see the shared_expert_amx_batch comment for the original stride bug).
static void router_amx_batch(const int8_t* __restrict xq_total,
                             const float* __restrict s_x_total,
                             const int32_t* __restrict x_sum_total,
                             const uint8_t* __restrict w_router,
                             const float* __restrict s_router_w,
                             float* __restrict scores,
                             int block_begin, int B,
                             int d_model, int num_experts,
                             int padded_experts) {
#if USE_AMX
    // x_tile layout: for each 4-K group, B tokens' 4 bytes contiguously, then
    // 16 K-rows of 64 bytes each tile-load.  Matches pack_token_amx_b but
    // reused here with the explicit stride for clarity.
    alignas(64) int8_t x_tile[(MAX_D_MODEL / 4) * 16 * 4];
    alignas(64) int32_t acc_tile[16 * 16];
    int32_t x_sum[16] = {};

    for (int b = 0; b < B; ++b) {
        x_sum[b] = x_sum_total[block_begin + b];
    }
    for (int k0 = 0; k0 < d_model; k0 += 64) {
        int8_t* const dst_block = x_tile + static_cast<size_t>(k0 / 4) * B * 4;
        for (int k4 = 0; k4 < 16; ++k4) {
            int8_t* const dst_row = dst_block + static_cast<size_t>(k4) * B * 4;
            for (int b = 0; b < B; ++b) {
                const int8_t* src = xq_total +
                    static_cast<size_t>(block_begin + b) * MAX_D_MODEL + k0 + k4 * 4;
                std::memcpy(dst_row + b * 4, src, 4);
            }
        }
    }

    load_amx_router_config(B);
    for (int e = 0; e < padded_experts; e += 16) {
        _tile_zero(0);
        for (int k0 = 0; k0 < d_model; k0 += 64) {
            // B tile: 16 expert rows x 64 K-bytes, laid out by pack_amx_weights
            // as dst[e * d_model + k0 * 16][i*64 + k].
            _tile_loadd(2,
                w_router + static_cast<size_t>(e) * d_model +
                    static_cast<size_t>(k0) * 16, 64);
            _tile_loadd(4,
                x_tile + static_cast<size_t>(k0 / 4) * B * 4, B * 4);
            _tile_dpbusd(0, 2, 4);
        }
        _tile_stored(0, acc_tile, B * 4);

        // Dequantize: int score = acc - 128 * x_sum; fp32 logit =
        // (s_x * s_router_e) * int_score; sigmoid.  Padded trailing experts have
        // zero weights (pack_router_vnni zero-pads), so their scores are 0.5
        // sigmoid of the x_correction term — these never enter top-K because
        // num_experts bounds the Top-M heap scan.
        for (int i = 0; i < 16; ++i) {
            const int expert = e + i;
            alignas(64) float logits[16] = {};
            for (int b = 0; b < B; ++b) {
                const int32_t corrected = acc_tile[i * B + b] - 128 * x_sum[b];
                const float sc = s_x_total[block_begin + b] * s_router_w[expert];
                logits[b] = static_cast<float>(corrected) * sc;
            }
            const __m512 acc_f = _mm512_load_ps(logits);
            const __m512 s = _mm512_div_ps(
                _mm512_set1_ps(1.0f),
                _mm512_add_ps(_mm512_set1_ps(1.0f),
                              exp512_approx_ps(_mm512_sub_ps(_mm512_setzero_ps(), acc_f))));
            alignas(64) float svals[16];
            _mm512_store_ps(svals, s);
            for (int b = 0; b < B; ++b) {
                scores[static_cast<size_t>(block_begin + b) * MAX_NUM_EXPERTS + expert] = svals[b];
            }
        }
    }
    _tile_release();
#else
    (void)xq_total; (void)s_x_total; (void)x_sum_total; (void)w_router;
    (void)s_router_w; (void)scores; (void)block_begin; (void)B;
    (void)d_model; (void)num_experts; (void)padded_experts;
#endif
}

// Pack a single token's int8 activation into AMX's expected layout for the
// "A" matrix in C[M,N] += A[M,K]*B[K,N].  Here the activation is the B side
// of a token-major matmul, so we lay out [K/4][B*4] groups of 4 contiguous
// bytes per token.  Used to lift a single routed-expert token batch into a
// dense AMX GEMM when tokens for one expert are scarce.
static inline void pack_token_amx_b(const int8_t* __restrict xq,
                                    int8_t* __restrict dst,
                                    int B, int d_model) {
    // dst layout: for each 4-K group, B tokens' 4 bytes contiguously.
    for (int k0 = 0; k0 < d_model; k0 += 64) {
        int8_t* const dst_block = dst + static_cast<size_t>(k0 / 4) * B * 4;
        for (int k4 = 0; k4 < 16; ++k4) {
            int8_t* const dst_row = dst_block + static_cast<size_t>(k4) * B * 4;
            for (int b = 0; b < B; ++b) {
                std::memcpy(dst_row + b * 4, xq + static_cast<size_t>(b) * MAX_D_MODEL + k0 + k4 * 4, 4);
            }
        }
    }
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
// noinline+cold: isolates the large AMX body from S1's icache.
__attribute__((noinline, cold))
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
        // xq_workspace is laid out with stride MAX_D_MODEL per token (see the
        // quantization stage), NOT d_model.  Indexing by d_model here was a
        // stride bug that corrupted every token except the first.
        x_sum[b] = sum_int8(xq_total + static_cast<size_t>(block_begin + b) * MAX_D_MODEL,
                             d_model);
    }
    for (int k0 = 0; k0 < d_model; k0 += 64) {
        int8_t* const dst_block = x_tile + static_cast<size_t>(k0 / 4) * B * 4;
        for (int k4 = 0; k4 < 16; ++k4) {
            int8_t* const dst_row = dst_block + static_cast<size_t>(k4) * B * 4;
            for (int b = 0; b < B; ++b) {
                const int8_t* src = xq_total + static_cast<size_t>(block_begin + b) * MAX_D_MODEL + k0 + k4 * 4;
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

// Intra-FFN stage 1: compute the gate/up projection + SwiGLU activation for
// one 16-wide block of f indices [f0, f0+16) of a single (token, slot) task.
// Writes h[t,slot,f0:f0+16] and a local amax that the caller reduces across
// all f-blocks of the task to get the global h_amax for requantization.
//
// Mirrors the inner body of expert_ffn's gate/up loop: VNNI dpbusd on 4-K
// groups with 4-way unroll, then SiLU(h_gate * scale) * h_up * scale.
static inline void expert_ffn_gateup_block(
    const uint8_t* __restrict w_gate,
    const uint8_t* __restrict w_up,
    float s_gate, float s_up,
    const int8_t* __restrict xq, float s_x, int32_t x_sum,
    int d_model, int d_ff,
    int f0, float* __restrict h_out, float* __restrict amax_out) {
    const __m512i x_correction = _mm512_set1_epi32(-128 * x_sum);
    const __m512 gate_scale = _mm512_set1_ps(s_x * s_gate);
    const __m512 up_scale = _mm512_set1_ps(s_x * s_up);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);

    __m512i gate_acc = x_correction;
    __m512i up_acc = x_correction;
    const size_t f_offset = static_cast<size_t>(f0) * d_model;
    const int k_groups = d_model / 4;
    int k = 0;
    for (; k + 3 < k_groups; k += 4) {
        uint32_t x4[4];
        std::memcpy(x4, xq + k * 4, sizeof(x4));
        const __m512i xa = _mm512_set1_epi32(static_cast<int32_t>(x4[0]));
        const __m512i xb = _mm512_set1_epi32(static_cast<int32_t>(x4[1]));
        const __m512i xc = _mm512_set1_epi32(static_cast<int32_t>(x4[2]));
        const __m512i xd = _mm512_set1_epi32(static_cast<int32_t>(x4[3]));
        const size_t o0 = static_cast<size_t>(k) * 64;
        const size_t o1 = static_cast<size_t>(k + 1) * 64;
        const size_t o2 = static_cast<size_t>(k + 2) * 64;
        const size_t o3 = static_cast<size_t>(k + 3) * 64;
        const __m512i wg0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o0));
        const __m512i wg1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o1));
        const __m512i wg2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o2));
        const __m512i wg3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o3));
        const __m512i wu0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o0));
        const __m512i wu1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o1));
        const __m512i wu2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o2));
        const __m512i wu3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o3));
        gate_acc = _mm512_dpbusd_epi32(gate_acc, wg0, xa);
        up_acc   = _mm512_dpbusd_epi32(up_acc,   wu0, xa);
        gate_acc = _mm512_dpbusd_epi32(gate_acc, wg1, xb);
        up_acc   = _mm512_dpbusd_epi32(up_acc,   wu1, xb);
        gate_acc = _mm512_dpbusd_epi32(gate_acc, wg2, xc);
        up_acc   = _mm512_dpbusd_epi32(up_acc,   wu2, xc);
        gate_acc = _mm512_dpbusd_epi32(gate_acc, wg3, xd);
        up_acc   = _mm512_dpbusd_epi32(up_acc,   wu3, xd);
    }
    for (; k < k_groups; ++k) {
        uint32_t x4;
        std::memcpy(&x4, xq + k * 4, sizeof(x4));
        const __m512i x4_i32 = _mm512_set1_epi32(static_cast<int32_t>(x4));
        const size_t k4_offset = static_cast<size_t>(k) * 64;
        const __m512i wg = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(w_gate + f_offset + k4_offset));
        const __m512i wu = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(w_up + f_offset + k4_offset));
        gate_acc = _mm512_dpbusd_epi32(gate_acc, wg, x4_i32);
        up_acc   = _mm512_dpbusd_epi32(up_acc,   wu, x4_i32);
    }
    const __m512 vg = _mm512_mul_ps(gate_scale, _mm512_cvtepi32_ps(gate_acc));
    const __m512 vu = _mm512_mul_ps(up_scale, _mm512_cvtepi32_ps(up_acc));
    const __m512 silu = _mm512_div_ps(
        vg, _mm512_add_ps(one, exp512_approx_ps(_mm512_sub_ps(zero, vg))));
    const __m512 h_vec = _mm512_mul_ps(silu, vu);
    _mm512_storeu_ps(h_out, h_vec);
    *amax_out = _mm512_reduce_max_ps(_mm512_abs_ps(h_vec));
    (void)d_ff;
}

// Intra-FFN stage 3: compute the down projection for one 16-wide block of d
// indices [d0, d0+16) of a single (token, slot) task.  Reads the already-
// requantized hq[t,slot,*] and the per-task s_h/s_down.
static inline void expert_ffn_down_block(
    const uint8_t* __restrict w_down, float s_down,
    const int8_t* __restrict hq, int32_t hq_sum, float s_h,
    int d_model, int d_ff, int d0, float* __restrict out) {
    const __m512i h_correction = _mm512_set1_epi32(-128 * hq_sum);
    const __m512 down_scale = _mm512_set1_ps(s_h * s_down);
    const int f_groups = d_ff / 4;
    const size_t d_offset = static_cast<size_t>(d0) * d_ff;
    __m512i acc = h_correction;
    int g = 0;
    for (; g + 3 < f_groups; g += 4) {
        uint32_t hq4[4];
        std::memcpy(hq4, hq + g * 4, sizeof(hq4));
        const __m512i qa = _mm512_set1_epi32(static_cast<int32_t>(hq4[0]));
        const __m512i qb = _mm512_set1_epi32(static_cast<int32_t>(hq4[1]));
        const __m512i qc = _mm512_set1_epi32(static_cast<int32_t>(hq4[2]));
        const __m512i qd = _mm512_set1_epi32(static_cast<int32_t>(hq4[3]));
        const size_t o0 = static_cast<size_t>(g) * 64;
        const size_t o1 = static_cast<size_t>(g + 1) * 64;
        const size_t o2 = static_cast<size_t>(g + 2) * 64;
        const size_t o3 = static_cast<size_t>(g + 3) * 64;
        const __m512i w0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o0));
        const __m512i w1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o1));
        const __m512i w2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o2));
        const __m512i w3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o3));
        acc = _mm512_dpbusd_epi32(acc, w0, qa);
        acc = _mm512_dpbusd_epi32(acc, w1, qb);
        acc = _mm512_dpbusd_epi32(acc, w2, qc);
        acc = _mm512_dpbusd_epi32(acc, w3, qd);
    }
    for (; g < f_groups; ++g) {
        uint32_t hq4;
        std::memcpy(&hq4, hq + g * 4, sizeof(hq4));
        const __m512i hq4_i32 = _mm512_set1_epi32(static_cast<int32_t>(hq4));
        const size_t f4_offset = static_cast<size_t>(g) * 64;
        acc = _mm512_dpbusd_epi32(
            acc,
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + f4_offset)),
            hq4_i32);
    }
    _mm512_storeu_ps(out, _mm512_mul_ps(_mm512_cvtepi32_ps(acc), down_scale));
}

// Intra-FFN stage 2 (serial per task): reduce the per-block amax values to a
// global h_amax, compute s_h, then requantize the full h[t,slot,*] into hq.
// Also computes hq_sum (needed by the down projection's correction term).
static inline void expert_ffn_requant(
    const float* __restrict fblock_amax, int n_fblocks,
    const float* __restrict h, int8_t* __restrict hq,
    float* __restrict s_h_out, int32_t* __restrict hq_sum_out) {
    __m512 amax_vec = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n_fblocks; i += 16) {
        amax_vec = _mm512_max_ps(amax_vec, _mm512_loadu_ps(fblock_amax + i));
    }
    float h_amax = _mm512_reduce_max_ps(amax_vec);
    for (; i < n_fblocks; ++i) {
        const float a = fblock_amax[i];
        if (a > h_amax) h_amax = a;
    }
    const float s_h = h_amax > 0.0f ? h_amax / 127.0f : 1.0f;
    *s_h_out = s_h;
    const __m512 inv_s_h = _mm512_set1_ps(h_amax > 0.0f ? 127.0f / h_amax : 1.0f);
    const int d_ff = n_fblocks * 16;
    for (int f = 0; f < d_ff; f += 16) {
        const __m512i h_i32 = _mm512_cvt_roundps_epi32(
            _mm512_mul_ps(_mm512_loadu_ps(h + f), inv_s_h),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(hq + f),
                         _mm512_cvtsepi32_epi8(h_i32));
    }
    *hq_sum_out = sum_int8(hq, d_ff);
}

// Batched expert FFN: process B tokens of the SAME expert in one call.
// Weights are loaded into ZMMs once per K-group and reused across the inner
// B-token loop, cutting weight load traffic by ~Bx for the batched tokens.
// This is the mechanism validated in round-004 (S4 -40.43%); the S2 +11472%
// regression there was a binary-level side effect.  Here the function is
// marked noinline+cold so its ~330 lines of intrinsics and 8KB stack frame
// are placed at the far end of the code segment, isolating the per-task and
// intra-FFN paths (S1/S2/S3) from icache pressure and code-layout shifts.
// h/hq are stack-local so each thread gets its own copy (no race).
__attribute__((noinline, cold))
static void expert_ffn_batch(
    const uint8_t* __restrict w_gate,
    const uint8_t* __restrict w_up,
    const uint8_t* __restrict w_down,
    float s_gate, float s_up, float s_down,
    const int8_t* __restrict xq_list[EXPERT_FFN_BATCH_B_MAX],
    const float* __restrict s_x_list,
    const int32_t* __restrict x_sum_list,
    float* __restrict out_list[EXPERT_FFN_BATCH_B_MAX],
    int B, int d_model, int d_ff) {

    alignas(64) float h[EXPERT_FFN_BATCH_B_MAX][MAX_D_FF];
    alignas(64) int8_t hq[EXPERT_FFN_BATCH_B_MAX][MAX_D_FF];
    float h_amax[EXPERT_FFN_BATCH_B_MAX] = {};
    float s_h[EXPERT_FFN_BATCH_B_MAX];
    int32_t hq_sum[EXPERT_FFN_BATCH_B_MAX] = {};

    const int k_groups = d_model / 4;

    // Stage 1: gate/up + SwiGLU, batched across B tokens per f-block.
    for (int f = 0; f < d_ff; f += 16) {
        const size_t f_offset = static_cast<size_t>(f) * d_model;

        __m512i gate_acc[EXPERT_FFN_BATCH_B_MAX];
        __m512i up_acc[EXPERT_FFN_BATCH_B_MAX];
        for (int b = 0; b < B; ++b) {
            gate_acc[b] = _mm512_set1_epi32(-128 * x_sum_list[b]);
            up_acc[b]   = _mm512_set1_epi32(-128 * x_sum_list[b]);
        }

        int k = 0;
        for (; k + 3 < k_groups; k += 4) {
            const size_t o0 = static_cast<size_t>(k) * 64;
            const size_t o1 = static_cast<size_t>(k + 1) * 64;
            const size_t o2 = static_cast<size_t>(k + 2) * 64;
            const size_t o3 = static_cast<size_t>(k + 3) * 64;

            const __m512i wg0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o0));
            const __m512i wg1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o1));
            const __m512i wg2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o2));
            const __m512i wg3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o3));
            const __m512i wu0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o0));
            const __m512i wu1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o1));
            const __m512i wu2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o2));
            const __m512i wu3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o3));

            for (int b = 0; b < B; ++b) {
                uint32_t x4[4];
                std::memcpy(x4, xq_list[b] + k * 4, sizeof(x4));
                const __m512i xa = _mm512_set1_epi32(static_cast<int32_t>(x4[0]));
                const __m512i xb = _mm512_set1_epi32(static_cast<int32_t>(x4[1]));
                const __m512i xc = _mm512_set1_epi32(static_cast<int32_t>(x4[2]));
                const __m512i xd = _mm512_set1_epi32(static_cast<int32_t>(x4[3]));
                gate_acc[b] = _mm512_dpbusd_epi32(gate_acc[b], wg0, xa);
                up_acc[b]   = _mm512_dpbusd_epi32(up_acc[b],   wu0, xa);
                gate_acc[b] = _mm512_dpbusd_epi32(gate_acc[b], wg1, xb);
                up_acc[b]   = _mm512_dpbusd_epi32(up_acc[b],   wu1, xb);
                gate_acc[b] = _mm512_dpbusd_epi32(gate_acc[b], wg2, xc);
                up_acc[b]   = _mm512_dpbusd_epi32(up_acc[b],   wu2, xc);
                gate_acc[b] = _mm512_dpbusd_epi32(gate_acc[b], wg3, xd);
                up_acc[b]   = _mm512_dpbusd_epi32(up_acc[b],   wu3, xd);
            }
        }
        for (; k < k_groups; ++k) {
            const size_t k4_offset = static_cast<size_t>(k) * 64;
            const __m512i wg = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + k4_offset));
            const __m512i wu = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + k4_offset));
            for (int b = 0; b < B; ++b) {
                uint32_t x4;
                std::memcpy(&x4, xq_list[b] + k * 4, sizeof(x4));
                const __m512i x4_i32 = _mm512_set1_epi32(static_cast<int32_t>(x4));
                gate_acc[b] = _mm512_dpbusd_epi32(gate_acc[b], wg, x4_i32);
                up_acc[b]   = _mm512_dpbusd_epi32(up_acc[b],   wu, x4_i32);
            }
        }

        const __m512 zero = _mm512_setzero_ps();
        const __m512 one = _mm512_set1_ps(1.0f);
        for (int b = 0; b < B; ++b) {
            const __m512 gate_scale = _mm512_set1_ps(s_x_list[b] * s_gate);
            const __m512 up_scale = _mm512_set1_ps(s_x_list[b] * s_up);
            const __m512 vg = _mm512_mul_ps(gate_scale, _mm512_cvtepi32_ps(gate_acc[b]));
            const __m512 vu = _mm512_mul_ps(up_scale, _mm512_cvtepi32_ps(up_acc[b]));
            const __m512 silu = _mm512_div_ps(
                vg, _mm512_add_ps(one, exp512_approx_ps(_mm512_sub_ps(zero, vg))));
            const __m512 h_vec = _mm512_mul_ps(silu, vu);
            _mm512_storeu_ps(&h[b][f], h_vec);
            const __m512 abs_h = _mm512_abs_ps(h_vec);
            const float block_amax = _mm512_reduce_max_ps(abs_h);
            if (block_amax > h_amax[b]) h_amax[b] = block_amax;
        }
    }

    // Stage 2: per-token requant (independent across B).
    for (int b = 0; b < B; ++b) {
        __m512 amax_vec = _mm512_setzero_ps();
        for (int f = 0; f < d_ff; f += 16) {
            amax_vec = _mm512_max_ps(amax_vec, _mm512_abs_ps(_mm512_loadu_ps(&h[b][f])));
        }
        const float full_amax = _mm512_reduce_max_ps(amax_vec);
        h_amax[b] = full_amax;
        const float inv_s_h = full_amax > 0.0f ? 127.0f / full_amax : 1.0f;
        s_h[b] = full_amax > 0.0f ? full_amax / 127.0f : 1.0f;
        const __m512 inv = _mm512_set1_ps(inv_s_h);
        for (int f = 0; f < d_ff; f += 16) {
            const __m512i h_i32 = _mm512_cvt_roundps_epi32(
                _mm512_mul_ps(_mm512_loadu_ps(&h[b][f]), inv),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&hq[b][f]),
                             _mm512_cvtsepi32_epi8(h_i32));
        }
        hq_sum[b] = sum_int8(&hq[b][0], d_ff);
    }

    // Stage 3: down projection, batched across B tokens per d-block.
    const int f_groups = d_ff / 4;
    for (int d = 0; d < d_model; d += 16) {
        const size_t d_offset = static_cast<size_t>(d) * d_ff;

        __m512i acc[EXPERT_FFN_BATCH_B_MAX];
        for (int b = 0; b < B; ++b) {
            acc[b] = _mm512_set1_epi32(-128 * hq_sum[b]);
        }

        int g = 0;
        for (; g + 3 < f_groups; g += 4) {
            const size_t o0 = static_cast<size_t>(g) * 64;
            const size_t o1 = static_cast<size_t>(g + 1) * 64;
            const size_t o2 = static_cast<size_t>(g + 2) * 64;
            const size_t o3 = static_cast<size_t>(g + 3) * 64;
            const __m512i w0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o0));
            const __m512i w1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o1));
            const __m512i w2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o2));
            const __m512i w3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o3));

            for (int b = 0; b < B; ++b) {
                uint32_t hq4[4];
                std::memcpy(hq4, &hq[b][g * 4], sizeof(hq4));
                const __m512i qa = _mm512_set1_epi32(static_cast<int32_t>(hq4[0]));
                const __m512i qb = _mm512_set1_epi32(static_cast<int32_t>(hq4[1]));
                const __m512i qc = _mm512_set1_epi32(static_cast<int32_t>(hq4[2]));
                const __m512i qd = _mm512_set1_epi32(static_cast<int32_t>(hq4[3]));
                acc[b] = _mm512_dpbusd_epi32(acc[b], w0, qa);
                acc[b] = _mm512_dpbusd_epi32(acc[b], w1, qb);
                acc[b] = _mm512_dpbusd_epi32(acc[b], w2, qc);
                acc[b] = _mm512_dpbusd_epi32(acc[b], w3, qd);
            }
        }
        for (; g < f_groups; ++g) {
            const size_t f4_offset = static_cast<size_t>(g) * 64;
            const __m512i w = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + f4_offset));
            for (int b = 0; b < B; ++b) {
                uint32_t hq4;
                std::memcpy(&hq4, &hq[b][g * 4], sizeof(hq4));
                const __m512i hq4_i32 = _mm512_set1_epi32(static_cast<int32_t>(hq4));
                acc[b] = _mm512_dpbusd_epi32(acc[b], w, hq4_i32);
            }
        }

        for (int b = 0; b < B; ++b) {
            const __m512 down_scale = _mm512_set1_ps(s_h[b] * s_down);
            _mm512_storeu_ps(out_list[b] + d,
                             _mm512_mul_ps(_mm512_cvtepi32_ps(acc[b]), down_scale));
        }
    }
}

// S1's single-token path uses this function.  V7 integration (from
// lab2_s1_scratch): 4×K unroll with 4 independent accumulator chains per
// projection + broadcastd+shuffle xq load.
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

    // Fuse gate and up weight loads by walking them together: each k-group
    // reuses the same 4-byte xq broadcast for both projections, so loading
    // xq once (already done per k) and issuing back-to-back dpbusd lets the
    // two independent accumulators pipeline on the VNNI ports.
    //
    // V7 integration (from lab2_s1_scratch): load 4 K-groups (16 bytes) in one
    // __m128i and broadcast each 4-byte lane via broadcastd+shuffle, instead
    // of memcpy+set1 which materializes x4[4] on the stack.  The 4 sub-
    // accumulators start at zero and the +128 correction is applied once to
    // the reduced sum (not per sub-accumulator, which would 4× the correction).
    const int k_groups = d_model / 4;
    const int k_groups4 = k_groups / 4 * 4;
    for (int f = 0; f < d_ff; f += 16) {
        __m512i gate_acc0 = _mm512_setzero_si512();
        __m512i gate_acc1 = _mm512_setzero_si512();
        __m512i gate_acc2 = _mm512_setzero_si512();
        __m512i gate_acc3 = _mm512_setzero_si512();
        __m512i up_acc0 = _mm512_setzero_si512();
        __m512i up_acc1 = _mm512_setzero_si512();
        __m512i up_acc2 = _mm512_setzero_si512();
        __m512i up_acc3 = _mm512_setzero_si512();
        const size_t f_offset = static_cast<size_t>(f) * d_model;
        int k = 0;
        for (; k < k_groups4; k += 4) {
            __m128i x4v;
            std::memcpy(&x4v, xq + k * 4, sizeof(__m128i));
            const __m512i xa = _mm512_broadcastd_epi32(x4v);
            const __m512i xb = _mm512_broadcastd_epi32(
                _mm_shuffle_epi32(x4v, _MM_SHUFFLE(1, 1, 1, 1)));
            const __m512i xc = _mm512_broadcastd_epi32(
                _mm_shuffle_epi32(x4v, _MM_SHUFFLE(2, 2, 2, 2)));
            const __m512i xd = _mm512_broadcastd_epi32(
                _mm_shuffle_epi32(x4v, _MM_SHUFFLE(3, 3, 3, 3)));
            const size_t o0 = static_cast<size_t>(k) * 64;
            const size_t o1 = static_cast<size_t>(k + 1) * 64;
            const size_t o2 = static_cast<size_t>(k + 2) * 64;
            const size_t o3 = static_cast<size_t>(k + 3) * 64;
            const __m512i wg0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o0));
            const __m512i wg1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o1));
            const __m512i wg2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o2));
            const __m512i wg3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_gate + f_offset + o3));
            const __m512i wu0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o0));
            const __m512i wu1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o1));
            const __m512i wu2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o2));
            const __m512i wu3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_up + f_offset + o3));
            gate_acc0 = _mm512_dpbusd_epi32(gate_acc0, wg0, xa);
            up_acc0   = _mm512_dpbusd_epi32(up_acc0,   wu0, xa);
            gate_acc1 = _mm512_dpbusd_epi32(gate_acc1, wg1, xb);
            up_acc1   = _mm512_dpbusd_epi32(up_acc1,   wu1, xb);
            gate_acc2 = _mm512_dpbusd_epi32(gate_acc2, wg2, xc);
            up_acc2   = _mm512_dpbusd_epi32(up_acc2,   wu2, xc);
            gate_acc3 = _mm512_dpbusd_epi32(gate_acc3, wg3, xd);
            up_acc3   = _mm512_dpbusd_epi32(up_acc3,   wu3, xd);
        }
        __m512i gate_acc = _mm512_add_epi32(_mm512_add_epi32(gate_acc0, gate_acc1),
                                            _mm512_add_epi32(gate_acc2, gate_acc3));
        __m512i up_acc = _mm512_add_epi32(_mm512_add_epi32(up_acc0, up_acc1),
                                          _mm512_add_epi32(up_acc2, up_acc3));
        gate_acc = _mm512_add_epi32(gate_acc, x_correction);
        up_acc = _mm512_add_epi32(up_acc, x_correction);
        for (; k < k_groups; ++k) {
            uint32_t x4;
            std::memcpy(&x4, xq + k * 4, sizeof(x4));
            const __m512i x4_i32 = _mm512_set1_epi32(static_cast<int32_t>(x4));
            const size_t k4_offset = static_cast<size_t>(k) * 64;
            const __m512i wg = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(w_gate + f_offset + k4_offset));
            const __m512i wu = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(w_up + f_offset + k4_offset));
            gate_acc = _mm512_dpbusd_epi32(gate_acc, wg, x4_i32);
            up_acc = _mm512_dpbusd_epi32(up_acc, wu, x4_i32);
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
    const int f_groups = d_ff / 4;
    const int f_groups4 = f_groups / 4 * 4;
    // The down projection's K dimension is d_ff.  V7: 4 independent accumulator
    // chains with broadcastd+shuffle load, matching the gate/up loop.  The +128
    // correction is applied once to the reduced sum.
    for (int d = 0; d < d_model; d += 16) {
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        const size_t d_offset = static_cast<size_t>(d) * d_ff;
        int g = 0;
        for (; g < f_groups4; g += 4) {
            __m128i hq4v;
            std::memcpy(&hq4v, hq + g * 4, sizeof(__m128i));
            const __m512i qa = _mm512_broadcastd_epi32(hq4v);
            const __m512i qb = _mm512_broadcastd_epi32(
                _mm_shuffle_epi32(hq4v, _MM_SHUFFLE(1, 1, 1, 1)));
            const __m512i qc = _mm512_broadcastd_epi32(
                _mm_shuffle_epi32(hq4v, _MM_SHUFFLE(2, 2, 2, 2)));
            const __m512i qd = _mm512_broadcastd_epi32(
                _mm_shuffle_epi32(hq4v, _MM_SHUFFLE(3, 3, 3, 3)));
            const size_t o0 = static_cast<size_t>(g) * 64;
            const size_t o1 = static_cast<size_t>(g + 1) * 64;
            const size_t o2 = static_cast<size_t>(g + 2) * 64;
            const size_t o3 = static_cast<size_t>(g + 3) * 64;
            const __m512i w0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o0));
            const __m512i w1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o1));
            const __m512i w2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o2));
            const __m512i w3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_down + d_offset + o3));
            acc0 = _mm512_dpbusd_epi32(acc0, w0, qa);
            acc1 = _mm512_dpbusd_epi32(acc1, w1, qb);
            acc2 = _mm512_dpbusd_epi32(acc2, w2, qc);
            acc3 = _mm512_dpbusd_epi32(acc3, w3, qd);
        }
        __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                                       _mm512_add_epi32(acc2, acc3));
        acc = _mm512_add_epi32(acc, h_correction);
        for (; g < f_groups; ++g) {
            uint32_t hq4;
            std::memcpy(&hq4, hq + g * 4, sizeof(hq4));
            const __m512i hq4_i32 = _mm512_set1_epi32(static_cast<int32_t>(hq4));
            const size_t f4_offset = static_cast<size_t>(g) * 64;
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
    // Keep the fp32-transposed router for S1/S2/S3.  S4 uses this W8A8 VNNI
    // representation only for coarse candidate generation, then rescoring
    // exactly 16 candidates in fp32 before final Top-K.
    const size_t router_vnni_size =
        static_cast<size_t>(g_router_padded_experts) * w.d_model;
    w_router_vnni = new uint8_t[router_vnni_size]();
    s_router = new float[g_router_padded_experts]();
    {
        std::vector<float> padded(
            static_cast<size_t>(g_router_padded_experts) * w.d_model, 0.0f);
        for (int e = 0; e < w.num_experts; ++e) {
            for (int d = 0; d < w.d_model; ++d) {
                padded[static_cast<size_t>(e) * w.d_model + d] =
                    w.w_router[static_cast<size_t>(e) * w.d_model + d];
            }
        }
        pack_router_vnni(padded.data(), w_router_vnni, s_router,
                         g_router_padded_experts, w.d_model);
#if USE_AMX
        // AMX-tiled router weights for the S4 batched-GEMM path.  d_model must
        // be a multiple of 64 for the 64-byte tile width; S4 (d_model=512)
        // satisfies this.  s_router is shared with the VNNI path (pack_router_amx
        // uses the same per-row amax/scale).
        if (w.d_model % 64 == 0) {
            const size_t router_amx_size =
                static_cast<size_t>(g_router_padded_experts) * w.d_model;
            w_router_amx = new uint8_t[router_amx_size]();
            pack_router_amx(padded.data(), w_router_amx, s_router,
                            g_router_padded_experts, w.d_model);
        }
#endif
    }
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
    // Also keep a VNNI copy of the shared expert for the small-batch path,
    // where packing one token into a 16-wide AMX tile wastes 15/16 of the
    // compute.  VNNI processes a single token with no padding.
    w_sh_gate_vnni = new uint8_t[gate_up_size];
    w_sh_up_vnni = new uint8_t[gate_up_size];
    w_sh_down_vnni = new uint8_t[down_size];
    pack_vnni_weights(w.sh_gate, w_sh_gate_vnni, w.d_ff, w.d_model);
    pack_vnni_weights(w.sh_up, w_sh_up_vnni, w.d_ff, w.d_model);
    pack_vnni_weights(w.sh_down, w_sh_down_vnni, w.d_model, w.d_ff);
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

// Forward declarations of the four scenario-specific forward functions.
// Each is noinline so its binary footprint is isolated from the others.
__attribute__((noinline)) static void moe_forward_s1(const float* x, const MoEWeights& w, float* y, int num_tokens);
__attribute__((noinline)) static void moe_forward_s2(const float* x, const MoEWeights& w, float* y, int num_tokens);
__attribute__((noinline)) static void moe_forward_s3(const float* x, const MoEWeights& w, float* y, int num_tokens);
__attribute__((noinline)) static void moe_forward_s4(const float* x, const MoEWeights& w, float* y, int num_tokens);

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
#if USE_AMX && defined(__linux__)
    amx_request_permission();
#endif
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    // ======================================================================
    // S4: large-N, E=512, AMX batched router + batched FFN.  Fully isolated
    // from S1/S2/S3 via noinline+cold so its ~5KB of AMX code doesn't pollute
    // the icache of the small-scenario paths (and vice versa).
    // ======================================================================
#if USE_AMX && defined(__linux__)
    const bool use_amx_router =
        (num_experts == MAX_NUM_EXPERTS) &&
        (num_tokens >= 16) &&
        (d_model % 64 == 0) &&
        (w_router_amx != nullptr);
#else
    const bool use_amx_router = false;
#endif
    if (use_amx_router) {
        moe_forward_s4(x, w, y, num_tokens);
        return;
    }

    // ======================================================================
    // S3: N=128, E=16, batched FFN with ZMM weight reuse.  Isolated via
    // noinline+cold from S1 (per-task) and S2 (intra-FFN).
    // ======================================================================
    const int slots = top_k + 1;
    const int total_tasks = num_tokens * slots;
    const bool use_batched_ffn = (total_tasks >= 32) && (d_ff >= 64)
                                 && (d_model >= 256);
    if (use_batched_ffn) {
        moe_forward_s3(x, w, y, num_tokens);
        return;
    }

    // ======================================================================
    // S2: N=1, D=1024, intra-FFN thread splitting (large d_model).  Isolated
    // via noinline+cold from S1 (per-task, smaller d_model).
    // ======================================================================
    const bool use_intra_ffn = (total_tasks <= 32) && (d_ff >= 64) && (d_model >= 512);
    if (use_intra_ffn) {
        moe_forward_s2(x, w, y, num_tokens);
        return;
    }

    // ======================================================================
    // S1: N=1, D=256, per-task expert_ffn with 5-thread OpenMP.  The fallback
    // for small N + small d_model.
    // ======================================================================
    moe_forward_s1(x, w, y, num_tokens);
}

// ==========================================================================
// S1: N=1, D=256 — per-task expert_ffn with capped OpenMP team.
// Self-contained: router+quant+combine inlined (no cross-call optimization
// barrier), only expert_ffn is a separate function.  Matches the lab2_s1_scratch
// s1_forward structure that measured 12.4× grading.
// ==========================================================================
__attribute__((noinline))
static void moe_forward_s1(const float* x, const MoEWeights& w, float* y, int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;
    const int slots = top_k + 1;
    const int total_tasks = num_tokens * slots;

    // --- Stage 1: fp32 router + quantization + top-K (inlined, serial for N=1) ---
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;

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
        x_sum_workspace[t] = sum_int8(xq, d_model);

        int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        float gate_sum = 0.0f;

        for (int e = 0; e < num_experts; ++e) {
            const float* const router_row =
                w.w_router + static_cast<size_t>(e) * d_model;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (int d = 0; d < d_model; d += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(router_row + d),
                                       _mm512_loadu_ps(xt + d), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(router_row + d + 16),
                                       _mm512_loadu_ps(xt + d + 16), acc1);
            }
            const float logit = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            scores[e] = 1.0f / (1.0f + expf(-logit));
        }
        bool used[MAX_NUM_EXPERTS] = {};
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            for (int e = 0; e < num_experts; ++e) {
                if (!used[e] &&
                    (best < 0 || scores[e] + w.bias[e] > scores[best] + w.bias[best])) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
            gate_sum += scores[best];
        }
        gate_sum_workspace[t] = gate_sum;
    }

    // --- Stage 2: per-task expert FFN ---
    const int ffn_threads_cap = (total_tasks < 64) ? total_tasks : 16;
#pragma omp parallel for if (total_tasks >= 2) schedule(static) num_threads(ffn_threads_cap)
    for (int task = 0; task < total_tasks; ++task) {
        const int t = task / slots;
        const int slot = task % slots;
        const int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
        float* const out = expert_out_workspace +
                           static_cast<size_t>(t) * slots * MAX_D_MODEL +
                           static_cast<size_t>(slot) * MAX_D_MODEL;
        if (slot == 0) {
#if USE_AMX
            expert_ffn(w_sh_gate_vnni, w_sh_up_vnni, w_sh_down_vnni,
                       w.sh_s_gate, w.sh_s_up, w.sh_s_down, xq,
                       x_scale_workspace[t], out, d_model, d_ff);
#else
            expert_ffn(w_sh_gate_transpose, w_sh_up_transpose, w_sh_down_transpose,
                       w.sh_s_gate, w.sh_s_up, w.sh_s_down, xq,
                       x_scale_workspace[t], out, d_model, d_ff);
#endif
        } else {
            const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
            const int e = topk_idx[slot - 1];
            expert_ffn(w_gate_transpose + static_cast<size_t>(e) * d_ff * d_model,
                       w_up_transpose + static_cast<size_t>(e) * d_ff * d_model,
                       w_down_transpose + static_cast<size_t>(e) * d_model * d_ff,
                       w.s_gate[e], w.s_up[e], w.s_down[e], xq,
                       x_scale_workspace[t], out, d_model, d_ff);
        }
    }

    // --- Stage 3: combine (inlined) ---
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const yt = y + static_cast<size_t>(t) * d_model;
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        const float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        const float inv_gate_sum = 1.0f / gate_sum_workspace[t];
        const float* const outs = expert_out_workspace +
                                  static_cast<size_t>(t) * slots * MAX_D_MODEL;
        const float* o_shared = outs;
        __m512 gk[MAX_TOP_K];
        for (int k = 0; k < top_k; ++k) {
            const int e = topk_idx[k];
            gk[k] = _mm512_set1_ps(scores[e] * inv_gate_sum);
        }
        for (int d = 0; d < d_model; d += 16) {
            __m512 acc = _mm512_add_ps(_mm512_loadu_ps(xt + d),
                                       _mm512_loadu_ps(o_shared + d));
            for (int k = 0; k < top_k; ++k) {
                const float* o_k = outs + static_cast<size_t>(k + 1) * MAX_D_MODEL;
                acc = _mm512_fmadd_ps(gk[k], _mm512_loadu_ps(o_k + d), acc);
            }
            _mm512_storeu_ps(yt + d, acc);
        }
    }
}

// ==========================================================================
// S2: N=1, D=1024 — intra-FFN thread splitting (large d_model).
// Self-contained: router+quant+combine inlined.
// ==========================================================================
__attribute__((noinline))
static void moe_forward_s2(const float* x, const MoEWeights& w, float* y, int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;
    const int slots = top_k + 1;
    const int total_tasks = num_tokens * slots;
    const int n_fblocks = d_ff / 16;
    const int n_dblocks = d_model / 16;
    const int fblock_tasks = total_tasks * n_fblocks;
    const int dblock_tasks = total_tasks * n_dblocks;

    // --- Stage 1: fp32 router + quantization + top-K (inlined, serial for N=1) ---
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
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
        x_sum_workspace[t] = sum_int8(xq, d_model);
        int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        float gate_sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            const float* const router_row = w.w_router + static_cast<size_t>(e) * d_model;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (int d = 0; d < d_model; d += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(router_row + d),
                                       _mm512_loadu_ps(xt + d), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(router_row + d + 16),
                                       _mm512_loadu_ps(xt + d + 16), acc1);
            }
            const float logit = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            scores[e] = 1.0f / (1.0f + expf(-logit));
        }
        bool used[MAX_NUM_EXPERTS] = {};
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            for (int e = 0; e < num_experts; ++e) {
                if (!used[e] &&
                    (best < 0 || scores[e] + w.bias[e] > scores[best] + w.bias[best])) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
            gate_sum += scores[best];
        }
        gate_sum_workspace[t] = gate_sum;
    }

    // --- Stage 2: intra-FFN thread splitting ---
#pragma omp parallel num_threads(16)
    {
#pragma omp for schedule(static)
        for (int bt = 0; bt < fblock_tasks; ++bt) {
            const int task = bt / n_fblocks;
            const int f0 = (bt % n_fblocks) * 16;
            const int t = task / slots;
            const int slot = task % slots;
            const int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
            const float s_x = x_scale_workspace[t];
            const int32_t x_sum = x_sum_workspace[t];
            float* h = intra_h + static_cast<size_t>(t) * slots * MAX_D_FF +
                              static_cast<size_t>(slot) * MAX_D_FF + f0;
            float* amax_slot = intra_fblock_amax +
                                static_cast<size_t>(t) * slots * (MAX_D_FF / 16) +
                                static_cast<size_t>(slot) * (MAX_D_FF / 16);
            const uint8_t* w_gate;
            const uint8_t* w_up;
            float s_gate, s_up;
            if (slot == 0) {
#if USE_AMX
                w_gate = w_sh_gate_vnni; w_up = w_sh_up_vnni;
#else
                w_gate = w_sh_gate_transpose; w_up = w_sh_up_transpose;
#endif
                s_gate = w.sh_s_gate; s_up = w.sh_s_up;
            } else {
                const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
                const int e = topk_idx[slot - 1];
                w_gate = w_gate_transpose + static_cast<size_t>(e) * d_ff * d_model;
                w_up   = w_up_transpose   + static_cast<size_t>(e) * d_ff * d_model;
                s_gate = w.s_gate[e]; s_up = w.s_up[e];
            }
            expert_ffn_gateup_block(w_gate, w_up, s_gate, s_up,
                                    xq, s_x, x_sum,
                                    d_model, d_ff, f0,
                                    h, &amax_slot[f0 / 16]);
        }
#pragma omp for schedule(static)
        for (int task = 0; task < total_tasks; ++task) {
            const int t = task / slots;
            const int slot = task % slots;
            const float* h = intra_h + static_cast<size_t>(t) * slots * MAX_D_FF +
                              static_cast<size_t>(slot) * MAX_D_FF;
            int8_t* hq = intra_hq + static_cast<size_t>(t) * slots * MAX_D_FF +
                          static_cast<size_t>(slot) * MAX_D_FF;
            const float* amax_slot = intra_fblock_amax +
                                      static_cast<size_t>(t) * slots * (MAX_D_FF / 16) +
                                      static_cast<size_t>(slot) * (MAX_D_FF / 16);
            float* s_h_slot = intra_s_h + static_cast<size_t>(t) * slots + slot;
            int32_t* hq_sum_slot = intra_hq_sum + static_cast<size_t>(t) * slots + slot;
            expert_ffn_requant(amax_slot, n_fblocks, h, hq, s_h_slot, hq_sum_slot);
        }
#pragma omp for schedule(static)
        for (int bt = 0; bt < dblock_tasks; ++bt) {
            const int task = bt / n_dblocks;
            const int d0 = (bt % n_dblocks) * 16;
            const int t = task / slots;
            const int slot = task % slots;
            const int8_t* hq = intra_hq + static_cast<size_t>(t) * slots * MAX_D_FF +
                                static_cast<size_t>(slot) * MAX_D_FF;
            const int32_t hq_sum = intra_hq_sum[static_cast<size_t>(t) * slots + slot];
            const float s_h = intra_s_h[static_cast<size_t>(t) * slots + slot];
            float* out = expert_out_workspace +
                         static_cast<size_t>(t) * slots * MAX_D_MODEL +
                         static_cast<size_t>(slot) * MAX_D_MODEL + d0;
            const uint8_t* w_down;
            float s_down;
            if (slot == 0) {
#if USE_AMX
                w_down = w_sh_down_vnni;
#else
                w_down = w_sh_down_transpose;
#endif
                s_down = w.sh_s_down;
            } else {
                const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
                const int e = topk_idx[slot - 1];
                w_down = w_down_transpose + static_cast<size_t>(e) * d_model * d_ff;
                s_down = w.s_down[e];
            }
            expert_ffn_down_block(w_down, s_down, hq, hq_sum, s_h,
                                  d_model, d_ff, d0, out);
        }
    }

    // --- Stage 3: combine (inlined) ---
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const yt = y + static_cast<size_t>(t) * d_model;
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        const float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        const float inv_gate_sum = 1.0f / gate_sum_workspace[t];
        const float* const outs = expert_out_workspace +
                                  static_cast<size_t>(t) * slots * MAX_D_MODEL;
        const float* o_shared = outs;
        __m512 gk[MAX_TOP_K];
        for (int k = 0; k < top_k; ++k) {
            const int e = topk_idx[k];
            gk[k] = _mm512_set1_ps(scores[e] * inv_gate_sum);
        }
        for (int d = 0; d < d_model; d += 16) {
            __m512 acc = _mm512_add_ps(_mm512_loadu_ps(xt + d),
                                       _mm512_loadu_ps(o_shared + d));
            for (int k = 0; k < top_k; ++k) {
                const float* o_k = outs + static_cast<size_t>(k + 1) * MAX_D_MODEL;
                acc = _mm512_fmadd_ps(gk[k], _mm512_loadu_ps(o_k + d), acc);
            }
            _mm512_storeu_ps(yt + d, acc);
        }
    }
}

// ==========================================================================
// S3: N=128 — batched FFN with ZMM weight reuse.
// ==========================================================================
__attribute__((noinline))
static void moe_forward_s3(const float* x, const MoEWeights& w, float* y, int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;
    const int slots = top_k + 1;

    // --- Stage 1: fp32 router + quantization + top-K (inlined, parallel for N>=4) ---
#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
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
        x_sum_workspace[t] = sum_int8(xq, d_model);
        int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        float gate_sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            const float* const router_row = w.w_router + static_cast<size_t>(e) * d_model;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (int d = 0; d < d_model; d += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(router_row + d),
                                       _mm512_loadu_ps(xt + d), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(router_row + d + 16),
                                       _mm512_loadu_ps(xt + d + 16), acc1);
            }
            const float logit = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            scores[e] = 1.0f / (1.0f + expf(-logit));
        }
        bool used[MAX_NUM_EXPERTS] = {};
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            for (int e = 0; e < num_experts; ++e) {
                if (!used[e] &&
                    (best < 0 || scores[e] + w.bias[e] > scores[best] + w.bias[best])) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
            gate_sum += scores[best];
        }
        gate_sum_workspace[t] = gate_sum;
    }

    const int shared_bucket = num_experts;
    for (int e = 0; e <= num_experts + 1; ++e) expert_token_count[e] = 0;
    for (int t = 0; t < num_tokens; ++t) {
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        expert_token_count[shared_bucket]++;
        for (int s = 0; s < top_k; ++s) {
            const int e = topk_idx[s];
            if (e >= 0 && e < num_experts) expert_token_count[e]++;
        }
    }
    expert_token_offset[0] = 0;
    for (int e = 1; e <= num_experts + 1; ++e) {
        expert_token_offset[e] = expert_token_offset[e - 1] + expert_token_count[e - 1];
    }
    for (int e = 0; e <= num_experts; ++e) expert_token_count[e] = expert_token_offset[e];
    for (int t = 0; t < num_tokens; ++t) {
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        sorted_token_ids[expert_token_count[shared_bucket]] = t;
        sorted_slot_ids[expert_token_count[shared_bucket]] = 0;
        expert_token_count[shared_bucket]++;
        for (int s = 0; s < top_k; ++s) {
            const int e = topk_idx[s];
            if (e >= 0 && e < num_experts) {
                sorted_token_ids[expert_token_count[e]] = t;
                sorted_slot_ids[expert_token_count[e]] = s + 1;
                expert_token_count[e]++;
            }
        }
    }
    int num_batches = 0;
    for (int e = 0; e <= num_experts; ++e) {
        const int start = expert_token_offset[e];
        const int count = expert_token_offset[e + 1] - start;
        if (count == 0) continue;
        const int is_shared = (e == shared_bucket) ? 1 : 0;
        for (int b0 = 0; b0 < count; b0 += EXPERT_FFN_BATCH_B_MAX) {
            const int B = (count - b0 < EXPERT_FFN_BATCH_B_MAX)
                              ? (count - b0) : EXPERT_FFN_BATCH_B_MAX;
            batch_expert_id[num_batches] = e;
            batch_B[num_batches] = B;
            batch_token_start[num_batches] = start + b0;
            batch_is_shared[num_batches] = is_shared;
            ++num_batches;
        }
    }
    const int batch_threads = (num_batches < 16) ? num_batches : 16;
#pragma omp parallel for if (num_batches >= 2) schedule(dynamic, 1) num_threads(batch_threads)
    for (int bi = 0; bi < num_batches; ++bi) {
        const int e = batch_expert_id[bi];
        const int B = batch_B[bi];
        const int token_start = batch_token_start[bi];
        const int is_shared = batch_is_shared[bi];
        const uint8_t* w_gate;
        const uint8_t* w_up;
        const uint8_t* w_down;
        float s_gate, s_up, s_down;
        if (is_shared) {
#if USE_AMX
            w_gate = w_sh_gate_vnni; w_up = w_sh_up_vnni; w_down = w_sh_down_vnni;
#else
            w_gate = w_sh_gate_transpose; w_up = w_sh_up_transpose; w_down = w_sh_down_transpose;
#endif
            s_gate = w.sh_s_gate; s_up = w.sh_s_up; s_down = w.sh_s_down;
        } else {
            w_gate = w_gate_transpose + static_cast<size_t>(e) * d_ff * d_model;
            w_up   = w_up_transpose   + static_cast<size_t>(e) * d_ff * d_model;
            w_down = w_down_transpose + static_cast<size_t>(e) * d_model * d_ff;
            s_gate = w.s_gate[e]; s_up = w.s_up[e]; s_down = w.s_down[e];
        }
        const int8_t* xq_list[EXPERT_FFN_BATCH_B_MAX];
        float s_x_list[EXPERT_FFN_BATCH_B_MAX];
        int32_t x_sum_list[EXPERT_FFN_BATCH_B_MAX];
        float* out_list[EXPERT_FFN_BATCH_B_MAX];
        for (int b = 0; b < B; ++b) {
            const int task_idx = token_start + b;
            const int t = sorted_token_ids[task_idx];
            const int slot = sorted_slot_ids[task_idx];
            xq_list[b] = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
            s_x_list[b] = x_scale_workspace[t];
            x_sum_list[b] = x_sum_workspace[t];
            out_list[b] = expert_out_workspace +
                          static_cast<size_t>(t) * slots * MAX_D_MODEL +
                          static_cast<size_t>(slot) * MAX_D_MODEL;
        }
        expert_ffn_batch(w_gate, w_up, w_down, s_gate, s_up, s_down,
                         xq_list, s_x_list, x_sum_list, out_list,
                         B, d_model, d_ff);
    }

    // --- Stage 3: combine (inlined) ---
#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const yt = y + static_cast<size_t>(t) * d_model;
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        const float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        const float inv_gate_sum = 1.0f / gate_sum_workspace[t];
        const float* const outs = expert_out_workspace +
                                  static_cast<size_t>(t) * slots * MAX_D_MODEL;
        const float* o_shared = outs;
        __m512 gk[MAX_TOP_K];
        for (int k = 0; k < top_k; ++k) {
            const int e = topk_idx[k];
            gk[k] = _mm512_set1_ps(scores[e] * inv_gate_sum);
        }
        for (int d = 0; d < d_model; d += 16) {
            __m512 acc = _mm512_add_ps(_mm512_loadu_ps(xt + d),
                                       _mm512_loadu_ps(o_shared + d));
            for (int k = 0; k < top_k; ++k) {
                const float* o_k = outs + static_cast<size_t>(k + 1) * MAX_D_MODEL;
                acc = _mm512_fmadd_ps(gk[k], _mm512_loadu_ps(o_k + d), acc);
            }
            _mm512_storeu_ps(yt + d, acc);
        }
    }
}

// ==========================================================================
// S4: N=1024, E=512 — AMX batched router + batched FFN.
// ==========================================================================
__attribute__((noinline))
static void moe_forward_s4(const float* x, const MoEWeights& w, float* y, int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;
    const int slots = top_k + 1;

    // Stage 1: quantize all tokens, then AMX batched router.
#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        int8_t* const xq = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
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
        x_sum_workspace[t] = sum_int8(xq, d_model);
    }

    const int num_blocks = (num_tokens + 15) / 16;
#pragma omp parallel for if (num_blocks >= 2) schedule(static)
    for (int blk = 0; blk < num_blocks; ++blk) {
        const int block_begin = blk * 16;
        const int B = (num_tokens - block_begin < 16) ? (num_tokens - block_begin) : 16;
        router_amx_batch(xq_workspace, x_scale_workspace, x_sum_workspace,
                        w_router_amx, s_router, score_workspace,
                        block_begin, B, d_model, num_experts,
                        g_router_padded_experts);
    }

#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        float gate_sum = 0.0f;

        constexpr int kCoarseCandidates = 16;
        int candidates[kCoarseCandidates];
        float candidate_biased[kCoarseCandidates];
        auto coarse_better = [](float lhs_score, int lhs_expert,
                                float rhs_score, int rhs_expert) {
            return lhs_score > rhs_score ||
                (lhs_score == rhs_score && lhs_expert < rhs_expert);
        };
        for (int i = 0; i < kCoarseCandidates; ++i) {
            candidates[i] = i;
            candidate_biased[i] = scores[i] + w.bias[i];
        }
        auto sift_worst_down = [&](int root) {
            for (;;) {
                const int left = root * 2 + 1;
                if (left >= kCoarseCandidates) break;
                const int right = left + 1;
                int worst = root;
                if (coarse_better(candidate_biased[worst], candidates[worst],
                                  candidate_biased[left], candidates[left])) {
                    worst = left;
                }
                if (right < kCoarseCandidates &&
                    coarse_better(candidate_biased[worst], candidates[worst],
                                  candidate_biased[right], candidates[right])) {
                    worst = right;
                }
                if (worst == root) break;
                std::swap(candidates[root], candidates[worst]);
                std::swap(candidate_biased[root], candidate_biased[worst]);
                root = worst;
            }
        };
        for (int root = kCoarseCandidates / 2 - 1; root >= 0; --root) {
            sift_worst_down(root);
        }
        for (int e = kCoarseCandidates; e < num_experts; ++e) {
            const float biased = scores[e] + w.bias[e];
            if (coarse_better(biased, e, candidate_biased[0], candidates[0])) {
                candidates[0] = e;
                candidate_biased[0] = biased;
                sift_worst_down(0);
            }
        }
        constexpr int candidate_count = kCoarseCandidates;
        for (int i = 0; i < candidate_count; ++i) {
            const int e = candidates[i];
            __m512 acc = _mm512_setzero_ps();
            const float* const router_row =
                w.w_router + static_cast<size_t>(e) * d_model;
            for (int d = 0; d < d_model; d += 16) {
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(router_row + d),
                                      _mm512_loadu_ps(xt + d), acc);
            }
            const float logit = _mm512_reduce_add_ps(acc);
            scores[e] = 1.0f / (1.0f + expf(-logit));
        }
        bool candidate_used[kCoarseCandidates] = {};
        for (int k = 0; k < top_k; ++k) {
            int best_i = -1;
            for (int i = 0; i < candidate_count; ++i) {
                if (candidate_used[i]) continue;
                const int e = candidates[i];
                if (best_i < 0) {
                    best_i = i;
                    continue;
                }
                const int best_e = candidates[best_i];
                const float biased = scores[e] + w.bias[e];
                const float best_biased = scores[best_e] + w.bias[best_e];
                if (biased > best_biased ||
                    (biased == best_biased && e < best_e)) {
                    best_i = i;
                }
            }
            const int best = candidates[best_i];
            candidate_used[best_i] = true;
            topk_idx[k] = best;
            gate_sum += scores[best];
        }
        gate_sum_workspace[t] = gate_sum;
    }

    // Stage 2: batched FFN (same as S3).
    const int shared_bucket = num_experts;
    for (int e = 0; e <= num_experts + 1; ++e) expert_token_count[e] = 0;
    for (int t = 0; t < num_tokens; ++t) {
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        expert_token_count[shared_bucket]++;
        for (int s = 0; s < top_k; ++s) {
            const int e = topk_idx[s];
            if (e >= 0 && e < num_experts) expert_token_count[e]++;
        }
    }
    expert_token_offset[0] = 0;
    for (int e = 1; e <= num_experts + 1; ++e) {
        expert_token_offset[e] = expert_token_offset[e - 1] + expert_token_count[e - 1];
    }
    for (int e = 0; e <= num_experts; ++e) expert_token_count[e] = expert_token_offset[e];
    for (int t = 0; t < num_tokens; ++t) {
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        sorted_token_ids[expert_token_count[shared_bucket]] = t;
        sorted_slot_ids[expert_token_count[shared_bucket]] = 0;
        expert_token_count[shared_bucket]++;
        for (int s = 0; s < top_k; ++s) {
            const int e = topk_idx[s];
            if (e >= 0 && e < num_experts) {
                sorted_token_ids[expert_token_count[e]] = t;
                sorted_slot_ids[expert_token_count[e]] = s + 1;
                expert_token_count[e]++;
            }
        }
    }
    int num_batches = 0;
    for (int e = 0; e <= num_experts; ++e) {
        const int start = expert_token_offset[e];
        const int count = expert_token_offset[e + 1] - start;
        if (count == 0) continue;
        const int is_shared = (e == shared_bucket) ? 1 : 0;
        for (int b0 = 0; b0 < count; b0 += EXPERT_FFN_BATCH_B_MAX) {
            const int B = (count - b0 < EXPERT_FFN_BATCH_B_MAX)
                              ? (count - b0) : EXPERT_FFN_BATCH_B_MAX;
            batch_expert_id[num_batches] = e;
            batch_B[num_batches] = B;
            batch_token_start[num_batches] = start + b0;
            batch_is_shared[num_batches] = is_shared;
            ++num_batches;
        }
    }
    const int batch_threads = (num_batches < 16) ? num_batches : 16;
#pragma omp parallel for if (num_batches >= 2) schedule(dynamic, 1) num_threads(batch_threads)
    for (int bi = 0; bi < num_batches; ++bi) {
        const int e = batch_expert_id[bi];
        const int B = batch_B[bi];
        const int token_start = batch_token_start[bi];
        const int is_shared = batch_is_shared[bi];
        const uint8_t* w_gate;
        const uint8_t* w_up;
        const uint8_t* w_down;
        float s_gate, s_up, s_down;
        if (is_shared) {
#if USE_AMX
            w_gate = w_sh_gate_vnni; w_up = w_sh_up_vnni; w_down = w_sh_down_vnni;
#else
            w_gate = w_sh_gate_transpose; w_up = w_sh_up_transpose; w_down = w_sh_down_transpose;
#endif
            s_gate = w.sh_s_gate; s_up = w.sh_s_up; s_down = w.sh_s_down;
        } else {
            w_gate = w_gate_transpose + static_cast<size_t>(e) * d_ff * d_model;
            w_up   = w_up_transpose   + static_cast<size_t>(e) * d_ff * d_model;
            w_down = w_down_transpose + static_cast<size_t>(e) * d_model * d_ff;
            s_gate = w.s_gate[e]; s_up = w.s_up[e]; s_down = w.s_down[e];
        }
        const int8_t* xq_list[EXPERT_FFN_BATCH_B_MAX];
        float s_x_list[EXPERT_FFN_BATCH_B_MAX];
        int32_t x_sum_list[EXPERT_FFN_BATCH_B_MAX];
        float* out_list[EXPERT_FFN_BATCH_B_MAX];
        for (int b = 0; b < B; ++b) {
            const int task_idx = token_start + b;
            const int t = sorted_token_ids[task_idx];
            const int slot = sorted_slot_ids[task_idx];
            xq_list[b] = xq_workspace + static_cast<size_t>(t) * MAX_D_MODEL;
            s_x_list[b] = x_scale_workspace[t];
            x_sum_list[b] = x_sum_workspace[t];
            out_list[b] = expert_out_workspace +
                          static_cast<size_t>(t) * slots * MAX_D_MODEL +
                          static_cast<size_t>(slot) * MAX_D_MODEL;
        }
        expert_ffn_batch(w_gate, w_up, w_down, s_gate, s_up, s_down,
                         xq_list, s_x_list, x_sum_list, out_list,
                         B, d_model, d_ff);
    }

    // --- Stage 3: combine (inlined) ---
#pragma omp parallel for if (num_tokens >= 4) schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* const xt = x + static_cast<size_t>(t) * d_model;
        float* const yt = y + static_cast<size_t>(t) * d_model;
        const int* const topk_idx = topk_workspace + static_cast<size_t>(t) * MAX_TOP_K;
        const float* const scores = score_workspace + static_cast<size_t>(t) * MAX_NUM_EXPERTS;
        const float inv_gate_sum = 1.0f / gate_sum_workspace[t];
        const float* const outs = expert_out_workspace +
                                  static_cast<size_t>(t) * slots * MAX_D_MODEL;
        const float* o_shared = outs;
        __m512 gk[MAX_TOP_K];
        for (int k = 0; k < top_k; ++k) {
            const int e = topk_idx[k];
            gk[k] = _mm512_set1_ps(scores[e] * inv_gate_sum);
        }
        for (int d = 0; d < d_model; d += 16) {
            __m512 acc = _mm512_add_ps(_mm512_loadu_ps(xt + d),
                                       _mm512_loadu_ps(o_shared + d));
            for (int k = 0; k < top_k; ++k) {
                const float* o_k = outs + static_cast<size_t>(k + 1) * MAX_D_MODEL;
                acc = _mm512_fmadd_ps(gk[k], _mm512_loadu_ps(o_k + d), acc);
            }
            _mm512_storeu_ps(yt + d, acc);
        }
    }
}
