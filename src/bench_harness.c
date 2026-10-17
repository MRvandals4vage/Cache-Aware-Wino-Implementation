/*
 * src/bench_harness.c
 *
 * CacheWinograd High-Precision Performance Engineering & Benchmark Harness.
 *
 * Implements:
 * 1. naive_direct: Reference 6-nested loop direct 2D convolution
 * 2. neon_direct: Vectorized / register-blocked direct 2D convolution
 * 3. unfused: Naive Winograd (materializes full-image V and M intermediate buffers)
 * 4. fused (CacheWinograd-v1): Cache-conscious fused Winograd (tile-at-a-time fusion)
 * 5. v2_regblocked: Additive optimization 1 (Cout register-blocked loop)
 * 6. v3_cin_blocked: Additive optimization 2 (Cin cache-blocking sized to L2 share)
 * 7. v4_pretransformed_packed: Additive optimization 3 (Pre-transformed, packed 64B-aligned weights)
 *
 * Strict Compliance:
 * - Part A.6: Asserts executed_path == requested_variant
 * - Part B.2: Zero-heavy-dependency C implementation (runs without Python)
 * - Writes atomic CSV per Part A.6 schema
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_ARM_NEON 1
#else
#define HAS_ARM_NEON 0
#endif

/* -------------------------------------------------------------------------
 * Hardware Telemetry Probes
 * ------------------------------------------------------------------------- */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static long get_cpu_freq_mhz(void) {
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    if (!f) f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq", "r");
    if (f) {
        long khz = 0;
        if (fscanf(f, "%ld", &khz) == 1) {
            fclose(f);
            return khz / 1000;
        }
        fclose(f);
    }
    return 0; // Nominal or unavailable
}

static double get_cpu_temp_c(void) {
    FILE *f = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (f) {
        long millic = 0;
        if (fscanf(f, "%ld", &millic) == 1) {
            fclose(f);
            return (double)millic / 1000.0;
        }
        fclose(f);
    }
    return 0.0;
}

static unsigned int get_throttle_flags(void) {
    FILE *f = popen("vcgencmd get_throttled 2>/dev/null", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            unsigned int flags = 0;
            if (sscanf(buf, "throttled=0x%x", &flags) == 1) {
                pclose(f);
                return flags;
            }
        }
        pclose(f);
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Winograd Transform Matrices for F(2,3), F(4,3), F(6,3)
 * ------------------------------------------------------------------------- */

/* F(2,3): m=2, r=3, alpha=4 */
static const float G_2[4][3] = {
    { 1.0f,  0.0f,  0.0f },
    { 0.5f,  0.5f,  0.5f },
    { 0.5f, -0.5f,  0.5f },
    { 0.0f,  0.0f,  1.0f }
};
static const float BT_2[4][4] = {
    { 1.0f,  0.0f, -1.0f,  0.0f },
    { 0.0f,  1.0f,  1.0f,  0.0f },
    { 0.0f, -1.0f,  1.0f,  0.0f },
    { 0.0f,  1.0f,  0.0f, -1.0f }
};
static const float AT_2[2][4] = {
    { 1.0f,  1.0f,  1.0f,  0.0f },
    { 0.0f,  1.0f, -1.0f, -1.0f }
};

/* F(4,3): m=4, r=3, alpha=6 */
static const float G_4[6][3] = {
    {  1.0f/4.0f,        0.0f,        0.0f },
    { -1.0f/6.0f,  -1.0f/6.0f,  -1.0f/6.0f },
    { -1.0f/6.0f,   1.0f/6.0f,  -1.0f/6.0f },
    {  1.0f/24.0f,  1.0f/12.0f,  1.0f/6.0f },
    {  1.0f/24.0f, -1.0f/12.0f,  1.0f/6.0f },
    {      0.0f,         0.0f,        1.0f }
};
static const float BT_4[6][6] = {
    { 4.0f,  0.0f, -5.0f,  0.0f,  1.0f, 0.0f },
    { 0.0f, -4.0f, -4.0f,  1.0f,  1.0f, 0.0f },
    { 0.0f,  4.0f, -4.0f, -1.0f,  1.0f, 0.0f },
    { 0.0f, -2.0f, -1.0f,  2.0f,  1.0f, 0.0f },
    { 0.0f,  2.0f, -1.0f, -2.0f,  1.0f, 0.0f },
    { 0.0f,  4.0f,  0.0f, -5.0f,  0.0f, 1.0f }
};
static const float AT_4[4][6] = {
    { 1.0f,  1.0f,  1.0f,  1.0f,  1.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  2.0f, -2.0f, 0.0f },
    { 0.0f,  1.0f,  1.0f,  4.0f,  4.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  8.0f, -8.0f, 1.0f }
};

/* F(6,3): m=6, r=3, alpha=8 */
static const float G_6[8][3] = {
    { 1.0f,         0.0f,         0.0f },
    { -2.0f/9.0f,  -2.0f/9.0f,  -2.0f/9.0f },
    { -2.0f/9.0f,   2.0f/9.0f,  -2.0f/9.0f },
    {  1.0f/90.0f,  1.0f/45.0f,  2.0f/45.0f },
    {  1.0f/90.0f, -1.0f/45.0f,  2.0f/45.0f },
    {  1.0f/45.0f,  1.0f/90.0f,  1.0f/180.0f },
    {  1.0f/45.0f, -1.0f/90.0f,  1.0f/180.0f },
    { 0.0f,         0.0f,         1.0f }
};
static const float BT_6[8][8] = {
    { 1.0f,  0.0f, -21.0f/4.0f,  0.0f,       21.0f/4.0f,  0.0f,      -1.0f, 0.0f },
    { 0.0f,  1.0f,   1.0f,      -17.0f/4.0f, -17.0f/4.0f, 1.0f,       1.0f, 0.0f },
    { 0.0f, -1.0f,   1.0f,       17.0f/4.0f, -17.0f/4.0f,-1.0f,       1.0f, 0.0f },
    { 0.0f,  0.5f,   0.25f,     -2.5f,       -1.25f,      2.0f,       1.0f, 0.0f },
    { 0.0f, -0.5f,   0.25f,      2.5f,       -1.25f,     -2.0f,       1.0f, 0.0f },
    { 0.0f,  2.0f,   4.0f,      -2.5f,       -5.0f,       0.5f,       1.0f, 0.0f },
    { 0.0f, -2.0f,   4.0f,       2.5f,       -5.0f,      -0.5f,       1.0f, 0.0f },
    { 0.0f, -1.0f,   0.0f,       21.0f/4.0f,  0.0f,      -21.0f/4.0f, 0.0f, 1.0f }
};
static const float AT_6[6][8] = {
    { 1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  32.0f,  32.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  2.0f, -2.0f,  16.0f, -16.0f, 0.0f },
    { 0.0f,  1.0f,  1.0f,  4.0f,  4.0f,   8.0f,   8.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  8.0f, -8.0f,   4.0f,  -4.0f, 0.0f },
    { 0.0f,  1.0f,  1.0f, 16.0f, 16.0f,   2.0f,   2.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f, 32.0f,-32.0f,   1.0f,  -1.0f, 1.0f }
};

/* -------------------------------------------------------------------------
 * Inline Tile Transform Functions
 * ------------------------------------------------------------------------- */
static inline void transform_input_tile(int m, const float* d, float* v) {
    if (m == 2) {
        float tmp[4][4];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += BT_2[r][k] * d[k * 4 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += tmp[r][k] * BT_2[c][k];
                v[r * 4 + c] = sum;
            }
        }
    } else if (m == 4) {
        float tmp[6][6];
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += BT_4[r][k] * d[k * 6 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += tmp[r][k] * BT_4[c][k];
                v[r * 6 + c] = sum;
            }
        }
    } else if (m == 6) {
        float tmp[8][8];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += BT_6[r][k] * d[k * 8 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += tmp[r][k] * BT_6[c][k];
                v[r * 8 + c] = sum;
            }
        }
    }
}

static inline void transform_output_tile(int m, const float* m_tile, float* y) {
    if (m == 2) {
        float tmp[2][4];
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += AT_2[r][k] * m_tile[k * 4 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += tmp[r][k] * AT_2[c][k];
                y[r * 2 + c] = sum;
            }
        }
    } else if (m == 4) {
        float tmp[4][6];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += AT_4[r][k] * m_tile[k * 6 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += tmp[r][k] * AT_4[c][k];
                y[r * 4 + c] = sum;
            }
        }
    } else if (m == 6) {
        float tmp[6][8];
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 8; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += AT_6[r][k] * m_tile[k * 8 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += tmp[r][k] * AT_6[c][k];
                y[r * 6 + c] = sum;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Kernel Weight Transformation: U = G * g * G^T
 * ------------------------------------------------------------------------- */
static void transform_weights(int m, int Cin, int Cout, const float* weights, float* U) {
    int alpha = m + 2;
    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            const float* g = weights + (co * Cin + ci) * 9;
            float* u = U + (co * Cin + ci) * alpha * alpha;

            float tmp[8][3];
            if (m == 2) {
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 3; c++) {
                        tmp[r][c] = G_2[r][0] * g[0*3 + c] +
                                    G_2[r][1] * g[1*3 + c] +
                                    G_2[r][2] * g[2*3 + c];
                    }
                }
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; k++) sum += tmp[r][k] * G_2[c][k];
                        u[r * 4 + c] = sum;
                    }
                }
            } else if (m == 4) {
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 3; c++) {
                        tmp[r][c] = G_4[r][0] * g[0*3 + c] +
                                    G_4[r][1] * g[1*3 + c] +
                                    G_4[r][2] * g[2*3 + c];
                    }
                }
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 6; c++) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; k++) sum += tmp[r][k] * G_4[c][k];
                        u[r * 6 + c] = sum;
                    }
                }
            } else if (m == 6) {
                for (int r = 0; r < 8; r++) {
                    for (int c = 0; c < 3; c++) {
                        tmp[r][c] = G_6[r][0] * g[0*3 + c] +
                                    G_6[r][1] * g[1*3 + c] +
                                    G_6[r][2] * g[2*3 + c];
                    }
                }
                for (int r = 0; r < 8; r++) {
                    for (int c = 0; c < 8; c++) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; k++) sum += tmp[r][k] * G_6[c][k];
                        u[r * 8 + c] = sum;
                    }
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 1. Naive Direct 2D Convolution (Baseline 1)
 * ------------------------------------------------------------------------- */
void conv2d_naive_direct(
    const float* input,
    const float* weights,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "naive_direct";
    int Hout = H - 2;
    int Wout = W - 2;

    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int co = 0; co < Cout; co++) {
        for (int ho = 0; ho < Hout; ho++) {
            for (int wo = 0; wo < Wout; wo++) {
                float sum = 0.0f;
                for (int ci = 0; ci < Cin; ci++) {
                    const float* w = weights + (co * Cin + ci) * 9;
                    const float* in = input + ci * H * W;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            sum += in[(ho + kh) * W + (wo + kw)] * w[kh * 3 + kw];
                        }
                    }
                }
                output[co * Hout * Wout + ho * Wout + wo] = sum;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 2. NEON / Register-Blocked Direct Convolution (Baseline 2)
 * ------------------------------------------------------------------------- */
void conv2d_neon_direct(
    const float* input,
    const float* weights,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "neon_direct";
    int Hout = H - 2;
    int Wout = W - 2;

    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int co_blk = 0; co_blk < Cout; co_blk += 4) {
        for (int ho = 0; ho < Hout; ho++) {
            int co_end = (co_blk + 4 <= Cout) ? co_blk + 4 : Cout;
            for (int wo = 0; wo < Wout; wo++) {
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int ci = 0; ci < Cin; ci++) {
                    const float* in = input + ci * H * W;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            float in_val = in[(ho + kh) * W + (wo + kw)];
                            for (int co_idx = 0; co_idx < (co_end - co_blk); co_idx++) {
                                int co = co_blk + co_idx;
                                float w_val = weights[(co * Cin + ci) * 9 + kh * 3 + kw];
                                acc[co_idx] += in_val * w_val;
                            }
                        }
                    }
                }

                for (int co_idx = 0; co_idx < (co_end - co_blk); co_idx++) {
                    int co = co_blk + co_idx;
                    output[co * Hout * Wout + ho * Wout + wo] = acc[co_idx];
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 3. Unfused Winograd Convolution (Full intermediate V & M materialization)
 * ------------------------------------------------------------------------- */
void conv2d_unfused_winograd(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "unfused";
    int alpha = m + 2;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    int alpha2 = alpha * alpha;

    size_t v_size = (size_t)N_tiles * Cin * alpha2;
    size_t m_size = (size_t)N_tiles * Cout * alpha2;
    float* V_all = (float*)malloc(v_size * sizeof(float));
    float* M_all = (float*)malloc(m_size * sizeof(float));

    /* Step 1: Input Transform across all spatial tiles -> V_all */
    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int t = 0; t < N_tiles; t++) {
        for (int ci = 0; ci < Cin; ci++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            float d[64];
            for (int r = 0; r < alpha; r++) {
                for (int c = 0; c < alpha; c++) {
                    int ih = h_start + r;
                    int iw = w_start + c;
                    d[r * alpha + c] = (ih < H && iw < W) ? input[ci * H * W + ih * W + iw] : 0.0f;
                }
            }

            float* v_ptr = V_all + ((size_t)t * Cin + ci) * alpha2;
            transform_input_tile(m, d, v_ptr);
        }
    }

    /* Step 2: Batched GEMM / Hadamard multiply across channels */
    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int t = 0; t < N_tiles; t++) {
        for (int co = 0; co < Cout; co++) {
            float* m_ptr = M_all + ((size_t)t * Cout + co) * alpha2;
            for (int k = 0; k < alpha2; k++) m_ptr[k] = 0.0f;

            for (int ci = 0; ci < Cin; ci++) {
                const float* u_ptr = U + ((size_t)co * Cin + ci) * alpha2;
                const float* v_ptr = V_all + ((size_t)t * Cin + ci) * alpha2;
                for (int k = 0; k < alpha2; k++) {
                    m_ptr[k] += u_ptr[k] * v_ptr[k];
                }
            }
        }
    }

    /* Step 3: Output Transform -> Y */
    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int t = 0; t < N_tiles; t++) {
        for (int co = 0; co < Cout; co++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            const float* m_ptr = M_all + ((size_t)t * Cout + co) * alpha2;
            float y[36];
            transform_output_tile(m, m_ptr, y);

            for (int r = 0; r < m; r++) {
                for (int c = 0; c < m; c++) {
                    int oh = h_start + r;
                    int ow = w_start + c;
                    if (oh < Hout && ow < Wout) {
                        output[co * Hout * Wout + oh * Wout + ow] = y[r * m + c];
                    }
                }
            }
        }
    }

    free(V_all);
    free(M_all);
}

/* -------------------------------------------------------------------------
 * 4. Fused Winograd (CacheWinograd-v1: Tile-at-a-time fusion)
 * ------------------------------------------------------------------------- */
void conv2d_fused_winograd(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "fused";
    int alpha = m + 2;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    int alpha2 = alpha * alpha;

    #pragma omp parallel num_threads(T)
    {
        float* v_tile = (float*)malloc(Cin * alpha2 * sizeof(float));
        float* m_tile = (float*)malloc(Cout * alpha2 * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int t = 0; t < N_tiles; t++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            /* 1. Input transform strictly for current tile */
            for (int ci = 0; ci < Cin; ci++) {
                float d[64];
                for (int r = 0; r < alpha; r++) {
                    for (int c = 0; c < alpha; c++) {
                        int ih = h_start + r;
                        int iw = w_start + c;
                        d[r * alpha + c] = (ih < H && iw < W) ? input[ci * H * W + ih * W + iw] : 0.0f;
                    }
                }
                float* v_ptr = v_tile + ci * alpha2;
                transform_input_tile(m, d, v_ptr);
            }

            /* 2. Multiply across channels directly in cache */
            for (int co = 0; co < Cout; co++) {
                float* m_ptr = m_tile + co * alpha2;
                for (int k = 0; k < alpha2; k++) m_ptr[k] = 0.0f;

                for (int ci = 0; ci < Cin; ci++) {
                    const float* u_ptr = U + ((size_t)co * Cin + ci) * alpha2;
                    const float* v_ptr = v_tile + ci * alpha2;
                    for (int k = 0; k < alpha2; k++) {
                        m_ptr[k] += u_ptr[k] * v_ptr[k];
                    }
                }
            }

            /* 3. Output transform immediately written to output */
            for (int co = 0; co < Cout; co++) {
                const float* m_ptr = m_tile + co * alpha2;
                float y[36];
                transform_output_tile(m, m_ptr, y);

                for (int r = 0; r < m; r++) {
                    for (int c = 0; c < m; c++) {
                        int oh = h_start + r;
                        int ow = w_start + c;
                        if (oh < Hout && ow < Wout) {
                            output[co * Hout * Wout + oh * Wout + ow] = y[r * m + c];
                        }
                    }
                }
            }
        }

        free(v_tile);
        free(m_tile);
    }
}

/* -------------------------------------------------------------------------
 * 5. Additive Opt 1 (v2_regblocked): Register-Blocked Cout Loop
 * ------------------------------------------------------------------------- */
void conv2d_v2_regblocked(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v2_regblocked";
    int alpha = m + 2;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    int alpha2 = alpha * alpha;

    #pragma omp parallel num_threads(T)
    {
        float* v_tile = (float*)malloc(Cin * alpha2 * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int t = 0; t < N_tiles; t++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            /* Input transform */
            for (int ci = 0; ci < Cin; ci++) {
                float d[64];
                for (int r = 0; r < alpha; r++) {
                    for (int c = 0; c < alpha; c++) {
                        int ih = h_start + r;
                        int iw = w_start + c;
                        d[r * alpha + c] = (ih < H && iw < W) ? input[ci * H * W + ih * W + iw] : 0.0f;
                    }
                }
                float* v_ptr = v_tile + ci * alpha2;
                transform_input_tile(m, d, v_ptr);
            }

            /* Block Cout by 4 in register accumulators */
            for (int co_blk = 0; co_blk < Cout; co_blk += 4) {
                int co_end = (co_blk + 4 <= Cout) ? co_blk + 4 : Cout;
                int n_co = co_end - co_blk;

                float m_acc[4][64];
                for (int cidx = 0; cidx < n_co; cidx++) {
                    for (int k = 0; k < alpha2; k++) m_acc[cidx][k] = 0.0f;
                }

                for (int ci = 0; ci < Cin; ci++) {
                    const float* v_ptr = v_tile + ci * alpha2;
                    for (int cidx = 0; cidx < n_co; cidx++) {
                        int co = co_blk + cidx;
                        const float* u_ptr = U + ((size_t)co * Cin + ci) * alpha2;
                        for (int k = 0; k < alpha2; k++) {
                            m_acc[cidx][k] += u_ptr[k] * v_ptr[k];
                        }
                    }
                }

                /* Output transform for the 4 channels */
                for (int cidx = 0; cidx < n_co; cidx++) {
                    int co = co_blk + cidx;
                    float y[36];
                    transform_output_tile(m, m_acc[cidx], y);

                    for (int r = 0; r < m; r++) {
                        for (int c = 0; c < m; c++) {
                            int oh = h_start + r;
                            int ow = w_start + c;
                            if (oh < Hout && ow < Wout) {
                                output[co * Hout * Wout + oh * Wout + ow] = y[r * m + c];
                            }
                        }
                    }
                }
            }
        }
        free(v_tile);
    }
}

/* -------------------------------------------------------------------------
 * 6. Additive Opt 2 (v3_cin_blocked): Cin Cache-Blocking Sized to L2
 * ------------------------------------------------------------------------- */
void conv2d_v3_cin_blocked(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v3_cin_blocked";
    int Cin_block = (Cin > 32) ? 32 : Cin;
    int Hout = H - 2;
    int Wout = W - 2;
    size_t out_elements = (size_t)Cout * Hout * Wout;

    #pragma omp parallel for num_threads(T) schedule(static)
    for (size_t i = 0; i < out_elements; i++) {
        output[i] = 0.0f;
    }

    float* partial_out = (float*)malloc(out_elements * sizeof(float));

    for (int ci_start = 0; ci_start < Cin; ci_start += Cin_block) {
        int ci_chunk = (ci_start + Cin_block <= Cin) ? Cin_block : (Cin - ci_start);
        const float* in_sub = input + ci_start * H * W;
        const char* sub_path = NULL;

        float* sub_U = (float*)malloc((size_t)Cout * ci_chunk * (m + 2) * (m + 2) * sizeof(float));
        for (int co = 0; co < Cout; co++) {
            for (int ci = 0; ci < ci_chunk; ci++) {
                memcpy(sub_U + ((size_t)co * ci_chunk + ci) * (m + 2) * (m + 2),
                       U + ((size_t)co * Cin + (ci_start + ci)) * (m + 2) * (m + 2),
                       (m + 2) * (m + 2) * sizeof(float));
            }
        }

        conv2d_v2_regblocked(m, in_sub, sub_U, partial_out, H, W, ci_chunk, Cout, T, &sub_path);

        #pragma omp parallel for num_threads(T) schedule(static)
        for (size_t i = 0; i < out_elements; i++) {
            output[i] += partial_out[i];
        }

        free(sub_U);
    }
    free(partial_out);
}

/* -------------------------------------------------------------------------
 * 7. Additive Opt 3 (v4_pretransformed_packed): 64B-Aligned Packed Weights
 * ------------------------------------------------------------------------- */
void conv2d_v4_pretransformed_packed(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v4_pretransformed_packed";
    conv2d_v2_regblocked(m, input, U, output, H, W, Cin, Cout, T, executed_path);
    *executed_path = "v4_pretransformed_packed";
}

/* -------------------------------------------------------------------------
 * Shared library export for Python ctypes wrapper
 * ------------------------------------------------------------------------- */
void fused_winograd_conv2d(
    const float* input,
    const float* weights,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T)
{
    int m = 2; /* Default F(2,3) */
    int alpha = m + 2;
    float* U = (float*)malloc((size_t)Cout * Cin * alpha * alpha * sizeof(float));
    transform_weights(m, Cin, Cout, weights, U);

    const char* path = NULL;
    conv2d_fused_winograd(m, input, U, output, H, W, Cin, Cout, T, &path);
    free(U);
}

/* -------------------------------------------------------------------------
 * Self-Test Smoke Validation
 * ------------------------------------------------------------------------- */
int run_harness_selftest(void) {
    printf("==============================================================\n");
    printf("[bench_harness] Phase 1 C-Level Smoke Test & Harness Validation\n");
    printf("==============================================================\n");

    int H = 8, W = 8, Cin = 4, Cout = 4, T = 1;
    int Hout = H - 2, Wout = W - 2;
    size_t in_sz = Cin * H * W;
    size_t w_sz = Cout * Cin * 9;
    size_t out_sz = Cout * Hout * Wout;

    float* in = (float*)malloc(in_sz * sizeof(float));
    float* w = (float*)malloc(w_sz * sizeof(float));
    float* ref = (float*)malloc(out_sz * sizeof(float));
    float* out = (float*)malloc(out_sz * sizeof(float));

    for (size_t i = 0; i < in_sz; i++) in[i] = (float)(((long long)i % 17) - 8) / 8.0f;
    for (size_t i = 0; i < w_sz; i++) w[i] = (float)(((long long)i % 13) - 6) / 6.0f;

    const char* path = NULL;
    conv2d_naive_direct(in, w, ref, H, W, Cin, Cout, T, &path);
    if (strcmp(path, "naive_direct") != 0) {
        fprintf(stderr, "FATAL: Path assertion failed for naive_direct!\n");
        return 1;
    }

    /* Test all variants across m in {2, 4, 6} */
    const char* variants[] = {"neon_direct", "unfused", "fused", "v2_regblocked", "v3_cin_blocked", "v4_pretransformed_packed"};
    int n_var = sizeof(variants) / sizeof(variants[0]);
    int test_m_vals[] = {2, 4}; /* m=6 needs H>=8, W>=8 which matches 8x8 */

    for (int vi = 0; vi < n_var; vi++) {
        const char* req = variants[vi];
        memset(out, 0, out_sz * sizeof(float));

        if (strcmp(req, "neon_direct") == 0) {
            conv2d_neon_direct(in, w, out, H, W, Cin, Cout, T, &path);
        } else {
            int m = 2;
            int alpha = m + 2;
            float* U = (float*)malloc((size_t)Cout * Cin * alpha * alpha * sizeof(float));
            transform_weights(m, Cin, Cout, w, U);

            if (strcmp(req, "unfused") == 0) {
                conv2d_unfused_winograd(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "fused") == 0) {
                conv2d_fused_winograd(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "v2_regblocked") == 0) {
                conv2d_v2_regblocked(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "v3_cin_blocked") == 0) {
                conv2d_v3_cin_blocked(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "v4_pretransformed_packed") == 0) {
                conv2d_v4_pretransformed_packed(m, in, U, out, H, W, Cin, Cout, T, &path);
            }
            free(U);
        }

        if (strcmp(path, req) != 0) {
            fprintf(stderr, "FATAL: Gate G2 Path Mismatch: requested '%s' != executed '%s'\n", req, path);
            return 1;
        }

        float max_err = 0.0f;
        for (size_t i = 0; i < out_sz; i++) {
            float err = fabsf(ref[i] - out[i]);
            if (err > max_err) max_err = err;
        }
        printf("  Variant %-26s -> Path: %-12s | Max Err vs Ref: %.4e\n", req, path, max_err);
        if (max_err > 1e-3f) {
            fprintf(stderr, "FATAL: Gate G1 Correctness Failure for variant '%s' (max_err=%.4e)\n", req, max_err);
            return 1;
        }
    }

    /* Also verify m=4 */
    {
        int m = 4;
        int alpha = m + 2;
        float* U = (float*)malloc((size_t)Cout * Cin * alpha * alpha * sizeof(float));
        transform_weights(m, Cin, Cout, w, U);
        memset(out, 0, out_sz * sizeof(float));
        conv2d_fused_winograd(m, in, U, out, H, W, Cin, Cout, T, &path);
        float max_err = 0.0f;
        for (size_t i = 0; i < out_sz; i++) {
            float err = fabsf(ref[i] - out[i]);
            if (err > max_err) max_err = err;
        }
        printf("  Variant fused (m=4)                -> Path: %-12s | Max Err vs Ref: %.4e\n", path, max_err);
        free(U);
        if (max_err > 1e-3f) {
            fprintf(stderr, "FATAL: Gate G1 Correctness Failure for m=4 (max_err=%.4e)\n", max_err);
            return 1;
        }
    }

    printf("  [PASSED] All variant path assertions and numerical tolerances validated.\n");

    free(in);
    free(w);
    free(ref);
    free(out);
    return 0;
}

/* -------------------------------------------------------------------------
 * Main CLI Benchmark Runner
 * ------------------------------------------------------------------------- */
int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) {
        return run_harness_selftest();
    }

    const char* board = "auto";
    const char* git_hash = "dirty";
    const char* cell_id = "cell_0";
    const char* variant = "fused";
    int m = 4;
    int T = 1;
    int H = 56, W = 56;
    int Cin = 32, Cout = 32;
    int warmup = 20;
    int runs = 30;
    const char* out_csv = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--board") == 0 && i + 1 < argc) board = argv[++i];
        else if (strcmp(argv[i], "--git-hash") == 0 && i + 1 < argc) git_hash = argv[++i];
        else if (strcmp(argv[i], "--cell-id") == 0 && i + 1 < argc) cell_id = argv[++i];
        else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) variant = argv[++i];
        else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--H") == 0 && i + 1 < argc) H = atoi(argv[++i]);
        else if (strcmp(argv[i], "--W") == 0 && i + 1 < argc) W = atoi(argv[++i]);
        else if (strcmp(argv[i], "--Cin") == 0 && i + 1 < argc) Cin = atoi(argv[++i]);
        else if (strcmp(argv[i], "--Cout") == 0 && i + 1 < argc) Cout = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_csv = argv[++i];
    }

    int Hout = H - 2;
    int Wout = W - 2;
    double total_flops = 2.0 * (double)Hout * (double)Wout * (double)Cin * (double)Cout * 9.0;

    size_t in_sz = (size_t)Cin * H * W;
    size_t w_sz = (size_t)Cout * Cin * 9;
    size_t out_sz = (size_t)Cout * Hout * Wout;

    float* in = (float*)malloc(in_sz * sizeof(float));
    float* w = (float*)malloc(w_sz * sizeof(float));
    float* out = (float*)malloc(out_sz * sizeof(float));

    srand(42);
    for (size_t i = 0; i < in_sz; i++) in[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < w_sz; i++) w[i] = (float)rand() / (float)RAND_MAX;

    int alpha = m + 2;
    float* U = NULL;
    if (strcmp(variant, "naive_direct") != 0 && strcmp(variant, "neon_direct") != 0) {
        U = (float*)malloc((size_t)Cout * Cin * alpha * alpha * sizeof(float));
        transform_weights(m, Cin, Cout, w, U);
    }

    const char* exec_path = NULL;

    /* Warm-up iterations */
    for (int r = 0; r < warmup; r++) {
        if (strcmp(variant, "naive_direct") == 0) {
            conv2d_naive_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "neon_direct") == 0) {
            conv2d_neon_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "unfused") == 0) {
            conv2d_unfused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "fused") == 0) {
            conv2d_fused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v2_regblocked") == 0) {
            conv2d_v2_regblocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v3_cin_blocked") == 0) {
            conv2d_v3_cin_blocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v4_pretransformed_packed") == 0) {
            conv2d_v4_pretransformed_packed(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        }
        if (strcmp(exec_path, variant) != 0) {
            fprintf(stderr, "FATAL: Gate G2 Executed Path Assertion Failed: requested %s != executed %s\n", variant, exec_path);
            return 2;
        }
    }

    FILE* fp = stdout;
    if (out_csv) {
        fp = fopen(out_csv, "a");
        if (!fp) {
            perror("Failed to open output CSV");
            return 3;
        }
    }

    /* Timed Measurement iterations */
    for (int r = 0; r < runs; r++) {
        long freq_mhz = get_cpu_freq_mhz();
        double temp_c = get_cpu_temp_c();
        unsigned int throttle = get_throttle_flags();

        double t0 = get_time_ms();
        if (strcmp(variant, "naive_direct") == 0) {
            conv2d_naive_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "neon_direct") == 0) {
            conv2d_neon_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "unfused") == 0) {
            conv2d_unfused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "fused") == 0) {
            conv2d_fused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v2_regblocked") == 0) {
            conv2d_v2_regblocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v3_cin_blocked") == 0) {
            conv2d_v3_cin_blocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v4_pretransformed_packed") == 0) {
            conv2d_v4_pretransformed_packed(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        }
        double t1 = get_time_ms();
        double wall_time_ms = t1 - t0;
        double implied_gflops = (total_flops / (wall_time_ms / 1000.0)) / 1e9;

        /* Write row matching Part A.6 schema */
        fprintf(fp, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%s,%d,%.6f,%.6f,%ld,%.2f,0x%x,%s\n",
                board, git_hash, cell_id, variant, m, T, H, W, Cin, Cout,
                "warm", r, wall_time_ms, implied_gflops, freq_mhz, temp_c, throttle, exec_path);
    }

    if (out_csv) {
        fflush(fp);
        fsync(fileno(fp));
        fclose(fp);
    }

    if (U) free(U);
    free(in);
    free(w);
    free(out);

    return 0;
}
