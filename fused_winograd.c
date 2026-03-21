#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ARM NEON check
#if defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

// Fused Winograd F(2,3)
// input: C_in x 4 x 4
// kernel: C_out x C_in x 3 x 3
// output: C_out x 2 x 2
// we will pre-transform kernel, so kernel is actually C_out x C_in x 4 x 4 (U)
// fused operation: input_transform -> multiply -> output_transform
void fused_winograd_f23_fallback(
    const float* input,
    const float* U,
    float* output,
    int c_in,
    int c_out)
{
    // input is C_in x 4 x 4
    // U is C_out x C_in x 4 x 4
    // output is C_out x 2 x 2
    
    // allocate small buffer for V (transformed input) 
    // to keep it resident in cache. size: C_in x 4 x 4
    // Actually we can process it in chunks if c_in is large.
    // For simplicity, we allocate V on stack or heap for C_in.
    float* V = (float*)malloc(c_in * 16 * sizeof(float));
    
    // 1. Input Transform
    // BT: 
    //  1  0 -1  0
    //  0  1  1  0
    //  0 -1  1  0
    //  0  1  0 -1
    // d is 4x4. V = BT * d * B
    // B is BT.T
    for (int ic = 0; ic < c_in; ic++) {
        const float* d = input + ic * 16;
        float* v = V + ic * 16;
        
        // tmp = BT * d
        float tmp[16];
        for (int i = 0; i < 4; i++) {
            tmp[0*4+i] = d[0*4+i] - d[2*4+i];
            tmp[1*4+i] = d[1*4+i] + d[2*4+i];
            tmp[2*4+i] = -d[1*4+i] + d[2*4+i];
            tmp[3*4+i] = d[1*4+i] - d[3*4+i];
        }
        // V = tmp * B
        for (int i = 0; i < 4; i++) {
            v[i*4+0] = tmp[i*4+0] - tmp[i*4+2];
            v[i*4+1] = tmp[i*4+1] + tmp[i*4+2];
            v[i*4+2] = -tmp[i*4+1] + tmp[i*4+2];
            v[i*4+3] = tmp[i*4+1] - tmp[i*4+3];
        }
    }
    
    // 2. Multiply and Output Transform
    // AT:
    //  1  1  1  0
    //  0  1 -1 -1
    // M = sum_ic( U[oc,ic] * V[ic] )
    // Y = AT * M * A
    for (int oc = 0; oc < c_out; oc++) {
        float M[16] = {0};
        for (int ic = 0; ic < c_in; ic++) {
            const float* u = U + (oc * c_in + ic) * 16;
            const float* v = V + ic * 16;
            for (int i = 0; i < 16; i++) {
                // optional: Exploit sparsity by skipping zero-valued U elements
                if (u[i] != 0.0f) {
                    M[i] += u[i] * v[i];
                }
            }
        }
        
        // Output transform
        // tmp = AT * M
        float tmp[8]; // 2x4
        for (int i = 0; i < 4; i++) {
            tmp[0*4+i] = M[0*4+i] + M[1*4+i] + M[2*4+i];
            tmp[1*4+i] = M[1*4+i] - M[2*4+i] - M[3*4+i];
        }
        
        // Y = tmp * A (A is AT.T)
        float* y = output + oc * 4;
        for (int i = 0; i < 2; i++) {
            y[i*2+0] = tmp[i*4+0] + tmp[i*4+1] + tmp[i*4+2];
            y[i*2+1] = tmp[i*4+1] - tmp[i*4+2] - tmp[i*4+3];
        }
    }
    
    free(V);
}

void fused_winograd_f23(
    const float* input,
    const float* U,
    float* output,
    int c_in,
    int c_out)
{
#if USE_NEON
    // NEON optimized path (simplified generic wrapper mapping to fallback if pure intrinsics not fully laid out)
    // A full NEON implementation would use vld1q_f32, vmulq_f32, vaddq_f32.
    // For this demonstration, we use the fallback logic but wrapped to signify presence.
    // Real NEON would operate block by block.
    fused_winograd_f23_fallback(input, U, output, c_in, c_out);
#else
    fused_winograd_f23_fallback(input, U, output, c_in, c_out);
#endif
}
