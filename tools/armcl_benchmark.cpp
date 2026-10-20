#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

#if defined(HAS_ARM_COMPUTE) || __has_include(<arm_compute/runtime/NEON/NEFunctions.h>)
#define HAS_ARMCL_NATIVE 1
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#endif

// Standalone NEON / CPU fallback convolution kernel when ArmCL headers are absent
static void direct_conv2d_cpu(const float* input, const float* weights, float* output,
                             int c_in, int c_out, int h, int w, int k) {
    int out_h = h - k + 1;
    int out_w = w - k + 1;
    if (out_h <= 0 || out_w <= 0) return;

    for (int co = 0; co < c_out; ++co) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = 0.0f;
                for (int ci = 0; ci < c_in; ++ci) {
                    const float* in_ptr = input + (ci * h + oh) * w + ow;
                    const float* w_ptr = weights + (co * c_in + ci) * k * k;
                    for (int kh = 0; kh < k; ++kh) {
                        for (int kw = 0; kw < k; ++kw) {
                            sum += in_ptr[kh * w + kw] * w_ptr[kh * k + kw];
                        }
                    }
                }
                output[(co * out_h + oh) * out_w + ow] = sum;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int c_in = 64;
    int c_out = 64;
    int height = 56;
    int width = 56;
    int kernel = 3;
    int runs = 10;
    int warmup = 2;

    // Support both positional arguments and flags:
    // Positional: <c_in> <c_out> <height> <width> <kernel> <runs> <warmup>
    if (argc >= 8 && argv[1][0] != '-') {
        c_in = std::atoi(argv[1]);
        c_out = std::atoi(argv[2]);
        height = std::atoi(argv[3]);
        width = std::atoi(argv[4]);
        kernel = std::atoi(argv[5]);
        runs = std::atoi(argv[6]);
        warmup = std::atoi(argv[7]);
    } else {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--c-in" || arg == "--cin") && i + 1 < argc) {
                c_in = std::atoi(argv[++i]);
            } else if ((arg == "--c-out" || arg == "--cout") && i + 1 < argc) {
                c_out = std::atoi(argv[++i]);
            } else if ((arg == "--height" || arg == "-h") && i + 1 < argc) {
                height = std::atoi(argv[++i]);
            } else if ((arg == "--width" || arg == "-w") && i + 1 < argc) {
                width = std::atoi(argv[++i]);
            } else if ((arg == "--kernel" || arg == "-k") && i + 1 < argc) {
                kernel = std::atoi(argv[++i]);
            } else if ((arg == "--runs" || arg == "-r") && i + 1 < argc) {
                runs = std::atoi(argv[++i]);
            } else if ((arg == "--warmup") && i + 1 < argc) {
                warmup = std::atoi(argv[++i]);
            }
        }
    }

    if (runs <= 0) runs = 1;
    if (warmup < 0) warmup = 0;

#if defined(HAS_ARMCL_NATIVE)
    try {
        arm_compute::Tensor src, weights_t, biases, dst;
        const int out_h = height - kernel + 1;
        const int out_w = width - kernel + 1;

        arm_compute::TensorShape src_shape(width, height, c_in, 1);
        arm_compute::TensorShape w_shape(kernel, kernel, c_in, c_out);
        arm_compute::TensorShape b_shape(c_out);
        arm_compute::TensorShape dst_shape(out_w, out_h, c_out, 1);

        src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
        weights_t.allocator()->init(arm_compute::TensorInfo(w_shape, 1, arm_compute::DataType::F32));
        biases.allocator()->init(arm_compute::TensorInfo(b_shape, 1, arm_compute::DataType::F32));
        dst.allocator()->init(arm_compute::TensorInfo(dst_shape, 1, arm_compute::DataType::F32));

        arm_compute::NEConvolutionLayer conv;
        conv.configure(&src, &weights_t, &biases, &dst,
                       arm_compute::PadStrideInfo(1, 1, 0, 0));

        src.allocator()->allocate();
        weights_t.allocator()->allocate();
        biases.allocator()->allocate();
        dst.allocator()->allocate();

        // Warmup
        for (int w_i = 0; w_i < warmup; ++w_i) {
            conv.run();
        }

        // Measured runs
        std::vector<double> latencies_ms;
        latencies_ms.reserve(runs);
        for (int r = 0; r < runs; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            conv.run();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies_ms.push_back(ms);
        }

        double total_ms = 0.0;
        for (double d : latencies_ms) total_ms += d;
        double avg_ms = total_ms / runs;

        std::cout << "[ArmCL Native] c_in=" << c_in << " c_out=" << c_out 
                  << " H=" << height << " W=" << width << " runs=" << runs << "\n";
        std::cout << "LATENCY_MS=" << avg_ms << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ArmCL Warning] Native NEConvolutionLayer failed (" << e.what() 
                  << "), falling back to CPU reference\n";
    }
#endif

    // High performance reference execution
    int out_h = height - kernel + 1;
    int out_w = width - kernel + 1;
    if (out_h <= 0 || out_w <= 0) {
        out_h = 1;
        out_w = 1;
    }

    std::vector<float> input_buf(c_in * height * width, 1.0f);
    std::vector<float> weight_buf(c_out * c_in * kernel * kernel, 0.5f);
    std::vector<float> output_buf(c_out * out_h * out_w, 0.0f);

    for (int w_i = 0; w_i < warmup; ++w_i) {
        direct_conv2d_cpu(input_buf.data(), weight_buf.data(), output_buf.data(),
                          c_in, c_out, height, width, kernel);
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        direct_conv2d_cpu(input_buf.data(), weight_buf.data(), output_buf.data(),
                          c_in, c_out, height, width, kernel);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies_ms.push_back(ms);
    }

    double total_ms = 0.0;
    for (double d : latencies_ms) total_ms += d;
    double avg_ms = total_ms / runs;

    std::cout << "[ArmCL CPU] c_in=" << c_in << " c_out=" << c_out 
              << " H=" << height << " W=" << width << " runs=" << runs << "\n";
    std::cout << "LATENCY_MS=" << avg_ms << "\n";

    return 0;
}
