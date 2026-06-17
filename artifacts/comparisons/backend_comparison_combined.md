# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm

## Workload 1: C_in=64, C_out=64, H=4, W=4, K=3 (Micro-Input Regime)
- Runs: 20, Warmup: 5

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.1807 | 0.0150 | 10.99x | Fused Winograd F(2,3) tile microbenchmark. Overhead-dominated regime; Winograd transform cost dominates on 4x4 input. |
| onnxruntime | ok | 0.0164 | 0.0010 | 1.00x | Single Conv ONNX Runtime CPU benchmark. Direct convolution is optimal on micro-inputs. |

## Workload 2: C_in=128, C_out=128, H=128, W=128, K=3 (Large-Input Regime)
- Runs: 30, Warmup: 5

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.2736 | 0.0644 | 1.00x | Fused Winograd F(2,3) tile microbenchmark. Arithmetic-dominated regime; fused path dominates on large tiles. |
| onnxruntime | ok | 5.9723 | 0.3014 | 21.82x | Single Conv ONNX Runtime CPU benchmark. |
