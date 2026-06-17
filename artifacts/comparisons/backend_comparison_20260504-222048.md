# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=128, C_out=128, H=128, W=128, K=3
- Runs: 30, Warmup: 5

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.2736458011592428 | 0.06436701108871085 | 1.00x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 5.9722930668309955 | 0.3014034034668172 | 21.82x | Single Conv ONNX Runtime CPU benchmark. |
