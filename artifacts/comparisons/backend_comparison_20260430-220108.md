# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 10, Warmup: 2

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.1669376011705026 | 0.014976683618047429 | 9.19x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.018162500055041164 | 0.0020400137452460513 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
