# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 5, Warmup: 2

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.18050847575068474 | 0.008574255879588858 | 9.90x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.01823361963033676 | 0.00305574169759244 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
| armcl | ok | 0.1214 | 0.0 | 6.66x | External ARM Compute Library wrapper command. |
