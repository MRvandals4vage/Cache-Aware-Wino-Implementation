# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 5, Warmup: 2

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.19735004752874374 | 0.0089724799547615 | 9.27x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.02129138447344303 | 0.003041701976587297 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
| armcl | error | N/A | N/A | N/A | /bin/sh: /Users/ishaanupponi/Documents/My: No such file or directory |
