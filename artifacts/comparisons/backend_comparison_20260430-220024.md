# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=64, C_out=64, H=14, W=14, K=3
- Runs: 5, Warmup: 1

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.17151659994851798 | 0.020636147327266267 | 3.89x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.04408340319059789 | 0.004519749724038058 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
