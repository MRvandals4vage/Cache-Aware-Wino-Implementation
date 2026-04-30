# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 20, Warmup: 5

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.18073750034091063 | 0.01504819511814893 | 10.99x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.01644784933887422 | 0.0009903959584868518 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
