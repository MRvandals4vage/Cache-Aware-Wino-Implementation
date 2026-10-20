# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 5, Warmup: 2

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.18013333901762962 | 0.00940743814236977 | 9.39x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.019182823598384857 | 0.0036624665685105586 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
| tvm | ok | 0.45553315430879593 | 0.0015818340608173105 | 23.75x | Single Conv TVM benchmark. |
| autotvm | ok | 0.4003777652978897 | 0.0011272247562666882 | 20.87x | Single Conv AutoTVM benchmark with up to 32 trials per task. |
| armcl | ok | 0.121583 | 0.0 | 6.34x | External ARM Compute Library wrapper command. |
