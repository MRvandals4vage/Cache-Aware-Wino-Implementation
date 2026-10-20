# Backend Comparison

- Platform: Darwin / arm64
- CPU: arm
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 5, Warmup: 2

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 0.21670013666152954 | 0.040657875718705085 | 11.39x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.019033346325159073 | 0.003519230652375799 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
| tvm | ok | 0.4548249766230583 | 0.001147579759284224 | 23.90x | Single Conv TVM benchmark. |
| autotvm | ok | 0.40404465794563293 | 0.004878550151791319 | 21.23x | Single Conv AutoTVM benchmark with up to 32 trials per task. |
| armcl | ok | 0.122083 | 0.0 | 6.41x | External ARM Compute Library wrapper command. |
