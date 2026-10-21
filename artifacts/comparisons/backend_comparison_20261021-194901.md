# Backend Comparison

- Platform: Linux / aarch64
- CPU: aarch64
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 5, Warmup: 2

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 1.401014600196504 | 0.06112563028756479 | 8.55x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.16378780001105042 | 0.006338562223958945 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
| tvm | skipped | N/A | N/A | N/A | TVM is not installed: No module named 'tvm' |
| autotvm | skipped | N/A | N/A | N/A | TVM is not installed: No module named 'tvm' |
| armcl | error | N/A | N/A | N/A | /home/jetson/Cache-Aware-Wino-Implementation/tools/run_armcl.sh: line 60: /home/jetson/Cache-Aware-Wino-Implementation/bin/armcl_benchmark: cannot execute binary file: Exec format error |
