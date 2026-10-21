# Backend Comparison

- Platform: Linux / aarch64
- CPU: aarch64
- Workload: C_in=64, C_out=64, H=4, W=4, K=3
- Runs: 20, Warmup: 5

| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |
| :------ | :----- | ---------------: | -------: | ---------------: | :---- |
| project | ok | 3.560439650027547 | 1.628968696625147 | 29.38x | Fused Winograd F(2,3) tile microbenchmark. |
| onnxruntime | ok | 0.12119464986426465 | 0.011561653811379014 | 1.00x | Single Conv ONNX Runtime CPU benchmark. |
| tvm | skipped | N/A | N/A | N/A | TVM is not installed: No module named 'tvm' |
| autotvm | skipped | N/A | N/A | N/A | TVM is not installed: No module named 'tvm' |
| armcl | error | N/A | N/A | N/A | /home/jetson/Cache-Aware-Wino-Implementation/tools/run_armcl.sh: line 60: /home/jetson/Cache-Aware-Wino-Implementation/bin/armcl_benchmark: cannot execute binary file: Exec format error |
