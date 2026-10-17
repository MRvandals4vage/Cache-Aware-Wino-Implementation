# Dependency & Toolchain Audit Report
Generated: 2026-10-16T06:40:13Z | Mode: ONLINE

| Component | Status | Resolution Method | Affected Paper Tables / Claims |
| :--- | :---: | :--- | :--- |
| **C/C++ Compiler Toolchain** | OK | Native host compiler (Apple clang version 17.0.0 (clang-1700.3.19.1)) | None (Full compilation enabled) |
| **OpenMP Runtime** | FALLBACK(pthread) | Using in-repo pthread pool fallback | Table 4/5 multi-thread scaling uses pthread scheduler |
| **Hardware PMU Counters** | FALLBACK(indirect) | Indirect pointer-chasing and stride sweeps | Counter-based claims labeled indirect |
| **ARM Compute Library (ArmCL)** | BLOCKED | Shared library not pre-installed | ArmCL comparisons in Table 6/7 omitted |
| **Apache TVM / AutoTVM** | FALLBACK(RPC_or_Omitted) | TVM runtime not installed on host | TVM comparisons labeled unbuilt/omitted |
| **ONNX Runtime** | FALLBACK(in_house_graph) | In-house sequential graph runner | ORT baseline replaced with native runner |
| **Statistical & Plotting Stack** | OK | NumPy, SciPy, Matplotlib verified | Offline statistics and LaTeX generation ready |
