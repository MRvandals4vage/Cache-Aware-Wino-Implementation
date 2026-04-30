# Cache-Aware Winograd Edge Benchmark Suite

This repository benchmarks a cache-aware fused Winograd implementation on CPU-class edge devices and compares it against optional external backends such as TVM, AutoTVM, and ARM Compute Library.

## Project Layout
- `src/`: core cache probing, autotiling, scheduling, and fused Winograd kernel code.
- `benchmarks/`: microbenchmark pipeline, result processing, and plot generation.
- `tools/`: executable Python utilities for backend comparison and ONNX export.
- `scripts/`: install and run scripts for macOS, Jetson Nano, and Raspberry Pi.
- `docs/`: focused runbooks for setup and the demo flow.
- `models/onnx/`: exported ONNX models.
- `artifacts/`: generated raw results, processed tables, plots, logs, and comparison reports.

## What Is Real Today
- The fused Winograd microbenchmark path is real and locally runnable.
- The benchmark pipeline under `benchmarks/run_all_benchmarks.py` is real and writes actual artifacts.
- The direct backend comparison harness under `tools/compare_edge_backends.py` is real.
- TVM and AutoTVM are optional. If they are not installed, they are reported as skipped instead of being faked.
- ARMCL is optional. The harness expects a real wrapper command that prints `LATENCY_MS=<value>`.

## Quick Start

### macOS
```bash
bash scripts/install_macos.sh
bash scripts/run_mac_benchmark.sh
```

### Jetson Nano
```bash
bash scripts/install_jetson_nano.sh
bash scripts/run_jetson_nano_benchmark.sh
```

### Raspberry Pi
```bash
bash scripts/install_raspberry_pi.sh
bash scripts/run_raspberry_pi_benchmark.sh
```

## Direct Comparison Harness
The comparison harness runs a single convolution workload across selected backends and writes CSV, Markdown, and JSON summaries to `artifacts/comparisons/`.

```bash
python3 tools/compare_edge_backends.py \
  --backends project,onnxruntime \
  --c-in 64 \
  --c-out 64 \
  --height 4 \
  --width 4 \
  --runs 20 \
  --warmup 5
```

On Jetson Nano or Raspberry Pi, you can include `tvm`, `autotvm`, and `armcl` in `--backends` if those runtimes are actually installed.

## ARMCL Integration
ARMCL support is intentionally explicit instead of guessed. Provide a wrapper command that runs your ARM Compute Library benchmark and prints:

```text
LATENCY_MS=<value>
```

Example:
```bash
export ARMCL_COMMAND='/absolute/path/run_armcl_wrapper.sh {c_in} {c_out} {height} {width} {runs}'
bash scripts/run_jetson_nano_benchmark.sh
```

The placeholders are expanded by `tools/compare_edge_backends.py`.

## Outputs
- `artifacts/raw/`: raw microbenchmark samples.
- `artifacts/processed/`: processed CSV and LaTeX tables.
- `artifacts/plots/`: generated figures.
- `artifacts/logs/`: platform descriptors, autotiling logs, and run logs.
- `artifacts/comparisons/`: direct backend comparison summaries.

## Tomorrow’s Runbook
Use [docs/tomorrow-demo.md](/Users/ishaanupponi/.codex/worktrees/2ebd/Cache-Aware-Wino-Implementation/docs/tomorrow-demo.md).
