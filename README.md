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



jetson@jetson-desktop:~/Documents/Cache Aware Winograd/Cache-Aware-Wino-Implementation$ make verify-jetson
python3 benchmarks/cache_probe_sanity.py \
	--out raw_logs/jetson_cacheprobe.csv
[A] Cache-probe sanity check
   Source  : sysfs
   L1D     : 32768 B  (32.0 KB)
   L2      : 2097152 B  (2.00 MB)
   Line    : 64 B
   L1D: paper=32768B, measured=32768B -> MATCH  (0.0%)
   L2: paper=2097152B, measured=2097152B -> MATCH  (0.0%)
[A] Written to raw_logs/jetson_cacheprobe.csv
python3 benchmarks/ws_model_validation.py \
	--runs 100 \
	--out raw_logs/jetson_ws_validation.csv \
	--summary summary/ws_model_validation_summary.csv
[B] Working-set model validation

  Config (16,32): tile=4, WS_model=18560B
  SKIPPED (16,32): perf returned no data

  Config (32,32): tile=4, WS_model=35968B
  SKIPPED (32,32): perf returned no data

  Config (64,64): tile=4, WS_model=70784B
  SKIPPED (64,64): perf returned no data

  Config (128,128): tile=4, WS_model=140416B
  SKIPPED (128,128): perf returned no data
[B] WARNING: No data collected. Skipping Task B.
python3 benchmarks/jetson_main_microbench.py \
	--runs 1000 --warmup 20 --height 56 --width 56 \
	--raw raw_logs/jetson_main_microbench.csv \
	--summary summary/jetson_main_microbench_summary.csv
[C] Jetson Nano Main Microbenchmark — ALL 7 configs
    n=1000, warmup=20, H'=56, W'=56

  Config (16,32):
      baseline_nonfused: mean=366.4786ms CI95=±0.5471ms
      fused: mean=545.5402ms CI95=±0.6980ms
    improvement=-48.86% | paper=None% | NO_PAPER_CLAIM

  Config (32,16):
      baseline_nonfused: mean=180.5611ms CI95=±0.1352ms
      fused: mean=272.3016ms CI95=±0.1021ms
    improvement=-50.81% | paper=None% | NO_PAPER_CLAIM

  Config (32,32):
      baseline_nonfused: mean=503.6642ms CI95=±0.8077ms
      fused: mean=609.2222ms CI95=±0.2950ms
    improvement=-20.96% | paper=None% | NO_PAPER_CLAIM

  Config (32,64):
      baseline_nonfused: mean=1655.0343ms CI95=±2.6678ms
      fused: mean=1466.1325ms CI95=±1.5765ms
    improvement=11.41% | paper=None% | NO_PAPER_CLAIM

  Config (64,32):
      baseline_nonfused: mean=789.5485ms CI95=±1.7410ms
      fused: mean=745.3864ms CI95=±0.7900ms
    improvement=5.59% | paper=None% | NO_PAPER_CLAIM

  Config (64,64):
      baseline_nonfused: mean=2736.3290ms CI95=±14.3341ms

