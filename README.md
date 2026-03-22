# Runtime-Adaptive Cache-Aware Fused Winograd Execution for Edge CPUs

This repository provides a benchmark suite and reference implementation for **Runtime-Adaptive Cache-Aware Fused Winograd Execution**. The focus of this work is analyzing and maximizing energy efficiency and runtime latency on resource-constrained Edge CPUs (ARM Cortex-A series, Jetson Nano, Raspberry Pi) via L1-cache footprints and dynamic tiling.

## Repository Layout
- `src/`: Core implementation containing the platform probing, autotiler, scheduling logic, and fused Winograd kernel.
- `benchmarks/`: CLI scripts and logic for executing benchmarks, processing data, collecting hardware counters, and generating plots.
- `scripts/`: Shell scripts for environment setup, particularly for Raspberry Pi.
- `docker/`: Dockerfiles for standard reproduction environments, including Raspberry Pi (ARM64).
- `artifacts/`: Automatically saves the CSV results, telemetry logs, configuration files, and plots.

## Key Contributions
1. **Cache Adaptive Autotiler (`src/cache_adaptive_autotiler.py`)**: Evaluates Winograd $F(m,r)$ footprints against physical L1 constraints collected at runtime. Falls back dynamically if memory sizes are unknown.
2. **Fused Winograd Execution (`src/fused_winograd_kernel.py`)**: Minimizes pipeline buffers (DRAM drops) by running transform, element-wise multiplication, and inverse transforms sequentially. Includes working numpy fallback path when compiled C/NEON paths are unavailable.
3. **Reproducible Benchmarks**: Comprehensive multi-metric evaluations with statistical significance tests against non-fused baselines.

## Environment Setup and Installation

### Raspberry Pi (Recommended ARM Deployment)
For a straightforward validation on Raspberry Pi, run the custom install script. This installs exclusively lightweight core requirements (NumPy, SciPy, Pandas, Matplotlib) and attempts to compile the C extension natively without requiring massive ML frameworks.

```bash
bash scripts/install_raspberry_pi.sh
```
*Note: This script will create a virtual environment (`venv_winograd`) and install necessary packages gracefully.*

### Dockerized Setup
We provide Docker containers for specific platform targets:
```bash
# Build Raspberry Pi ARM64 Image (assuming you are on a compatible host or using buildx)
docker build -t edge-winograd:rpi -f docker/raspberry-pi.Dockerfile .

# Run the base microbenchmarks automatically
docker run --rm -v $(pwd)/artifacts:/app/artifacts edge-winograd:rpi
```

## Running Benchmarks (Publication Pipeline)

### 1. Benchmark Execution and Paper Assets
To generate raw data across different configurations and automatically output paper-facing latex tables and plots, use the orchestrator:
```bash
# 1. Microbenchmarks (Evaluates Custom Fused Winograd Kernels)
python3 benchmarks/run_all_benchmarks.py --mode micro --runs 1000 --warmup 20 --paper-assets all

# 2. End-to-End ONNX CNN Workloads (ResNet, VGG, AlexNet)
python3 benchmarks/run_all_benchmarks.py --mode end-to-end --model all --runs 50 --warmup 10 --paper-assets all
```
This writes raw datasets into `artifacts/raw/`, processes statistics seamlessly into `artifacts/processed/`, and generates matplotlib publication figures into `artifacts/plots/`.

### 2. Manual Processing / Latex Tuning (Optional)
If you wish to rerun the generators on existing raw data to toggle latex features or format tables:
```bash
python3 benchmarks/process_results.py --export-latex true
```

### 3. Collecting Telemetry
Collect ad-hoc hardware counters and statistics (e.g., perf, vcgencmd for temperature/clocks):
```bash
python3 benchmarks/collect_counters.py
```

## Supported Platforms and Limitations

| Feature | Supported Systems | Limitations & Truthful Disclaimers |
| :--- | :--- | :--- |
| **Microbenchmarks (Latency)** | Linux (ARM/x86), macOS | Runs reliably on all devices parsing NumPy. Non-essential architectures use a pure-numpy path. |
| **Platform Discovery** | Linux `sysfs`/`lscpu`, macOS `sysctl` | Accurately discovers L1/L2 on standard Linux/macOS. Gracefully falls back to 32KB/2MB defaults if permissions or OS fail. |
| **Hardware Telemetry** | Linux (`perf`), Raspberry Pi (`vcgencmd`) | `perf` requires `linux-perf` and sysctl `kernel.perf_event_paranoid` to be $\le 1$. `vcgencmd` is Pi exclusive. |
| **End-to-End Inference** | N/A | Current benchmark orchestrator is explicitly restricted to `micro` mode. E2E benchmarks (PyTorch/ONNX) are not supported on resource-constrained Pi loops. |
| **NEON Intrinsics** | Unsupported in Python | Python implementation measures baseline structural fusion algorithm performance. For true SIMD NEON speedup, the underlying GCC compiled `.so` extension must be loaded. |
| **Energy Measurement** | Unsupported natively on Pi | Direct hardware power nodes are missing on Pis. We do not fabricate energy metrics programmatically. Requires external power meters. |

## Success Criteria and Artifacts
The entire pipeline is verifiable: none of the plot generations or resulting data points are hardcoded placeholders. What you see logged in `artifacts/processed` represents actual execution footprints of the local machine.
