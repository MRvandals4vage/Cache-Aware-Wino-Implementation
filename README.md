# Runtime-Adaptive Cache-Aware Fused Winograd Execution for Edge CPUs

This repository provides a benchmark suite and reference implementation for **Runtime-Adaptive Cache-Aware Fused Winograd Execution**. The focus of this work is analyzing and maximizing energy efficiency and runtime latency on resource-constrained Edge CPUs (ARM Cortex-A series, Jetson Nano, Raspberry Pi 5) via L1-cache footprints and dynamic tiling.

## Repository Layout
- `benchmarks/`: Contains statistically rigorous microbenchmarks generating CSV outputs.
- `scripts/`: Contains one-click execution wrappers.
- `docker/`: Dockerfiles for standard reproduction environments (Jetson Nano, RPi 5).
- `artifacts/`: Automatically saves the CSV results and plot generation.
- `fused_winograd_kernel.py`: Python module / C-ext wrapper implementing our custom fused memory-aware operations.
- `cache_adaptive_autotiler.py` & `runtime_cache_probe.py`: Heuristics framework for dynamic L1 footprints.

## Key Contributions
1. **Cache Adaptive Autotiler**: Replaces naive manual fixed-schedule tiles with a probing mechanism that bounds Winograd $F(m,r)$ footprints to the physical L1 constraints.
2. **Fused Winograd Execution**: Minimizes pipeline buffers (DRAM drops) by running transform, element-wise multiplication, and inverse transforms within an L1-pinned block.
3. **Reproducible Benchmarks**: Comprehensive multi-metric evaluations with Welch t-tests and baseline comparisons.

## Reproduction Flow (1-Command)
For a straightforward validation of the methodology:

```bash
bash scripts/run_microbenchmarks.sh
```
This runs `benchmarks/microbenchmarks.py`, simulating combinations across core scaling ($1$ vs $N$), kernel types (Baseline vs Fused), and cache shapes. Results are automatically output into the `artifacts/` folder as structured `.csv` datasets.

## Environment Setup
Dependencies are pinned up to versions explicitly targeting our release profile.

### Option A: Local Python Environment
Create a `venv` and source dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# To enable the fast C extension fallback
gcc -shared -o fused_winograd.so -fPIC -O3 -march=native fused_winograd.c
```

### Option B: Dockerized Reproducibility Setup
We provide robust Docker containers minimizing host configuration differences on specific platforms (Jetson or RPi5):
```bash
# Build Jetson Nano Image
docker build -t edge-winograd:jetson -f docker/jetson-nano.Dockerfile .

# Run entrypoint script
docker run --rm -v $(pwd)/artifacts:/workspace/artifacts edge-winograd:jetson
```

## Release & Archival
This repository is tagged as `v1.0.0-release` for artifact evaluation (Zenodo/ACM guidelines matching).
All metadata for experiments, including raw CSVs and generated plots, refer to the timestamped files in the `artifacts/` directory. For a full breakdown of theoretical vs empirical measurements across $resnet$ and $vgg$ backbones, see the provided `benchmark_results_measured.md`.

## Raspberry Pi Support Matrix

| Feature | Raspberry Pi 4 | Raspberry Pi 5 | Notes |
| :--- | :--- | :--- | :--- |
| **Microbenchmarks (Latency/Tile Scheduling)** | Supported | Supported | Full core cache-adaptation. |
| **Platform Discovery (`vcgencmd`/`psutil`)** | Supported | Supported | Outputs raw hardware status. |
| **Hardware Counters (`perf`)** | Partial | Partial | Requires OS to have `linux-perf` and permissions. |
| **Full Architecture Inference (PyTorch/ONNX)** | Optional | Optional | Only loaded if 64-bit OS + sufficient RAM handles framework installs. |
| **Power/Energy Measurement** | Unsupported | Unsupported | Direct hardware power nodes missing on Pis. Requires external meter. |

## Raspberry Pi Installation

For an environment completely abstracted away from heavy ML tools that fail on ARMv7 or low RAM nodes, you can execute our Pi-native deployment script. This installs exclusively lightweight core requirements:

```bash
bash scripts/setup_raspberry_pi.sh
```
*Note: The script outputs logs to `artifacts/logs/raspberry_pi_setup.log`.*

## Raspberry Pi Benchmark Commands

Once installed, use the tailored runtime wrapper that automatically probes standard ARM `vcgencmd` structures and logs data properly:

```bash
bash scripts/run_raspberry_pi_benchmark.sh
```
*Note: The wrapper runs the rigorous microbenchmarks, captures thermal/clock throttling, and saves logs to `artifacts/logs/raspberry_pi_benchmark.log`.*

## Known Limitations on Raspberry Pi

- **No Direct Power/Energy API**: Unlike Jetson's `tegrastats`, the Raspberry Pi lacks native high-resolution system power profiling nodes in `sysfs`. Energy calculations cannot be blindly fabricated. 
- **Optional HW Counters**: If permissions are missing for `perf` (e.g., standard Docker runs without `--privileged`), the benchmark falls back accurately to `psutil`/`vcgencmd` latency logging and omits simulated L1 cache miss stats.
- **Python ML Framework Overhead**: Large ONNX packages are not officially tested or supported on small Pi configurations. We explicitly disable full-model tests if PyTorch/ONNX fail to invoke, protecting the runtime from Silent crashing. 

## Optional External Power Measurement Workflow

If you require precise *Joules/MAC* evaluation on the Raspberry Pi:
1. Connect a USB hardware power meter (e.g., Makerhawk or RuiDeng) intercepting the Pi's power feed.
2. Synchronize your meter's logging timeframe to match the output timestamps inside the `artifacts/processed/microbenchmark_results.csv`.
3. Calculate `Energy = Average_Power` measured physically `* average_latency` exported by the benchmark. Do not substitute this algorithmically without genuine hardware.
