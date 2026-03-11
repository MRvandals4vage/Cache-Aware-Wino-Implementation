# Cache-Aware Winograd Scheduling for Energy-Efficient CNN Inference on Edge CPUs

[![Research Grade](https://img.shields.io/badge/Status-Research--Grade-blueviolet)](#)
[![Hardware](https://img.shields.io/badge/Hardware-Jetson--Nano-green)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch--ONNX--TVM-blue)](#)

This repository contains a comprehensive experimental framework for evaluating **Memory-Aware Convolution Execution Strategies** on Edge CPUs, specifically targeting the **NVIDIA Jetson Nano (ARM Cortex-A57)**.

The framework benchmarks convolution backends across standard CNN architectures (**ResNet-18, VGG16, AlexNet, ResNet34**) to evaluate how Winograd scheduling affecting latency, DRAM traffic, and total system energy.

---

## 🚀 Key Features

- **Multi-Rail Hardware Power Monitoring**: Real-time hardware sampling of VDD_IN (Total), VDD_CPU, and VDD_GPU via INA3221 sensors at 10ms intervals.
- **Measurement-Driven Profiling**: Replaces theoretical constants with high-precision metrics (`time.perf_counter`, `psutil`) for latency and power.
- **Dynamic MAC Counting**: Runtime graph observation using the `thop` profiling library for precise operation counts across different layer types.
- **Memory Traffic Validation**: Dynamic `MemoryTracer` that performs analytical tracing of architecture-specific data movement patterns.
- **Cross-Backend Support**: Seamless comparison between **PyTorch Baseline**, **Naive Winograd**, **Cache-Aware**, and **TVM Model**.
- **Publication-Ready Artifacts**: Automatic generation of Markdown reports and research-grade plots (`latency_comparison.png`, `energy_comparison.png`, etc.).

---

## 🛠 Project Structure

```text
├── run_jetson_benchmark.py   # Primary research pipeline for real hardware profiling
├── main.py                   # Main entry point for multi-architecture benchmarking
├── benchmark.py              # Core measurement-driven benchmarking engine
├── cnn_model.py              # standard full-scale architecture defs (ImageNet)
├── memory_scheduler.py       # Winograd scheduling implementations
├── memory_trace.py           # Analytical DRAM traffic estimator
├── visualization.py          # Publication-ready plotting engine
├── power_monitor.py          # Jetson INA3221 sensor integration & generic falls-backs
└── tvm_compiler.py           # ONNX to TVM compilation utilities
```

---

## 📊 Methodology

### 1. Power & Energy Measurement
Total energy for an inference pass is derived from instantaneous hardware power samples and high-precision latency observations:
$$E_{inference} = \int_{0}^{t} P(t) dt \approx P_{avg} \times t_{measured}$$

### 2. Architecture Suite
All models are standard full-sized versions (Input: $1 \times 3 \times 224 \times 224$):
- **VGG16**: Deep stack of $3 \times 3$ convolutions, highly sensitive to memory traffic.
- **AlexNet**: Classic architecture with diverse kernel sizes.
- **ResNet-18/34**: Modern residual networks with varying depth.

---

## 🏃 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch & Torchvision
- ONNX & ONNX Runtime
- Matplotlib, NumPy, Psutil, Thop

### Installation
```bash
git clone https://github.com/ishaanupponi/edge_conv_benchmarking.git
cd edge_conv_benchmarking
pip install -r requirements.txt
```

### Running Benchmarks
To run the full multi-architecture experimental suite:
```bash
python3 main.py --model all
```

---

## 📈 Results Preview

The framework produces a comprehensive measurement breakdown in `benchmark_results_measured.md`:

| Architecture | Strategy | Latency (ms) | Power (mW) | Energy (mJ) | MACs/J |
| :--- | :--- | :---: | :---: | :---: | :---: |
| VGG16 | Baseline | 36.24 | 2531.1 | 91.73 | 1.69e+11 |
| VGG16 | TVM Model | 44.16 | 2526.7 | 111.57 | 1.39e+11 |

Reports and traces are automatically saved to `benchmark_results_measured.md` and `memory_analysis_report.md`.

---

## 📝 Citation
If you use this framework in your research, please cite:
> "Cache-Aware Winograd Scheduling for Energy-Efficient CNN Inference on Edge CPUs" (2026).
