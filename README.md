# Cache-Aware Winograd Scheduling for Energy-Efficient CNN Inference on Edge CPUs

[![Research Grade](https://img.shields.io/badge/Status-Research--Grade-blueviolet)](#)
[![Hardware](https://img.shields.io/badge/Hardware-Jetson--Nano-green)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch--ONNX--TVM-blue)](#)

This repository contains a comprehensive experimental framework for evaluating **Memory-Aware Convolution Execution Strategies** on Edge CPUs, specifically targeting the **NVIDIA Jetson Nano (ARM Cortex-A57)**.

The framework benchmarks convolution backends across standard CNN architectures (**ResNet-18**, **MobileNetV2**) to evaluate how Winograd scheduling affecting latency, DRAM traffic, and total system energy.

---

## 🚀 Key Features

- **Multi-Rail Power Monitoring**: Real-time hardware sampling of VDD_IN (Total), VDD_CPU, and VDD_GPU via INA3221 sensors at 10ms intervals.
- **Analytical Energy Model**: Deep-dive energy breakdown using pJ-level constants ($E_{MAC}=3.1$ pJ, $E_{DRAM}=220$ pJ).
- **Memory Traffic Validation**: Dynamic `MemoryTracer` that replaces hardcoded estimates with analytical tracing of feature map and weight bytes.
- **Cross-Backend Support**: Comparison between **PyTorch Baseline**, **ONNX Runtime (CPU)**, and **TVM AutoScheduler**.
- **Publication-Ready Artifacts**: Automatic generation of Markdown reports, energy breakdown tables, and research-grade standard library plots.

---

## 🛠 Project Structure

```text
├── run_jetson_benchmark.py   # Primary research pipeline for real hardware
├── run_experiments.py        # Simulation-based benchmarking suite
├── energy_model.py           # Analytical energy constants & logic
├── memory_scheduler.py       # Custom Winograd scheduling algorithms
├── memory_trace.py           # Analytical DRAM traffic estimator
├── visualization.py          # core plotting engine
├── power_monitor.py          # Jetson hardware sensor integration
├── tvm_compiler.py           # ONNX to TVM AutoScheduler compilation
└── generate_plots.py         # Dedicated script for unified graph generation
```

---

## 📊 Methodology

### 1. Energy Model
We utilize a two-component energy model to assess architectural efficiency:
$$E_{total} = E_{MAC} \times N_{MAC} + E_{DRAM} \times N_{DRAM}$$

### 2. Memory Complexity
The framework evaluates four distinct scheduling modes:
- **Baseline**: Standard direct convolution ($O(C_{in} \cdot C_{out})$ DRAM scale).
- **Naive Winograd**: Tile-by-tile processing.
- **Cache-Aware**: Exploits temporal locality by loading input tiles once for all filters.
- **TVM Model**: Heavily optimized ARM NEON backend.

---

## 🏃 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch & Torchvision
- ONNX & ONNX Runtime
- Matplotlib, NumPy, Psutil
- (Optional) Apache TVM for AutoScheduler benchmarks

sudo apt update
sudo apt install -y python3-dev libopenblas-dev libopenmpi-dev
wget https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch, psutil, matplotlib, pandas; print('Environment ready')"


### Installation
```bash
git clone https://github.com/MRvandals4vage/Cache-Aware-Winograd-Scheduling-for-Energy-Efficient-CNN-Inference-on-Edge-CPUs.git
cd Cache-Aware-Winograd-Scheduling-for-Energy-Efficient-CNN-Inference-on-Edge-CPUs
pip install -r requirements.txt
```

### Running Benchmarks
To run the full research suite on a Jetson Nano:
```bash
python3 run_jetson_benchmark.py
```

To run the simulation-based comparison graphs:
```bash
python3 generate_plots.py
```
python3 - <<EOF
import psutil
import matplotlib
import pandas
print("psutil OK")
print("matplotlib OK")
print("pandas OK")
EOF

---

## 📈 Results Preview

The framework produces a comprehensive energy breakdown:

| Model | Strategy | Time (ms) | Energy (mJ) | MACs/J |
| :--- | :--- | :---: | :---: | :---: |
| ResNet18 | Baseline | 8.90 | 10.71 | 7.9e+10 |
| ResNet18 | TVM Model | 7.04 | 10.47 | 9.6e+10 |

Reports and traces are automatically saved to `jetson_energy_breakdown.md` and `scheduler_trace.log`.

---

## 📝 Citation
If you use this framework in your research, please cite:
> "Cache-Aware Winograd Scheduling for Energy-Efficient CNN Inference on Edge CPUs" (2026).
