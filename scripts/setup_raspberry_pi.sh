#!/bin/bash
set -e

mkdir -p artifacts/logs
LOG_FILE="artifacts/logs/raspberry_pi_setup.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo "Raspberry Pi Benchmark Setup"
echo "Date: $(date)"
echo "========================================"

# Check for apt and Debian/Raspi environment
if ! command -v apt-get &> /dev/null; then
    echo "This script is designed for Debian/Raspberry Pi OS. apt-get not found."
    exit 1
fi

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3-pip \
    python3-dev \
    python3-venv \
    linux-perf \
    libraspberrypi-bin || echo "Some system dependencies could not be fetched. We will continue anyway."

echo "Setting up Python Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing python requirements..."
# Pin to known safe dependencies for ARM64 on Pi
pip install --upgrade pip
pip install setuptools wheel
pip install numpy scipy pandas matplotlib psutil pyyaml tqdm

# Optional: try installing ONNX backend if it's available for this ARM version without too much pain
echo "Attempting to install torch/onnx. This might take a while or fail on some 32-bit platforms..."
pip install torch onnx onnxruntime || echo "Failed to install heavy ML frameworks natively. Using core evaluation paths only."

echo "Compiling fused Winograd C Extension natively..."
gcc -shared -o fused_winograd.so -fPIC -O3 -march=native fused_winograd.c || echo "Native compilation failed. Fallback to Python will be active."

echo "========================================"
echo "Setup complete! You can now run the benchmarks using:"
echo "./scripts/run_raspberry_pi_benchmark.sh"
echo "========================================"
