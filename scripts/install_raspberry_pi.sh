#!/usr/bin/env bash
set -e

echo "================================================="
echo " Cache-Aware Winograd: Raspberry Pi Setup Script"
echo "================================================="

LOG_FILE="pi_install.log"
exec > >(tee -i $LOG_FILE)
exec 2>&1

echo "Checking OS and architecture..."
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "armv7l" ]; then
    echo "Warning: System architecture is $ARCH. This script is intended for Raspberry Pi (ARM)."
fi

echo "Updating package lists..."
sudo apt-get update || echo "Failed to update package lists."

echo "Installing required system dependencies..."
# We need python3, pip, venv, gcc for compiling the C extension, and perf for hardware telemetry
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    linux-perf \
    git \
    libopenblas-dev \
    || echo "Some system packages failed to install, continuing anyway..."

VENV_DIR="venv_winograd"
echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies (NumPy, Pandas, SciPy)..."
# These packages are available and compile/install cleanly on Pi
pip install numpy pandas scipy psutil matplotlib seaborn argparse || {
    echo "ERROR: Failed to install core Python dependencies."
    exit 1
}

echo "Attempting to compile the Fused Winograd C-extension for ARM..."
export CFLAGS="-O3 -march=native -fPIC"
gcc -shared -o ../fused_winograd.so ../fused_winograd.c $CFLAGS || {
    echo "Warning: GCC compilation of fused_winograd.c failed."
    echo "The code will fallback to the pure-numpy non-NEON path."
}

echo "================================================="
echo " Installation Complete!"
echo " A log has been saved to $LOG_FILE."
echo " To start using the benchmark suite, activate the environment:"
echo "     source $VENV_DIR/bin/activate"
echo " Then run:"
echo "     python3 benchmarks/run_all_benchmarks.py --mode micro"
echo "================================================="
