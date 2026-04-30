#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "================================================="
echo "Cache-Aware Winograd: Raspberry Pi Setup Script"
echo "================================================="

LOG_FILE="artifacts/logs/pi_install.log"
mkdir -p artifacts/logs
exec > >(tee -i "${LOG_FILE}")
exec 2>&1

ARCH="$(uname -m)"
if [ "${ARCH}" != "aarch64" ] && [ "${ARCH}" != "armv7l" ]; then
    echo "Warning: System architecture is ${ARCH}. This script is intended for Raspberry Pi."
fi

sudo apt-get update || echo "Failed to update package lists."
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    linux-perf \
    git \
    libopenblas-dev \
    || echo "Some system packages failed to install, continuing anyway..."

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt

echo "Attempting to compile the optional fused Winograd shared object..."
export CFLAGS="-O3 -march=native -fPIC"
gcc -shared -o fused_winograd.so fused_winograd.c ${CFLAGS} || {
    echo "Warning: fused_winograd.c compilation failed."
    echo "The Python benchmark will continue to use the pure NumPy path."
}

echo "================================================="
echo "Installation complete."
echo "Log: ${LOG_FILE}"
echo "Activate with: source venv/bin/activate"
echo "Run with: bash scripts/run_raspberry_pi_benchmark.sh"
echo "================================================="
