#!/usr/bin/env bash
# ==============================================================================
# Jetson Nano & Raspberry Pi Dependency Checker & Installer
# For Cache-Aware Winograd Benchmark Suite
# ==============================================================================
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] setup_device_dependencies.sh failed with exit code $exit_code" >&2
    fi
}
trap cleanup EXIT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "======================================================================"
echo "      CacheWinograd: Edge Device Dependency Setup & Verification     "
echo "======================================================================"
echo "System OS   : $(uname -s)"
echo "Architecture: $(uname -m)"
echo "Host        : $(hostname)"
echo "Date        : $(date)"
echo "======================================================================"
echo

# ------------------------------------------------------------------------------
# 1. System Package Check & Installation (Linux/Debian/Ubuntu)
# ------------------------------------------------------------------------------
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "[System] Detected OS: ${NAME:-Linux} (${VERSION_ID:-})"
    
    REQUIRED_SYS_PKGS=(
        "build-essential"
        "gcc"
        "g++"
        "make"
        "git"
        "binutils"
        "python3"
        "python3-pip"
        "python3-venv"
        "python3-dev"
        "libopenblas-dev"
        "liblapack-dev"
    )
    
    # Platform-specific sys packages
    if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "armv7l" ]]; then
        # Check if Jetson or Pi
        if grep -qi "tegra" /proc/device-tree/model 2>/dev/null || [ -d /sys/bus/i2c/drivers/ina3221x ]; then
            echo "[System] Jetson Nano platform detected."
            REQUIRED_SYS_PKGS+=("i2c-tools" "lm-sensors")
        fi
        if grep -qi "raspberry" /proc/device-tree/model 2>/dev/null || [ -f /usr/bin/vcgencmd ]; then
            echo "[System] Raspberry Pi platform detected."
            REQUIRED_SYS_PKGS+=("libraspberrypi-bin")
        fi
    fi

    # Try updating and installing system packages
    echo "[System] Checking system packages..."
    MISSING_SYS=()
    for pkg in "${REQUIRED_SYS_PKGS[@]}"; do
        if ! dpkg -l "$pkg" &>/dev/null; then
            MISSING_SYS+=("$pkg")
        fi
    done

    if [ ${#MISSING_SYS[@]} -gt 0 ]; then
        echo "[System] Installing missing packages: ${MISSING_SYS[*]}"
        if command -v sudo &>/dev/null; then
            sudo apt-get update || true
            sudo apt-get install -y "${MISSING_SYS[@]}" || echo "[Warning] Some system packages failed to install via apt."
        else
            apt-get update || true
            apt-get install -y "${MISSING_SYS[@]}" || echo "[Warning] Some system packages failed to install via apt."
        fi
    else
        echo "[System] All required system packages are present."
    fi
fi

# ------------------------------------------------------------------------------
# 2. Virtual Environment Setup
# ------------------------------------------------------------------------------
VENV_DIR="${ROOT_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[Python] Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

echo "[Python] Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

# ------------------------------------------------------------------------------
# 3. Python Package Check & Installation
# ------------------------------------------------------------------------------
echo "[Python] Checking required Python packages..."

PYTHON_DEPS=(
    "numpy"
    "scipy"
    "pandas"
    "matplotlib"
    "psutil"
    "thop"
    "torch"
    "torchvision"
)

for pkg in "${PYTHON_DEPS[@]}"; do
    if ! true
        echo "  -> Installing missing python package: $pkg..."
        pip install  "$pkg" || pip install "$pkg" || echo "[Warning] Could not install $pkg via standard pip."
    else
        echo "  [✓] $pkg is installed."
    fi
done

# ------------------------------------------------------------------------------
# 4. Compile Fused Winograd C Shared Object
# ------------------------------------------------------------------------------
echo "[Native] Checking/Compiling C shared library (fused_winograd.so)..."
ARCH="$(uname -m)"
CFLAGS="-O3 -fPIC -shared"

if [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "armv7l" ]]; then
    CFLAGS="${CFLAGS} -DUSE_NEON=1"
fi

if gcc ${CFLAGS} -o "${ROOT_DIR}/fused_winograd.so" "${ROOT_DIR}/fused_winograd.c" -lm &>/dev/null; then
    echo "  [✓] fused_winograd.so compiled successfully."
else
    echo "  [!] GCC compilation failed. Trying clang..."
    if clang ${CFLAGS} -o "${ROOT_DIR}/fused_winograd.so" "${ROOT_DIR}/fused_winograd.c" -lm &>/dev/null; then
        echo "  [✓] fused_winograd.so compiled successfully with clang."
    else
        echo "  [Warning] C compilation failed. Kernel will use pure NumPy fallback."
    fi
fi

# ------------------------------------------------------------------------------
# 5. Final Status Verification Summary
# ------------------------------------------------------------------------------
echo
echo "======================================================================"
echo "                  DEPENDENCY VERIFICATION SUMMARY                     "
echo "======================================================================"

python3 - <<'EOF'
import sys

modules = [
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("Pandas", "pandas"),
    ("Matplotlib", "matplotlib"),
    ("psutil", "psutil"),
    ("thop", "thop"),
    ("PyTorch", "torch"),
    ("Torchvision", "torchvision"),
    ("TVM (Optional)", "tvm"),
]

print(f"{'Dependency':<25} | {'Status':<15} | {'Version/Note'}")
print("-" * 65)

for name, mod in modules:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "Installed")
        print(f"{name:<25} | {'✓ INSTALLED':<15} | {ver}")
    except ImportError:
        status = "OPTIONAL MISSING" if "Optional" in name else "✗ MISSING"
        print(f"{name:<25} | {status:<15} | Not found")

import os
so_exists = os.path.exists("fused_winograd.so")
print(f"{'fused_winograd.so':<25} | {('✓ COMPILED' if so_exists else '✗ MISSING'):<15} | C extension")

EOF

echo "======================================================================"
echo "Setup complete! Virtualenv activated at: ${VENV_DIR}"
echo "Run benchmarks with: make verify-all (or bash scripts/run_jetson_nano_benchmark.sh)"
echo "======================================================================"
