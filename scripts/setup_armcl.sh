#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_armcl.sh
#
# Arm Compute Library (ArmCL) Environment & Harness Setup for Jetson Nano / ARM64
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "======================================================================"
echo "          Arm Compute Library (ArmCL) Setup & Verification           "
echo "======================================================================"

# 1. Check if system packages are available via apt
if [ -f /etc/os-release ]; then
    echo "[ArmCL] Checking apt for Arm Compute Library packages..."
    if command -v apt-get &>/dev/null; then
        if apt-cache search libarm-compute-dev | grep -q libarm-compute-dev; then
            echo "[ArmCL] Found libarm-compute-dev in apt repository. Installing..."
            if command -v sudo &>/dev/null; then
                sudo apt-get update || true
                sudo apt-get install -y libarm-compute-dev libarm-compute-core-dev || true
            else
                apt-get update || true
                apt-get install -y libarm-compute-dev libarm-compute-core-dev || true
            fi
        fi
    fi
fi

# 2. Check for existing headers/libraries
ARMCL_FOUND=0
for inc_dir in "/usr/include" "/usr/local/include" "/opt/arm_compute/include" "${ROOT_DIR}/third_party/armcl/include"; do
    if [ -f "${inc_dir}/arm_compute/runtime/NEON/NEFunctions.h" ]; then
        echo "[ArmCL] Detected ArmCL headers at ${inc_dir}"
        ARMCL_FOUND=1
        break
    fi
done

# 3. Build the ArmCL benchmark executable
mkdir -p "${ROOT_DIR}/bin"
echo "[ArmCL] Building ArmCL benchmark harness..."
bash "${ROOT_DIR}/tools/run_armcl.sh" 16 16 14 14 3 2 1

echo
echo "======================================================================"
echo "ArmCL setup verified. Default runner: ${ROOT_DIR}/tools/run_armcl.sh"
echo "======================================================================"
