#!/usr/bin/env bash
# ==============================================================================
# scripts/00_detect_board.sh
# Board ID, hardware preconditions (B0), and environment introspection.
# Outputs: results/<board>/env.json
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BOARD="unknown"
MODEL="Unknown"
TEGRA_RELEASE="N/A"
ARCH="$(uname -m)"
OS_NAME="$(uname -s)"

# 1. Architecture Check
if [[ "${ARCH}" != "aarch64" ]]; then
    if [[ "${OS_NAME}" == "Darwin" ]]; then
        echo "[00_detect_board] Running on macOS development host (${ARCH})."
        BOARD="workstation"
        MODEL="macOS Workstation Development Host"
    else
        echo "[00_detect_board] WARNING: Architecture is ${ARCH}, expected aarch64."
        echo "If this is 32-bit ARM (armv7l), STOP: AArch32 has only 16 Q registers, invalidating register audit."
        BOARD="aarch32_unsupported"
        MODEL="Unsupported 32-bit ARM Architecture"
    fi
fi

# 2. Board Detection on Linux
if [[ -f "/proc/device-tree/model" ]]; then
    MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "Unknown")"
    if [[ "${MODEL}" =~ "Raspberry Pi 4" ]]; then
        BOARD="rpi4"
    elif [[ "${MODEL}" =~ "Raspberry Pi" ]]; then
        BOARD="rpi_generic"
    elif [[ "${MODEL}" =~ "Jetson Nano" || "${MODEL}" =~ "tegra" ]]; then
        BOARD="jetson_nano"
    fi
fi

if [[ -f "/etc/nv_tegra_release" ]]; then
    BOARD="jetson_nano"
    TEGRA_RELEASE="$(head -n 1 /etc/nv_tegra_release | tr -d '\n')"
fi

# Allow CLI override: ./00_detect_board.sh --board <name>
if [[ "${1:-}" == "--board" && -n "${2:-}" ]]; then
    BOARD="$2"
fi

echo "[00_detect_board] Identified Board: ${BOARD} (${MODEL})"

OUT_DIR="results/${BOARD}"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/raw" "${OUT_DIR}/quarantine"

# 3. Cache & CPU Telemetry Gathering
NUM_CORES=1
if command -v nproc &>/dev/null; then
    NUM_CORES="$(nproc)"
elif command -v sysctl &>/dev/null; then
    NUM_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 1)"
fi

L1D_BYTES=32768
L2_BYTES=1048576
LINE_SIZE=64

if [[ -d "/sys/devices/system/cpu/cpu0/cache/index0" ]]; then
    L1D_STR="$(cat /sys/devices/system/cpu/cpu0/cache/index0/size 2>/dev/null || echo "32K")"
    if [[ "${L1D_STR}" =~ "K" ]]; then
        L1D_BYTES=$(( ${L1D_STR//K/} * 1024 ))
    fi
    LINE_SIZE="$(cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size 2>/dev/null || echo 64)"
fi

if [[ -d "/sys/devices/system/cpu/cpu0/cache/index2" ]]; then
    L2_STR="$(cat /sys/devices/system/cpu/cpu0/cache/index2/size 2>/dev/null || echo "1024K")"
    if [[ "${L2_STR}" =~ "K" ]]; then
        L2_BYTES=$(( ${L2_STR//K/} * 1024 ))
    elif [[ "${L2_STR}" =~ "M" ]]; then
        L2_BYTES=$(( ${L2_STR//M/} * 1024 * 1024 ))
    fi
fi

GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo "dirty_tree")"
GIT_DIRTY="$(git diff --stat 2>/dev/null | wc -l | tr -d ' ')"

# 4. Write env.json
ENV_JSON="${OUT_DIR}/env.json"
cat <<EOF > "${ENV_JSON}"
{
  "board": "${BOARD}",
  "model": "${MODEL}",
  "os": "${OS_NAME}",
  "arch": "${ARCH}",
  "tegra_release": "${TEGRA_RELEASE}",
  "num_cores": ${NUM_CORES},
  "l1d_size_bytes": ${L1D_BYTES},
  "l2_size_bytes": ${L2_BYTES},
  "line_size_bytes": ${LINE_SIZE},
  "git_hash": "${GIT_HASH}",
  "git_dirty_files": ${GIT_DIRTY},
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "[00_detect_board] Environment written to ${ENV_JSON}"
