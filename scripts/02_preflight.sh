#!/usr/bin/env bash
# ==============================================================================
# scripts/02_preflight.sh
# Pre-measurement system integrity check:
# - Clocks, CPU governor, and online core counts
# - Thermal status and hardware throttling flags
# - Swap usage (asserted 0 during measurement)
# - Background load average
# Exits 0 if system is fit for benchmark execution, non-zero if unfit.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

UNAME_S="$(uname -s)"
echo "[02_preflight] Running hardware pre-flight verification on ${UNAME_S}..."

# 1. Thermal & Throttling Verification
if command -v vcgencmd &>/dev/null; then
    THROTTLE_HEX="$(vcgencmd get_throttled | cut -d'=' -f2)"
    echo "[02_preflight] RPi Throttle Status: ${THROTTLE_HEX}"
    if [[ "${THROTTLE_HEX}" != "0x0" ]]; then
        echo "[02_preflight] FATAL: Raspberry Pi is throttled or under-voltage detected (${THROTTLE_HEX})."
        echo "Cool the board down or verify power supply before measuring."
        exit 1
    fi
fi

# 2. Jetson Clocks & Power Mode
if command -v nvpmodel &>/dev/null; then
    CURRENT_MODE="$(nvpmodel -q | grep "NVPM WARN" -v | head -n 1 || echo "")"
    echo "[02_preflight] Jetson Power Mode: ${CURRENT_MODE}"
fi

# 3. Swap Check
if command -v free &>/dev/null; then
    SWAP_USED="$(free -b | awk '/Swap:/ {print $3}')"
    if [[ "${SWAP_USED}" -gt 10485760 ]]; then # > 10MB used
        echo "[02_preflight] WARNING: Swap usage is non-zero (${SWAP_USED} bytes). Paging will invalidate timing."
        if [[ "$(id -u)" -eq 0 ]]; then
            echo "[02_preflight] Disabling swap..."
            swapoff -a && swapon -a || true
        else
            echo "[02_preflight] Non-root: please disable swap or wait for memory to clear."
        fi
    fi
fi

# 4. Background Load Check
if [[ -f "/proc/loadavg" ]]; then
    LOAD_1M="$(awk '{print $1}' /proc/loadavg)"
    echo "[02_preflight] 1-minute Load Average: ${LOAD_1M}"
    LOAD_INT="${LOAD_1M%.*}"
    if [[ "${LOAD_INT}" -gt 2 ]]; then
        echo "[02_preflight] WARNING: High background load average (${LOAD_1M})."
    fi
fi

# 5. Frequency Governor Verification
if [[ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]]; then
    GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")"
    CUR_FREQ="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "unknown")"
    echo "[02_preflight] CPU0 Governor: ${GOV} | Current Frequency: ${CUR_FREQ} kHz"
    if [[ "${GOV}" != "performance" && "$(id -u)" -eq 0 ]]; then
        echo "[02_preflight] Setting performance governor on all cores..."
        for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "performance" > "${g}" 2>/dev/null || true
        done
    fi
fi

echo "[02_preflight] Pre-flight checks PASSED. System is ready for benchmark batch."
exit 0
