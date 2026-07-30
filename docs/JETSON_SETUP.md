# NVIDIA Jetson Setup & Operating Guide

This guide details environment setup, dependency management, hardware performance configuration, and troubleshooting for running **CacheWinograd** on NVIDIA Jetson devices (Jetson Nano, Jetson Xavier, Jetson Orin).

---

## 1. Supported Platforms & Environments

| Device | Target OS / JetPack | Python Version | Architecture |
| :--- | :--- | :--- | :--- |
| **Jetson Nano** | Ubuntu 18.04 (JetPack 4.4 / 4.5 / 4.6) | Python 3.6+ | `aarch64` / `armv7l` |
| **Jetson Xavier (AGX / NX)** | Ubuntu 18.04 / 20.04 (JetPack 4.x / 5.x) | Python 3.6+ | `aarch64` |
| **Jetson Orin (AGX / NX / Nano)**| Ubuntu 20.04 / 22.04 (JetPack 5.x / 6.x) | Python 3.6+ | `aarch64` |
| **Raspberry Pi 4 / 5** | Raspberry Pi OS (Debian 10/11/12) | Python 3.6+ | `aarch64` / `armv7l` |

---

## 2. Quick-Start (Automated Setup)

Execute the self-healing setup script directly on your device:

```bash
# 1. Setup system and Python dependencies
bash scripts/install_jetson_nano.sh

# 2. Run system health check
make doctor

# 3. Backup any existing raw benchmark logs
make preflight-backup

# 4. Run the complete Jetson verification suite
make verify-jetson
```

---

## 3. Required Dependencies

### System Packages (APT)
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gcc g++ make git binutils \
    python3 python3-pip python3-venv python3-dev \
    libopenblas-dev liblapack-dev \
    i2c-tools lm-sensors
```

### Installing `perf` Profiling Tools
`perf` is required for cache-miss and hardware event profiling.

On **Jetson Nano (JetPack 4.x / Ubuntu 18.04)**:
```bash
sudo apt-get install -y linux-tools-generic linux-tools-common
# If kernel version mismatch occurs:
sudo apt-get install -y linux-tools-4.9-tegra
```

**Configuring `perf` Permissions**:
By default, unprivileged users cannot capture hardware performance counters. Enable access by running:
```bash
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```
*(To make this persistent across reboots, add `kernel.perf_event_paranoid = 0` to `/etc/sysctl.conf`)*.

---

## 4. Hardware Performance Tuning

For accurate and reproducible benchmarks on Jetson devices, lock CPU and GPU clocks to max frequency and select high-power mode:

```bash
# 1. Set MAX-N Power Mode (0 is max performance on Jetson Nano)
sudo nvpmodel -m 0

# 2. Lock CPU/GPU clocks to maximum frequency
sudo jetson_clocks

# 3. Verify CPU scaling governor is set to performance
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## 5. Repository Health Check & Diagnostic Commands

Use the integrated `doctor.py` tool to verify your environment at any time:

```bash
# Run repository health check
make doctor

# Auto-fix missing directories and compile C extensions
python3 scripts/doctor.py --fix

# Clean temporary build files without touching raw_logs or summary
make clean-safe

# Fully rebuild environment and C shared object from scratch
make reset
```

---

## 6. Common Issues & Troubleshooting

### Issue 1: `fused_winograd.so` compilation failure
- **Symptom**: `WARNING: C compilation failed. Pure NumPy fallback will be used.`
- **Fix**: Re-run `make reset` or manually compile:
  ```bash
  gcc -O3 -fPIC -shared -DUSE_NEON=1 -o fused_winograd.so fused_winograd.c -lm
  ```

### Issue 2: `perf` Permission Denied
- **Symptom**: `perf stat` fails with `Permission denied` or `sysctl kernel.perf_event_paranoid`.
- **Fix**: Run `echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`.

### Issue 3: Missing `raw_logs/` or `summary/` files in `verify-all`
- **Symptom**: `make verify-all` exits non-zero with missing raw logs.
- **Fix**: Run `make verify-jetson` on Jetson Nano to execute all required benchmark tasks first.

---

## 7. Execution Workflow Summary

```bash
make doctor          # Check system health
make preflight-backup # Rotate prior raw data
make verify-jetson   # Run Jetson experiments (Tasks A-K)
make verify-all      # Compute derived tables, paper plots, & reproducibility report
```
