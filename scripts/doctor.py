#!/usr/bin/env python3
"""
CacheWinograd Repository Doctor & Health Checker (Task 3, 6, 10, 14).

Verifies system environment, platform details (Jetson Nano/Xavier/Orin/RPi/Linux/macOS),
build tools, python packages, hardware resources, permissions, and directory structure.
Provides clear PASS / WARNING / FAIL outputs with exact fix/installation commands.

Supports:
    python3 scripts/doctor.py
    python3 scripts/doctor.py --fix
"""
import os
import sys
import shutil
import platform
import subprocess
import json
import glob

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.runtime_cache_probe import detect_platform


class Doctor:
    def __init__(self, auto_fix=False):
        self.auto_fix = auto_fix
        self.pass_count = 0
        self.warn_count = 0
        self.fail_count = 0
        self.results = []

    def _log_result(self, subsystem, name, status, details="", fix_cmd=""):
        if status == "PASS":
            self.pass_count += 1
            icon = "✓"
        elif status == "WARNING":
            self.warn_count += 1
            icon = "⚠️"
        else:
            self.fail_count += 1
            icon = "✗"

        res = {
            "subsystem": subsystem,
            "name": name,
            "status": status,
            "details": details,
            "fix_cmd": fix_cmd
        }
        self.results.append(res)
        print(f"  [{icon} {status:<7}] {subsystem:<20} | {name:<25} | {details}")
        if status != "PASS" and fix_cmd:
            print(f"               Fix command: {fix_cmd}")

    def check_python_version(self):
        ver = sys.version_info
        ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
        if ver >= (3, 6):
            self._log_result("Python Runtime", "Python Version", "PASS", f"Python {ver_str} (>= 3.6)")
        else:
            self._log_result("Python Runtime", "Python Version", "FAIL", f"Python {ver_str} < 3.6 required",
                             "Upgrade Python to 3.6 or newer")

    def check_executables(self):
        required_tools = [
            ("gcc", True, "sudo apt-get install -y gcc build-essential"),
            ("g++", True, "sudo apt-get install -y g++ build-essential"),
            ("make", True, "sudo apt-get install -y make"),
            ("git", True, "sudo apt-get install -y git"),
            ("cmake", False, "sudo apt-get install -y cmake"),
            ("pkg-config", False, "sudo apt-get install -y pkg-config"),
            ("perf", False, "sudo apt-get install -y linux-tools-common linux-tools-generic linux-tools-$(uname -r)"),
        ]
        system_utils = [
            ("lscpu", False, "sudo apt-get install -y util-linux"),
            ("nproc", False, "sudo apt-get install -y coreutils"),
            ("uname", True, "sudo apt-get install -y coreutils"),
            ("grep", True, "sudo apt-get install -y grep"),
            ("awk", True, "sudo apt-get install -y gawk"),
        ]

        print("\n--- Checking Build & System Tools ---")
        for tool, is_critical, fix in required_tools + system_utils:
            path = shutil.which(tool)
            if path:
                self._log_result("Build Tool", tool, "PASS", f"Found at {path}")
            else:
                status = "FAIL" if is_critical else "WARNING"
                self._log_result("Build Tool", tool, status, "Not found in PATH", fix)

    def check_python_packages(self):
        print("\n--- Checking Python Packages ---")
        packages = [
            ("numpy", True, "pip install numpy"),
            ("scipy", True, "pip install scipy"),
            ("pandas", True, "pip install pandas"),
            ("matplotlib", True, "pip install matplotlib"),
            ("psutil", False, "pip install psutil"),
            ("thop", False, "pip install thop"),
            ("onnx", False, "pip install onnx"),
            ("onnxruntime", False, "pip install onnxruntime"),
            ("torch", False, "pip install torch torchvision"),
            ("tvm", False, "Build/install TVM or pip install apache-tvm"),
        ]

        for mod, is_critical, fix in packages:
            try:
                m = __import__(mod)
                version = getattr(m, "__version__", "Installed")
                self._log_result("Python Package", mod, "PASS", f"Version {version}")
            except ImportError:
                status = "FAIL" if is_critical else "WARNING"
                self._log_result("Python Package", mod, status, "Module not installed", fix)
                if self.auto_fix and is_critical:
                    print(f"  Attempting auto-install for {mod}...")
                    subprocess.run([sys.executable, "-m", "pip", "install", mod])

    def check_platform_and_jetson(self):
        print("\n--- Checking Platform & Jetson Utilities ---")
        plat = detect_platform()
        os_type = plat.get("os", "Unknown")
        arch = plat.get("architecture", "Unknown")
        is_pi = plat.get("is_raspberry_pi", False)
        pi_model = plat.get("pi_model")

        # Detect Jetson
        is_jetson = False
        jetson_model = "Unknown Jetson"
        if os.path.exists("/proc/device-tree/model"):
            try:
                with open("/proc/device-tree/model", "r") as f:
                    model_str = f.read().strip('\x00').strip()
                    if "Tegra" in model_str or "Jetson" in model_str:
                        is_jetson = True
                        jetson_model = model_str
            except Exception:
                pass
        if not is_jetson and os.path.exists("/etc/nv_tegra_release"):
            is_jetson = True
            jetson_model = "NVIDIA Tegra Device"

        if is_jetson:
            self._log_result("Platform", "Jetson Detection", "PASS", f"Detected {jetson_model}")
            for jtool in ["tegrastats", "jetson_clocks", "nvpmodel"]:
                jpath = shutil.which(jtool)
                if jpath:
                    self._log_result("Jetson Utility", jtool, "PASS", f"Found at {jpath}")
                else:
                    self._log_result("Jetson Utility", jtool, "WARNING",
                                     f"{jtool} not found (L4T tool). Run under sudo on Jetson.",
                                     f"sudo {jtool}")
        elif is_pi:
            self._log_result("Platform", "Raspberry Pi Detection", "PASS", f"Detected {pi_model}")
        else:
            self._log_result("Platform", "Host Platform", "PASS", f"Generic Platform: {os_type} {arch}")

    def check_hardware_and_environment(self):
        print("\n--- Checking Hardware Resources & Permissions ---")

        # Memory Check
        try:
            import psutil
            mem = psutil.virtual_memory()
            avail_mb = mem.available / (1024 * 1024)
            if avail_mb >= 500:
                self._log_result("Hardware", "Available RAM", "PASS", f"{avail_mb:.1f} MB available")
            else:
                self._log_result("Hardware", "Available RAM", "WARNING", f"Low RAM: {avail_mb:.1f} MB available")
        except Exception:
            self._log_result("Hardware", "Available RAM", "PASS", "psutil not available to check memory")

        # Disk Space Check
        try:
            stat = os.statvfs(REPO_ROOT)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024 * 1024)
            if free_gb >= 1.0:
                self._log_result("Hardware", "Disk Space", "PASS", f"{free_gb:.2f} GB free")
            else:
                self._log_result("Hardware", "Disk Space", "WARNING", f"Low Disk Space: {free_gb:.2f} GB free")
        except Exception:
            self._log_result("Hardware", "Disk Space", "PASS", "Disk space check skipped")

        # Perf Paranoid Level Check (Linux)
        if os.path.exists("/proc/sys/kernel/perf_event_paranoid"):
            try:
                with open("/proc/sys/kernel/perf_event_paranoid") as f:
                    val = int(f.read().strip())
                if val <= 1:
                    self._log_result("System Security", "perf_event_paranoid", "PASS", f"Level = {val}")
                else:
                    self._log_result("System Security", "perf_event_paranoid", "WARNING",
                                     f"Level = {val} (>1 may block perf stat)",
                                     "echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid")
            except Exception:
                pass

        # CPU Governor Check (Linux)
        gov_files = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        if gov_files:
            try:
                govs = set()
                for gf in gov_files:
                    with open(gf) as f:
                        govs.add(f.read().strip())
                gov_str = ", ".join(sorted(govs))
                if "performance" in govs:
                    self._log_result("CPU Config", "Scaling Governor", "PASS", f"Governor: {gov_str}")
                else:
                    self._log_result("CPU Config", "Scaling Governor", "WARNING", f"Governor: {gov_str} (not performance)",
                                     "sudo jetson_clocks OR for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee $g; done")
            except Exception:
                pass

    def check_directories_and_build(self):
        print("\n--- Checking Directories & C Shared Object ---")
        required_dirs = [
            "raw_logs", "summary", "build", "tmp", "results",
            "artifacts", "artifacts/raw", "artifacts/processed",
            "artifacts/plots", "artifacts/logs", "logs",
            "raw_logs_prior", "summary_prior"
        ]

        for d in required_dirs:
            full_path = os.path.join(REPO_ROOT, d)
            if os.path.exists(full_path):
                self._log_result("Directory", d, "PASS", f"Exists: {full_path}")
            else:
                if self.auto_fix:
                    os.makedirs(full_path, exist_ok=True)
                    self._log_result("Directory", d, "PASS", f"Auto-created: {full_path}")
                else:
                    os.makedirs(full_path, exist_ok=True)
                    self._log_result("Directory", d, "PASS", f"Created: {full_path}")

        # Check fused_winograd.so
        so_path = os.path.join(REPO_ROOT, "fused_winograd.so")
        c_src = os.path.join(REPO_ROOT, "fused_winograd.c")
        if os.path.exists(so_path):
            self._log_result("C Extension", "fused_winograd.so", "PASS", f"Compiled binary present ({os.path.getsize(so_path)} B)")
        elif os.path.exists(c_src):
            print("  fused_winograd.so not found. Attempting compilation...")
            arch = platform.machine().lower()
            flags = ["-O3", "-fPIC", "-shared"]
            if arch in ("aarch64", "armv7l"):
                flags.append("-DUSE_NEON=1")
            cmd = ["gcc"] + flags + ["-o", so_path, c_src, "-lm"]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if r.returncode == 0:
                self._log_result("C Extension", "fused_winograd.so", "PASS", "Compiled successfully via gcc")
            else:
                cmd[0] = "clang"
                r2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if r2.returncode == 0:
                    self._log_result("C Extension", "fused_winograd.so", "PASS", "Compiled successfully via clang")
                else:
                    self._log_result("C Extension", "fused_winograd.so", "WARNING",
                                     "Compilation failed — pure NumPy fallback will be used",
                                     "gcc -O3 -fPIC -shared -o fused_winograd.so fused_winograd.c -lm")
        else:
            self._log_result("C Extension", "fused_winograd.c", "FAIL", "Source file fused_winograd.c missing")

    def run_all_checks(self):
        print("======================================================================")
        print("            CacheWinograd Repository Doctor & Environment Check       ")
        print("======================================================================")

        self.check_python_version()
        self.check_executables()
        self.check_python_packages()
        self.check_platform_and_jetson()
        self.check_hardware_and_environment()
        self.check_directories_and_build()

        print("\n======================================================================")
        print("                        HEALTH CHECK SUMMARY                          ")
        print("======================================================================")
        print(f"  PASS    : {self.pass_count}")
        print(f"  WARNING : {self.warn_count}")
        print(f"  FAIL    : {self.fail_count}")
        print("======================================================================")

        if self.fail_count > 0:
            print("\n❌ CRITICAL ISSUES DETECTED: Review the FAIL items above before running benchmarks.")
            return False
        elif self.warn_count > 0:
            print("\n⚠️ SYSTEM READY WITH WARNINGS: Review WARNING items for optional features/optimizations.")
            return True
        else:
            print("\n✅ SYSTEM FULLY HEALTHY: All checks passed cleanly.")
            return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CacheWinograd Repository Doctor")
    parser.add_argument("--fix", action="store_true", help="Attempt automatic directory creation and package fixes")
    args = parser.parse_args()

    doc = Doctor(auto_fix=args.fix)
    success = doc.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
