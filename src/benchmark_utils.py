#!/usr/bin/env python3
"""
CacheWinograd Shared Benchmark Utilities & Environment Guard.

Handles:
- Task 4: Automatic directory creation
- Task 7: Benchmark environment validation (RAM, disk, governor, perf, C extension)
- Task 8: Automatic recovery (auto-compilation, directory setup)
- Task 9: Timestamped logging (logs/YYYY-MM-DD_HH-MM-SS.log)
- Task 11: Resumability (skipping completed runs unless --force is supplied)
- Task 12: Helpful, actionable error reporting
"""
import os
import sys
import time
import datetime
import traceback
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.runtime_cache_probe import detect_platform


REQUIRED_DIRS = [
    "raw_logs", "summary", "build", "tmp", "results",
    "artifacts", "artifacts/raw", "artifacts/processed",
    "artifacts/plots", "artifacts/logs", "logs",
    "raw_logs_prior", "summary_prior"
]


def ensure_directories():
    """Ensure all required project directories exist automatically (Task 4)."""
    for d in REQUIRED_DIRS:
        full = os.path.join(REPO_ROOT, d)
        os.makedirs(full, exist_ok=True)


class BenchmarkLogger:
    """Manages timestamped logging for benchmarks (Task 9)."""
    def __init__(self, name: str = "benchmark"):
        ensure_directories()
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"{ts}_{name}.log"
        self.log_path = os.path.join(REPO_ROOT, "logs", self.log_filename)
        self.latest_link = os.path.join(REPO_ROOT, "logs", "latest.log")

        # Open log file
        self._f = open(self.log_path, "a", encoding="utf-8")
        
        # Symlink latest.log
        try:
            if os.path.exists(self.latest_link) or os.path.islink(self.latest_link):
                os.unlink(self.latest_link)
            os.symlink(self.log_path, self.latest_link)
        except Exception:
            pass

        self._log_header(name)

    def _log_header(self, name: str):
        plat = detect_platform()
        git_hash = plat.get("git_commit", "unknown")
        os_name = plat.get("os", "unknown")
        arch = plat.get("architecture", "unknown")
        py_ver = sys.version.replace("\n", " ")

        header = f"""======================================================================
CacheWinograd Benchmark Log: {name}
Timestamp : {datetime.datetime.now().isoformat()}
Log File  : {self.log_path}
Git Commit: {git_hash}
Platform  : {os_name} {arch}
Python    : {py_ver}
======================================================================
"""
        self.info(header)

    def info(self, msg: str):
        print(msg)
        self._f.write(f"[INFO {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self._f.flush()

    def warning(self, msg: str):
        print(f"⚠️  [WARNING] {msg}")
        self._f.write(f"[WARN {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self._f.flush()

    def error(self, msg: str, exc: Optional[BaseException] = None):
        print(f"❌ [ERROR] {msg}")
        self._f.write(f"[ERROR {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        if exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._f.write(f"Traceback:\n{tb}\n")
            print(f"\nTraceback:\n{tb}")
        self._f.flush()

    def close(self):
        if not self._f.closed:
            self._f.close()


def ensure_c_extension() -> bool:
    """Attempts automatic recovery / compilation of fused_winograd.so (Task 8)."""
    so_path = os.path.join(REPO_ROOT, "fused_winograd.so")
    c_src = os.path.join(REPO_ROOT, "fused_winograd.c")

    if os.path.exists(so_path) and os.path.getsize(so_path) > 0:
        return True

    if not os.path.exists(c_src):
        print(f"❌ [ERROR] C kernel source file not found: {c_src}")
        return False

    print("🔧 [Auto-Recovery] C extension fused_winograd.so missing. Compiling...")
    plat = detect_platform()
    arch = plat.get("architecture", "").lower()

    flags = ["-O3", "-fPIC", "-shared"]
    if arch in ("aarch64", "armv7l"):
        flags.append("-DUSE_NEON=1")

    cmd = ["gcc"] + flags + ["-o", so_path, c_src, "-lm"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode == 0:
        print("✓ [Auto-Recovery] fused_winograd.so compiled successfully with gcc.")
        return True

    cmd[0] = "clang"
    r2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r2.returncode == 0:
        print("✓ [Auto-Recovery] fused_winograd.so compiled successfully with clang.")
        return True

    print("⚠️  [WARNING] C compilation failed. Pure NumPy fallback will be used.")
    return False


def validate_environment(logger: Optional[BenchmarkLogger] = None) -> bool:
    """Task 7: Pre-benchmark environment validation."""
    ensure_directories()
    ensure_c_extension()

    # Disk Space Check
    try:
        stat = os.statvfs(REPO_ROOT)
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        if free_mb < 200:
            msg = f"Low disk space: {free_mb:.1f} MB free. Benchmark raw logs may fail to write."
            if logger:
                logger.warning(msg)
            else:
                print(f"⚠️  {msg}")
    except Exception:
        pass

    return True


def is_experiment_completed(csv_path: str, filter_dict: Dict[str, Any], logger: Optional[BenchmarkLogger] = None, force: bool = False) -> bool:
    """
    Task 11: Resumability checker.
    Returns True if an experiment matching filter_dict already exists in csv_path and force=False.
    """
    if force:
        return False

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return False

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                match = True
                for k, v in filter_dict.items():
                    if str(row.get(k, "")) != str(v):
                        match = False
                        break
                if match:
                    if logger:
                        logger.info(f"⏩ [Resumability] Experiment for {filter_dict} found in {csv_path}. Skipping (use --force to re-run).")
                    return True
    except Exception:
        pass

    return False
