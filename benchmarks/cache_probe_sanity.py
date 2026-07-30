#!/usr/bin/env python3
"""
Task A — Cache-probe sanity check.

Reads L1D and L2 cache sizes from sysfs (Linux) or sysctl (macOS).
Compares against paper's expected values: L1D=32KB, L2=2MB on Jetson Nano.
Writes raw_logs/jetson_cacheprobe.csv.

MUST be run on the target device.
"""
import os, sys, csv, datetime, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.runtime_cache_probe import read_sysfs_cache, read_lscpu_cache, build_platform_descriptor

PAPER_L1D_BYTES = 32 * 1024       # 32 KB
PAPER_L2_BYTES  = 2 * 1024 * 1024 # 2 MB


def run_cache_probe(out_path="raw_logs/jetson_cacheprobe.csv"):
    ts = datetime.datetime.now().isoformat()
    print("[A] Cache-probe sanity check")

    platform = build_platform_descriptor()
    l1d = platform.get("l1d_size_bytes")
    l2  = platform.get("l2_size_bytes")
    line = platform.get("line_size_bytes", "unknown")
    source = "sysfs" if os.path.exists("/sys/devices/system/cpu/cpu0/cache") else "lscpu/sysctl"

    print(f"   Source  : {source}")
    print(f"   L1D     : {l1d} B  ({l1d/1024:.1f} KB)" if l1d else "   L1D     : NOT FOUND")
    print(f"   L2      : {l2} B  ({l2/1024/1024:.2f} MB)" if l2 else "   L2      : NOT FOUND")
    print(f"   Line    : {line} B")

    rows = []
    for name, measured, paper in [("L1D", l1d, PAPER_L1D_BYTES), ("L2", l2, PAPER_L2_BYTES)]:
        if measured is None:
            status = "MISSING"
            pct_diff = "N/A"
        else:
            pct_diff = round((measured - paper) / paper * 100, 2)
            status = "MATCH" if abs(pct_diff) < 1.0 else "MISMATCH"
        rows.append({
            "timestamp": ts,
            "cache_level": name,
            "paper_bytes": paper,
            "measured_bytes": measured if measured else "N/A",
            "pct_diff": pct_diff,
            "status": status,
            "source": source,
            "line_size_bytes": line,
        })
        print(f"   {name}: paper={paper}B, measured={measured}B -> {status}"
              f"  ({pct_diff}%)" if measured else f"   {name}: MISSING")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_header = not (os.path.exists(out_path) and os.path.getsize(out_path) > 0)
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[A] Written to {out_path}")

    # Fail loudly if cache sizes can't be read at all
    if l1d is None or l2 is None:
        print("ERROR: Could not read cache sizes. Install linux-tools or check /sys/devices/system/cpu/.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="raw_logs/jetson_cacheprobe.csv")
    args = parser.parse_args()
    run_cache_probe(args.out)

if __name__ == "__main__":
    main()
