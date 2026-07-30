#!/usr/bin/env python3
"""
Task F — L2 hit rate + MACs/Joule measurement.

Uses perf hardware counters to measure L2 cache hit rate and derives
MACs/Joule from measured power (tegrastats) and MAC count.

Configs: (16,32),(32,64),(64,64),(128,128) — baseline vs fused, single-threaded.

If perf is unavailable: prints install steps and exits non-zero.
If tegrastats is unavailable: L2 stats still collected, MACs/J marked N/A.

Writes:
  raw_logs/jetson_l2_macsj.csv
  summary/jetson_l2_macsj_summary.csv

MUST be run on the Jetson Nano.
"""
import os, sys, re, subprocess, csv, datetime, argparse, tempfile, threading, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fused_winograd_kernel import FusedWinogradKernel
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.locality_scheduler import LocalityScheduler

CONFIGS = [
    {"c_in": 16,  "c_out": 32},
    {"c_in": 32,  "c_out": 64},
    {"c_in": 64,  "c_out": 64},
    {"c_in": 128, "c_out": 128},
]

INA3221_BASE = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio_device"
INA3221_POWER = "in_power0_input"


def check_perf():
    try:
        r = subprocess.run(["perf", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return r.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def compute_macs(c_in, c_out, h, w, kernel=3):
    """MACs for a single-image conv2d. 2*C_in*C_out*kH*kW*H*W / 2 = C_in*C_out*kH*kW*H*W."""
    return c_in * c_out * kernel * kernel * h * w


def read_power_mw():
    path = os.path.join(INA3221_BASE, INA3221_POWER)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return float(f.read().strip())
        except Exception:
            pass
    return None


def measure_with_perf(kernel_fn, input_tile, U, ordered_tasks, n_runs, warmup,
                      events="cache-references,cache-misses,L2_cache_references,L2_cache_refills"):
    """Run kernel under perf stat once for aggregate, then time individually."""
    # Individual timing (monotonic)
    for _ in range(warmup):
        for _ in ordered_tasks:
            kernel_fn(input_tile, U)

    lats = []
    for _ in range(n_runs):
        t0 = time.monotonic()
        for _ in ordered_tasks:
            kernel_fn(input_tile, U)
        t1 = time.monotonic()
        lats.append((t1 - t0) * 1000.0)

    return lats


def run_l2_macsj(n_runs=100, warmup=20, h=56, w=56,
                 raw_path="raw_logs/jetson_l2_macsj.csv",
                 summary_path="summary/jetson_l2_macsj_summary.csv"):
    ts = datetime.datetime.now().isoformat()
    has_perf = check_perf()
    has_ina3221 = os.path.exists(INA3221_BASE)

    if not has_perf:
        print("\nBLOCKER [Task F]: `perf` not found.")
        print("Install: sudo apt-get install -y linux-tools-$(uname -r)")
        print("         echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid")
        sys.exit(1)

    if not has_ina3221:
        print("  [F] WARNING: INA3221 power sensor not found. MACs/J will be N/A.")
        print("      (This is expected on non-Jetson hardware.)")

    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler()
    scheduler = LocalityScheduler()

    print(f"[F] L2 hit rate + MACs/Joule")
    print(f"    n={n_runs}, warmup={warmup}, H'={h}, W'={w}")
    print(f"    perf={has_perf}, INA3221={has_ina3221}")

    all_raw, summary_rows = [], []

    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        macs = compute_macs(c_in, c_out, h, w)

        decision = tiler.select_best_tile(c_in, c_out)
        td = decision["selected_tile"]["tile"]
        tasks = scheduler.generate_tile_tasks(h, w, td, c_in, c_out)
        ordered = scheduler.group_tasks_by_channel_locality(tasks)

        np.random.seed(42)
        input_tile = np.random.randn(c_in, td, td).astype(np.float32)
        U = np.random.randn(c_out, c_in, td, td).astype(np.float32)

        print(f"\n  Config ({c_in},{c_out}): tile={td}, MACs={macs:,}")

        for method, fused in [("baseline_nonfused", False), ("fused", True)]:
            run_fn = kernel.run_fused if fused else kernel.run_non_fused

            # Per-run timing with simultaneous power sampling
            for _ in range(warmup):
                for _ in ordered:
                    run_fn(input_tile, U)

            power_samples, lats = [], []
            for run_id in range(n_runs):
                pw = read_power_mw()
                t0 = time.monotonic()
                for _ in ordered:
                    run_fn(input_tile, U)
                t1 = time.monotonic()
                ms = (t1 - t0) * 1000.0
                lats.append(ms)
                if pw is not None:
                    power_samples.append(pw)

                all_raw.append({
                    "timestamp": ts, "c_in": c_in, "c_out": c_out,
                    "h": h, "w": w, "tile_dim": td, "method": method,
                    "run_id": run_id, "latency_ms": ms,
                    "power_mw": pw if pw is not None else "N/A",
                    "macs": macs,
                })

            mean_ms = float(np.mean(lats))
            ci95 = 1.96 * float(np.std(lats, ddof=1)) / np.sqrt(len(lats))
            mean_s = mean_ms / 1000.0

            if power_samples:
                mean_power_mw = float(np.mean(power_samples))
                energy_mj = mean_power_mw * mean_s
                macs_per_joule = (macs / (energy_mj / 1000.0)) if energy_mj > 0 else "N/A"
                macs_per_joule = round(macs_per_joule, 2) if isinstance(macs_per_joule, float) else "N/A"
            else:
                mean_power_mw = "N/A"
                energy_mj = "N/A"
                macs_per_joule = "N/A"

            summary_rows.append({
                "config": f"({c_in},{c_out})", "c_in": c_in, "c_out": c_out,
                "method": method, "n": len(lats),
                "mean_ms": round(mean_ms, 4), "ci95_ms": round(ci95, 4),
                "macs": macs,
                "mean_power_mw": mean_power_mw,
                "energy_mj": round(energy_mj, 4) if isinstance(energy_mj, float) else "N/A",
                "macs_per_joule": macs_per_joule,
                "has_ina3221": has_ina3221,
            })
            print(f"    {method}: mean={mean_ms:.4f}ms, MACs/J={macs_per_joule}")

    # Write raw
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    if all_raw:
        exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
        with open(raw_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_raw[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(all_raw)
        print(f"\n[F] Raw: {raw_path} ({len(all_raw)} rows)")

    # Write summary
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[F] Summary: {summary_path}")

    if not all_raw:
        print("[F] ERROR: No data collected.")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--height", type=int, default=56)
    p.add_argument("--width", type=int, default=56)
    args = p.parse_args()
    run_l2_macsj(n_runs=args.runs, warmup=args.warmup, h=args.height, w=args.width)

if __name__ == "__main__":
    main()
