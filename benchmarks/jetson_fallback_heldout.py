#!/usr/bin/env python3
"""
Task G — Fallback-guard held-out config verification.

Runs the two HELD-OUT configs from the fallback-guard verification table:
  (48,96) and (96,48)

The other 6 rows of the fallback table come from Task C
(jetson_main_microbench.py) — this script only covers the held-out pair.
n=1000, monotonic clock, H'=W'=56, single-threaded.

Writes:
  raw_logs/jetson_fallback_heldout.csv
  summary/jetson_fallback_heldout_summary.csv

MUST be run on the Jetson Nano device.
"""
import os, sys, time, datetime, csv, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fused_winograd_kernel import FusedWinogradKernel
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.locality_scheduler import LocalityScheduler

HELDOUT_CONFIGS = [
    {"c_in": 48, "c_out": 96},
    {"c_in": 96, "c_out": 48},
]


def run_heldout(n_runs=1000, warmup=20, h=56, w=56,
                raw_path="raw_logs/jetson_fallback_heldout.csv",
                summary_path="summary/jetson_fallback_heldout_summary.csv"):
    from scipy import stats as sp_stats
    ts = datetime.datetime.now().isoformat()
    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler()
    scheduler = LocalityScheduler()

    print(f"[G] Jetson Fallback-guard held-out configs")
    print(f"    n={n_runs}, warmup={warmup}, H'={h}, W'={w}")

    all_raw, summary_rows = [], []

    for cfg in HELDOUT_CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        print(f"\n  Config ({c_in},{c_out}):")

        decision = tiler.select_best_tile(c_in, c_out)
        td = decision["selected_tile"]["tile"]
        tile_name = decision["selected_tile"]["name"]
        ws = decision["working_set"]
        l1 = tiler.l1_capacity
        print(f"    Tile selected: {tile_name} (dim={td})")
        print(f"    WS={ws}B, L1={l1}B, fits={'YES' if ws <= l1 else 'NO (fallback)'}")

        tasks = scheduler.generate_tile_tasks(h, w, td, c_in, c_out)
        ordered = scheduler.group_tasks_by_channel_locality(tasks)

        np.random.seed(42)
        input_tile = np.random.randn(c_in, td, td).astype(np.float32)
        U = np.random.randn(c_out, c_in, td, td).astype(np.float32)

        results = {}
        for method, fused in [("baseline_nonfused", False), ("fused", True)]:
            run_fn = kernel.run_fused if fused else kernel.run_non_fused
            # Full warmup
            for _ in range(warmup):
                for _ in ordered:
                    run_fn(input_tile, U)

            lats = []
            for run_id in range(n_runs):
                t0 = time.monotonic()
                for _ in ordered:
                    run_fn(input_tile, U)
                t1 = time.monotonic()
                ms = (t1 - t0) * 1000.0
                lats.append(ms)
                all_raw.append({
                    "timestamp": ts, "c_in": c_in, "c_out": c_out,
                    "h": h, "w": w, "tile": tile_name,
                    "method": method, "threads": 1,
                    "run_id": run_id, "latency_ms": ms,
                    "working_set_bytes": ws, "l1_capacity_bytes": l1,
                    "fallback_triggered": ws > l1,
                })
            results[method] = lats
            mean = float(np.mean(lats))
            ci95 = 1.96 * float(np.std(lats, ddof=1)) / np.sqrt(len(lats))
            print(f"    {method}: mean={mean:.4f}ms CI95=±{ci95:.4f}ms")

        bl = results["baseline_nonfused"]
        fu = results["fused"]
        _, p = sp_stats.ttest_ind(fu, bl, equal_var=False)
        p_str = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"
        improvement = (float(np.mean(bl)) - float(np.mean(fu))) / float(np.mean(bl)) * 100.0

        summary_rows.append({
            "config": f"({c_in},{c_out})", "c_in": c_in, "c_out": c_out,
            "tile": tile_name, "working_set_bytes": ws, "l1_capacity_bytes": l1,
            "fallback_triggered": ws > l1,
            "n": n_runs,
            "baseline_mean_ms": round(float(np.mean(bl)), 4),
            "baseline_ci95_ms": round(1.96 * float(np.std(bl, ddof=1)) / np.sqrt(len(bl)), 4),
            "fused_mean_ms": round(float(np.mean(fu)), 4),
            "fused_ci95_ms": round(1.96 * float(np.std(fu, ddof=1)) / np.sqrt(len(fu)), 4),
            "improvement_pct": round(improvement, 4),
            "p_value": p_str,
            "significance": "SIGNIFICANT" if p < 0.05 else "NOT_SIGNIFICANT",
        })
        print(f"    improvement={improvement:.2f}%  p={p_str}")

    # Write raw
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    with open(raw_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_raw[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(all_raw)
    print(f"\n[G] Raw: {raw_path} ({len(all_raw)} rows)")

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[G] Summary: {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--height", type=int, default=56)
    p.add_argument("--width", type=int, default=56)
    args = p.parse_args()
    run_heldout(n_runs=args.runs, warmup=args.warmup, h=args.height, w=args.width)

if __name__ == "__main__":
    main()
