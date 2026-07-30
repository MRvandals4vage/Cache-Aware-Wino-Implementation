#!/usr/bin/env python3
"""
T2 — Jetson Nano (128,128) re-verification benchmark.

Re-runs baseline direct conv and fused Winograd at (Cin,Cout)=(128,128),
H'=W'=56, single-threaded, n=1000, on Jetson Nano.

Compares against paper's claimed 807.97ms / 860.65ms / -6.52%.
Logs every individual run to raw_logs/jetson_128x128_reverify.csv.

MUST be executed on the actual Jetson Nano device.
"""
import os
import sys
import time
import datetime
import argparse
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fused_winograd_kernel import FusedWinogradKernel
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.locality_scheduler import LocalityScheduler

# Paper's claimed values for comparison
PAPER_BASELINE_MS = 860.65
PAPER_FUSED_MS = 807.97
PAPER_IMPROVEMENT_PCT = -6.52  # negative = regression in paper's convention


def run_reverify(n_runs=1000, warmup=20, c_in=128, c_out=128, h=56, w=56):
    """Run the (128,128) re-verification benchmark."""
    print(f"[T2] Jetson Nano (128,128) Re-verification")
    print(f"     Config: Cin={c_in}, Cout={c_out}, H'={h}, W'={w}")
    print(f"     Runs: {n_runs}, Warmup: {warmup}")
    print(f"     Paper claims: baseline={PAPER_BASELINE_MS}ms, fused={PAPER_FUSED_MS}ms, imp={PAPER_IMPROVEMENT_PCT}%")

    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler()
    scheduler = LocalityScheduler()

    # Get tile decision
    decision = tiler.select_best_tile(c_in, c_out)
    tile_dim = decision["selected_tile"]["tile"]
    tile_name = decision["selected_tile"]["name"]
    print(f"     Tile selected: {tile_name} (dim={tile_dim})")

    # Generate tile tasks for full spatial extent
    tasks = scheduler.generate_tile_tasks(h, w, tile_dim, c_in, c_out)
    ordered_tasks = scheduler.group_tasks_by_channel_locality(tasks)
    print(f"     Total tiles: {len(ordered_tasks)}")

    # Pre-generate data
    np.random.seed(42)
    input_tile = np.random.randn(c_in, tile_dim, tile_dim).astype(np.float32)
    U = np.random.randn(c_out, c_in, tile_dim, tile_dim).astype(np.float32)

    timestamp = datetime.datetime.now().isoformat()
    raw_rows = []

    for method_name, fused in [("baseline_nonfused", False), ("fused", True)]:
        print(f"\n     Running {method_name}...")
        run_func = kernel.run_fused if fused else kernel.run_non_fused

        # Warmup — iterate ALL tasks to exercise the full working set
        for _ in range(warmup):
            for task in ordered_tasks:
                run_func(input_tile, U)

        # Measurement — time.monotonic() maps to CLOCK_MONOTONIC on Linux
        latencies = []
        for run_id in range(n_runs):
            t0 = time.monotonic()
            for task in ordered_tasks:
                run_func(input_tile, U)
            t1 = time.monotonic()
            duration_ms = (t1 - t0) * 1000.0
            latencies.append(duration_ms)

            raw_rows.append({
                "timestamp": timestamp,
                "method": method_name,
                "c_in": c_in,
                "c_out": c_out,
                "h": h,
                "w": w,
                "tile": tile_name,
                "fused": fused,
                "threads": 1,
                "run_id": run_id,
                "latency_ms": duration_ms,
            })

            if (run_id + 1) % 100 == 0:
                print(f"       Run {run_id + 1}/{n_runs} — last: {duration_ms:.2f}ms")

        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies, ddof=1)
        ci95 = 1.96 * std_lat / np.sqrt(n_runs)
        print(f"     {method_name}: mean={mean_lat:.2f}ms, std={std_lat:.2f}ms, CI95=±{ci95:.2f}ms")

    # Write raw log (append with timestamp, never overwrite)
    out_path = os.path.join("raw_logs", "jetson_128x128_reverify.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(raw_rows)

    print(f"\n[T2] Appended {len(raw_rows)} rows to {out_path}")

    # Write summary
    _write_summary(raw_rows, n_runs)


def _write_summary(raw_rows, n_runs):
    """Compute summary statistics and write summary CSV."""
    from scipy import stats as sp_stats

    summary_path = os.path.join("summary", "jetson_128x128_reverify_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Group by method
    methods = {}
    for row in raw_rows:
        m = row["method"]
        if m not in methods:
            methods[m] = []
        methods[m].append(row["latency_ms"])

    baseline_lats = methods.get("baseline_nonfused", [])
    fused_lats = methods.get("fused", [])

    summary_rows = []
    for method_name, lats in methods.items():
        mean_lat = float(np.mean(lats))
        std_lat = float(np.std(lats, ddof=1))
        ci95 = 1.96 * std_lat / np.sqrt(len(lats))

        p_value = "N/A"
        if method_name == "fused" and baseline_lats:
            _, p = sp_stats.ttest_ind(fused_lats, baseline_lats, equal_var=False)
            p_value = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"

        paper_val = PAPER_FUSED_MS if method_name == "fused" else PAPER_BASELINE_MS
        pct_diff = ((mean_lat - paper_val) / paper_val) * 100.0

        summary_rows.append({
            "method": method_name,
            "n": len(lats),
            "mean_ms": round(mean_lat, 4),
            "std_ms": round(std_lat, 4),
            "ci95_ms": round(ci95, 4),
            "p_value": p_value,
            "paper_claimed_ms": paper_val,
            "pct_diff_from_paper": round(pct_diff, 2),
            "match": "MATCH" if abs(pct_diff) < 5.0 else "MISMATCH",
        })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[T2] Summary written to {summary_path}")
    for row in summary_rows:
        print(f"     {row['method']}: paper={row['paper_claimed_ms']}ms -> measured={row['mean_ms']}ms "
              f"({row['pct_diff_from_paper']:+.2f}%) -> {row['match']}")


def main():
    parser = argparse.ArgumentParser(description="T2: Jetson Nano (128,128) re-verification")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--c-in", type=int, default=128)
    parser.add_argument("--c-out", type=int, default=128)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    args = parser.parse_args()

    run_reverify(
        n_runs=args.runs,
        warmup=args.warmup,
        c_in=args.c_in,
        c_out=args.c_out,
        h=args.height,
        w=args.width,
    )


if __name__ == "__main__":
    main()
