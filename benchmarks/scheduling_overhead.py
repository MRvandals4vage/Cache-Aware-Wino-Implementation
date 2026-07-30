#!/usr/bin/env python3
"""
Task H — Scheduling-overhead direct timing.

Measures Phase 1 (sysfs cache-probe) and Phase 2 (tile-selection arithmetic)
wall-clock time via instrumented calls — NOT a theoretical/architectural estimate.

Paper claims: ~20μs total, ~5ns per-tile.
n=1000 per phase. Monotonic clock.

Writes:
  raw_logs/jetson_scheduling_overhead.csv
  summary/jetson_scheduling_overhead_summary.csv

Runs on any Linux device (sysfs) or macOS (sysctl fallback).
"""
import os, sys, time, datetime, csv, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.runtime_cache_probe import read_sysfs_cache, read_lscpu_cache
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.locality_scheduler import LocalityScheduler

# Paper claims
PAPER_TOTAL_OVERHEAD_US = 20.0   # ~20 μs
PAPER_PER_TILE_NS       = 5.0    # ~5 ns

CONFIGS_FOR_PER_TILE = [
    {"c_in": 64,  "c_out": 64,  "h": 56, "w": 56},
    {"c_in": 128, "c_out": 128, "h": 56, "w": 56},
]


def measure_phase1(n=1000):
    """Phase 1: sysfs cache probe latency."""
    lats_us = []
    for _ in range(n):
        t0 = time.monotonic_ns()
        # Replicate what the probe does: read sysfs
        _ = read_sysfs_cache() or read_lscpu_cache()
        t1 = time.monotonic_ns()
        lats_us.append((t1 - t0) / 1000.0)
    return lats_us


def measure_phase2(tiler, c_in, c_out, n=1000):
    """Phase 2: tile-selection arithmetic latency (pure Python)."""
    lats_us = []
    for _ in range(n):
        t0 = time.monotonic_ns()
        _ = tiler.select_best_tile(c_in, c_out)
        t1 = time.monotonic_ns()
        lats_us.append((t1 - t0) / 1000.0)
    return lats_us


def measure_per_tile(scheduler, c_in, c_out, h, w, tile_dim, n=1000):
    """Per-tile scheduling cost: generate + group tasks, divided by tile count."""
    tasks = scheduler.generate_tile_tasks(h, w, tile_dim, c_in, c_out)
    n_tiles = len(tasks)
    lats_ns = []
    for _ in range(n):
        t0 = time.monotonic_ns()
        _ = scheduler.generate_tile_tasks(h, w, tile_dim, c_in, c_out)
        _ = scheduler.group_tasks_by_channel_locality(_)
        t1 = time.monotonic_ns()
        total_ns = t1 - t0
        per_tile_ns = total_ns / max(n_tiles, 1)
        lats_ns.append(per_tile_ns)
    return lats_ns, n_tiles


def run_overhead(n=1000,
                 raw_path="raw_logs/jetson_scheduling_overhead.csv",
                 summary_path="summary/jetson_scheduling_overhead_summary.csv"):
    ts = datetime.datetime.now().isoformat()
    tiler = CacheAdaptiveAutotiler()
    scheduler = LocalityScheduler()

    print(f"[H] Scheduling overhead — direct timing (n={n})")
    print(f"    Paper claims: total~{PAPER_TOTAL_OVERHEAD_US}μs, per-tile~{PAPER_PER_TILE_NS}ns")

    all_raw, summary_rows = [], []

    # --- Phase 1: sysfs probe ---
    print("\n  Phase 1: sysfs cache-probe ...")
    p1_lats = measure_phase1(n)
    p1_mean = float(np.mean(p1_lats))
    p1_ci95 = 1.96 * float(np.std(p1_lats, ddof=1)) / np.sqrt(len(p1_lats))
    print(f"    mean={p1_mean:.2f}μs CI95=±{p1_ci95:.2f}μs")

    for run_id, v in enumerate(p1_lats):
        all_raw.append({
            "timestamp": ts, "phase": "sysfs_probe", "config": "N/A",
            "c_in": "N/A", "c_out": "N/A", "n_tiles": "N/A",
            "run_id": run_id, "latency_us": v, "latency_ns": v * 1000,
        })

    summary_rows.append({
        "phase": "sysfs_probe", "config": "N/A",
        "n": n, "mean_us": round(p1_mean, 4), "ci95_us": round(p1_ci95, 4),
        "mean_ns": round(p1_mean * 1000, 2), "ci95_ns": round(p1_ci95 * 1000, 2),
        "paper_claim_us": PAPER_TOTAL_OVERHEAD_US,
        "pct_diff_from_paper": round((p1_mean - PAPER_TOTAL_OVERHEAD_US) / PAPER_TOTAL_OVERHEAD_US * 100, 2),
        "match": "MATCH" if abs(p1_mean - PAPER_TOTAL_OVERHEAD_US) / PAPER_TOTAL_OVERHEAD_US < 0.5 else "MISMATCH",
    })

    # --- Phase 2: tile-selection arithmetic ---
    print("\n  Phase 2: tile-selection arithmetic ...")
    p2_lats = measure_phase2(tiler, 64, 64, n)
    p2_mean = float(np.mean(p2_lats))
    p2_ci95 = 1.96 * float(np.std(p2_lats, ddof=1)) / np.sqrt(len(p2_lats))
    print(f"    mean={p2_mean:.2f}μs CI95=±{p2_ci95:.2f}μs")

    for run_id, v in enumerate(p2_lats):
        all_raw.append({
            "timestamp": ts, "phase": "tile_selection", "config": "(64,64)",
            "c_in": 64, "c_out": 64, "n_tiles": "N/A",
            "run_id": run_id, "latency_us": v, "latency_ns": v * 1000,
        })
    summary_rows.append({
        "phase": "tile_selection", "config": "(64,64)",
        "n": n, "mean_us": round(p2_mean, 4), "ci95_us": round(p2_ci95, 4),
        "mean_ns": round(p2_mean * 1000, 2), "ci95_ns": round(p2_ci95 * 1000, 2),
        "paper_claim_us": "N/A", "pct_diff_from_paper": "N/A", "match": "N/A",
    })

    # --- Per-tile ---
    print("\n  Per-tile scheduling cost:")
    for cfg in CONFIGS_FOR_PER_TILE:
        c_in, c_out, h, w = cfg["c_in"], cfg["c_out"], cfg["h"], cfg["w"]
        dec = tiler.select_best_tile(c_in, c_out)
        td = dec["selected_tile"]["tile"]
        pt_lats, n_tiles = measure_per_tile(scheduler, c_in, c_out, h, w, td, n)
        pt_mean = float(np.mean(pt_lats))
        pt_ci95 = 1.96 * float(np.std(pt_lats, ddof=1)) / np.sqrt(len(pt_lats))
        pct_diff = round((pt_mean - PAPER_PER_TILE_NS) / PAPER_PER_TILE_NS * 100, 2)
        print(f"    ({c_in},{c_out}) n_tiles={n_tiles}: {pt_mean:.2f}ns/tile CI95=±{pt_ci95:.2f}ns  "
              f"(paper={PAPER_PER_TILE_NS}ns, {'MATCH' if abs(pct_diff)<50 else 'MISMATCH'})")

        for run_id, v in enumerate(pt_lats):
            all_raw.append({
                "timestamp": ts, "phase": "per_tile", "config": f"({c_in},{c_out})",
                "c_in": c_in, "c_out": c_out, "n_tiles": n_tiles,
                "run_id": run_id, "latency_us": v / 1000.0, "latency_ns": v,
            })
        summary_rows.append({
            "phase": "per_tile", "config": f"({c_in},{c_out})",
            "n": n, "mean_us": round(pt_mean / 1000, 4), "ci95_us": round(pt_ci95 / 1000, 4),
            "mean_ns": round(pt_mean, 2), "ci95_ns": round(pt_ci95, 2),
            "paper_claim_ns": PAPER_PER_TILE_NS,
            "pct_diff_from_paper": pct_diff,
            "match": "MATCH" if abs(pct_diff) < 50 else "MISMATCH",
        })

    # Write raw
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    with open(raw_path, "a", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_raw[0].keys()), extrasaction="ignore")
        if not exists:
            wtr.writeheader()
        wtr.writerows(all_raw)
    print(f"\n[H] Raw: {raw_path} ({len(all_raw)} rows)")

    # Write summary
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    all_keys = []
    for r in summary_rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(summary_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(summary_rows)
    print(f"[H] Summary: {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1000)
    args = p.parse_args()
    run_overhead(n=args.runs)

if __name__ == "__main__":
    main()
