#!/usr/bin/env python3
"""
T4 — Raspberry Pi 4 adaptive-m sweep.

Runs all 7 microbenchmark configs at m=2 and m=6 on Raspberry Pi 4
(existing m=4 data stays as-is), n>=30 per config.

For each config, verifies the adaptive policy's m* selection logic
(WS_ext <= gamma*C_cache) picks the same m that empirically performs best.

Logs to raw_logs/rpi4_adaptive_m_sweep.csv.

MUST be executed on a Raspberry Pi 4 device.
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
from src.runtime_cache_probe import build_platform_descriptor

# All 7 microbenchmark configs from the paper
CONFIGS = [
    {"c_in": 16, "c_out": 32},
    {"c_in": 32, "c_out": 16},
    {"c_in": 32, "c_out": 32},
    {"c_in": 32, "c_out": 64},
    {"c_in": 64, "c_out": 32},
    {"c_in": 64, "c_out": 64},
    {"c_in": 128, "c_out": 128},
]

# Candidate m values -> tile_dim = m + r - 1 = m + 2
M_VALUES = [2, 6]  # m=4 (tile=6) data stays as-is


def _tile_dim_from_m(m):
    """F(m,3) uses tile_dim = m + 3 - 1 = m + 2."""
    return m + 2


def run_sweep(n_runs=30, warmup=10, h=14, w=14):
    """Run the adaptive-m sweep."""
    from scipy import stats as sp_stats

    platform = build_platform_descriptor()
    print(f"[T4] RPi4 Adaptive-m Sweep")
    print(f"     Platform: {platform.get('os', 'unknown')} / {platform.get('cpu_model', 'unknown')}")
    print(f"     L1D: {platform.get('l1d_size_bytes', 'N/A')}B, L2: {platform.get('l2_size_bytes', 'N/A')}B")
    print(f"     Configs: {len(CONFIGS)}, m values: {M_VALUES}, Runs: {n_runs}")

    kernel = FusedWinogradKernel()
    scheduler = LocalityScheduler()

    # Build tiler with the detected platform
    tiler = CacheAdaptiveAutotiler(platform_descriptor=platform)

    timestamp = datetime.datetime.now().isoformat()
    raw_rows = []
    verification_rows = []

    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        print(f"\n  Config ({c_in},{c_out}):")

        # Get the adaptive policy's recommendation
        decision = tiler.select_best_tile(c_in, c_out)
        policy_tile_dim = decision["selected_tile"]["tile"]
        policy_m = policy_tile_dim - 2
        policy_name = decision["selected_tile"]["name"]
        policy_ws = decision["working_set"]
        print(f"    Adaptive policy selects: {policy_name} (m={policy_m}), WS={policy_ws}B")

        best_m = None
        best_mean = float("inf")
        m_results = {}

        for m in M_VALUES:
            tile_dim = _tile_dim_from_m(m)
            tile_name = f"F({m},3)"
            print(f"    Running m={m} ({tile_name}, tile_dim={tile_dim})...")

            tasks = scheduler.generate_tile_tasks(h, w, tile_dim, c_in, c_out)
            ordered_tasks = scheduler.group_tasks_by_channel_locality(tasks)

            np.random.seed(42)
            input_tile = np.random.randn(c_in, tile_dim, tile_dim).astype(np.float32)
            U = np.random.randn(c_out, c_in, tile_dim, tile_dim).astype(np.float32)

            # Warmup
            for _ in range(warmup):
                for _ in ordered_tasks[:min(10, len(ordered_tasks))]:
                    kernel.run_fused(input_tile, U)

            # Measurement — CLOCK_MONOTONIC via time.monotonic()
            latencies = []
            for run_id in range(n_runs):
                t0 = time.monotonic()
                for task in ordered_tasks:
                    kernel.run_fused(input_tile, U)
                t1 = time.monotonic()
                duration_ms = (t1 - t0) * 1000.0
                latencies.append(duration_ms)

                raw_rows.append({
                    "timestamp": timestamp,
                    "c_in": c_in, "c_out": c_out,
                    "m": m, "tile_dim": tile_dim, "tile_name": tile_name,
                    "h": h, "w": w,
                    "run_id": run_id,
                    "latency_ms": duration_ms,
                })

            mean_lat = float(np.mean(latencies))
            std_lat = float(np.std(latencies, ddof=1))
            ci95 = 1.96 * std_lat / np.sqrt(len(latencies))
            m_results[m] = {"mean": mean_lat, "std": std_lat, "ci95": ci95, "lats": latencies}

            print(f"      mean={mean_lat:.2f}ms, CI95=±{ci95:.2f}ms")

            if mean_lat < best_mean:
                best_mean = mean_lat
                best_m = m

        # Verify policy matches empirical best
        policy_matches = (policy_m == best_m)
        print(f"    Empirical best: m={best_m} ({best_mean:.2f}ms)")
        print(f"    Policy selected: m={policy_m}")
        print(f"    Match: {'YES ✓' if policy_matches else 'NO ✗'}")

        # Welch's t-test between m=2 and m=6 to quantify significance
        m2_lats = m_results.get(2, {}).get("lats", [])
        m6_lats = m_results.get(6, {}).get("lats", [])
        if m2_lats and m6_lats:
            from scipy import stats as sp_stats
            _, p_m2_m6 = sp_stats.ttest_ind(m2_lats, m6_lats, equal_var=False)
            p_m2_m6_str = f"{p_m2_m6:.6e}" if p_m2_m6 >= 1e-10 else "< 1e-10"
        else:
            p_m2_m6_str = "N/A"

        verification_rows.append({
            "config": f"({c_in},{c_out})",
            "c_in": c_in, "c_out": c_out,
            "policy_m": policy_m,
            "policy_tile": policy_name,
            "policy_ws": policy_ws,
            "empirical_best_m": best_m,
            "m2_mean_ms": round(m_results.get(2, {}).get("mean", float("nan")), 4),
            "m2_ci95_ms": round(m_results.get(2, {}).get("ci95", float("nan")), 4),
            "m6_mean_ms": round(m_results.get(6, {}).get("mean", float("nan")), 4),
            "m6_ci95_ms": round(m_results.get(6, {}).get("ci95", float("nan")), 4),
            "p_value_m2_vs_m6": p_m2_m6_str,
            "policy_matches_empirical": policy_matches,
        })

    # Write raw log (append)
    raw_path = os.path.join("raw_logs", "rpi4_adaptive_m_sweep.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    file_exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    with open(raw_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\n[T4] Appended {len(raw_rows)} rows to {raw_path}")

    # Write verification summary
    summary_path = os.path.join("summary", "rpi4_adaptive_m_verification.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(verification_rows[0].keys()))
        writer.writeheader()
        writer.writerows(verification_rows)
    print(f"[T4] Verification summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="T4: RPi4 adaptive-m sweep")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--width", type=int, default=14)
    args = parser.parse_args()
    run_sweep(n_runs=args.runs, warmup=args.warmup, h=args.height, w=args.width)


if __name__ == "__main__":
    main()
