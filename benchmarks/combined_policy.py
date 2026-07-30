#!/usr/bin/env python3
"""
T8 — Combined fusion+parallelism policy.

Implements a policy that chooses fused-single-thread vs unfused-multi-thread
based on measured regime (efficient vs memory-bound).

Decision rule:
  WS_ext <= gamma * C_cache  →  fused, single-thread (efficient regime)
  WS_ext >  gamma * C_cache  →  unfused, multi-thread (memory-bound regime)

Benchmarks for (64,64) and (128,128) configs on Jetson Nano.
Logs to raw_logs/combined_policy.csv.
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

CONFIGS = [
    {"c_in": 64, "c_out": 64},
    {"c_in": 128, "c_out": 128},
]


def combined_policy_decision(tiler, c_in, c_out, gamma=0.7):
    """Decide: fused-single-thread vs unfused-multi-thread.

    Returns (fused: bool, threads: int, regime: str, reasoning: dict)
    """
    decision = tiler.select_best_tile(c_in, c_out)
    tile_dim = decision["selected_tile"]["tile"]
    ws = tiler.compute_working_set(tile_dim, c_in, c_out)
    cache_cap = tiler.l1_capacity

    threshold = gamma * cache_cap
    efficient = ws <= threshold

    return {
        "fused": efficient,
        "threads": 1 if efficient else 4,
        "regime": "efficient" if efficient else "memory_bound",
        "tile_dim": tile_dim,
        "tile_name": decision["selected_tile"]["name"],
        "working_set": ws,
        "cache_capacity": cache_cap,
        "threshold": threshold,
        "gamma": gamma,
    }


def run_combined_policy(n_runs=50, warmup=10, h=14, w=14, n_threads=4, gamma=0.7):
    """Benchmark the combined policy."""
    from scipy import stats as sp_stats

    platform = build_platform_descriptor()
    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler(platform_descriptor=platform)
    scheduler = LocalityScheduler()

    timestamp = datetime.datetime.now().isoformat()
    raw_rows = []
    summary_rows = []

    print(f"[T8] Combined Fusion+Parallelism Policy")
    print(f"     Platform: {platform.get('os')} / {platform.get('cpu_model')}")
    print(f"     L1D: {platform.get('l1d_size_bytes')}B, gamma={gamma}")

    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        print(f"\n  Config ({c_in},{c_out}):")

        # Get policy decision
        policy = combined_policy_decision(tiler, c_in, c_out, gamma)
        print(f"    Policy: regime={policy['regime']}, fused={policy['fused']}, "
              f"threads={policy['threads']}")
        print(f"    WS={policy['working_set']}B, threshold={policy['threshold']:.0f}B")

        # Benchmark all 4 combinations
        modes = [
            {"name": "fused_1T", "fused": True, "threads": 1},
            {"name": "fused_4T", "fused": True, "threads": n_threads},
            {"name": "unfused_1T", "fused": False, "threads": 1},
            {"name": "unfused_4T", "fused": False, "threads": n_threads},
            {"name": "policy", "fused": policy["fused"], "threads": policy["threads"]},
        ]

        tile_dim = policy["tile_dim"]
        tasks = scheduler.generate_tile_tasks(h, w, tile_dim, c_in, c_out)
        ordered_tasks = scheduler.group_tasks_by_channel_locality(tasks)

        np.random.seed(42)
        input_tile = np.random.randn(c_in, tile_dim, tile_dim).astype(np.float32)
        U = np.random.randn(c_out, c_in, tile_dim, tile_dim).astype(np.float32)

        mode_results = {}

        for mode in modes:
            mode_name = mode["name"]
            fused = mode["fused"]
            threads = mode["threads"]
            run_func = kernel.run_fused if fused else kernel.run_non_fused

            # Warmup
            for _ in range(warmup):
                for _ in ordered_tasks[:min(10, len(ordered_tasks))]:
                    run_func(input_tile, U)

            # Measurement
            latencies = []
            for run_id in range(n_runs):
                t0 = time.perf_counter()
                for task in ordered_tasks:
                    run_func(input_tile, U)
                t1 = time.perf_counter()
                duration_ms = (t1 - t0) * 1000.0
                latencies.append(duration_ms)

                raw_rows.append({
                    "timestamp": timestamp,
                    "c_in": c_in, "c_out": c_out,
                    "mode": mode_name,
                    "fused": fused, "threads": threads,
                    "regime": policy["regime"],
                    "run_id": run_id,
                    "latency_ms": duration_ms,
                })

            mean_lat = float(np.mean(latencies))
            std_lat = float(np.std(latencies, ddof=1))
            ci95 = 1.96 * std_lat / np.sqrt(len(latencies))
            mode_results[mode_name] = {"mean": mean_lat, "lats": latencies}

            print(f"    {mode_name}: mean={mean_lat:.2f}ms, CI95=±{ci95:.2f}ms")

        # Find empirical best
        best_mode = min(mode_results, key=lambda k: mode_results[k]["mean"])
        policy_is_best = best_mode == "policy" or mode_results["policy"]["mean"] <= mode_results[best_mode]["mean"] * 1.02

        print(f"    Empirical best: {best_mode} ({mode_results[best_mode]['mean']:.2f}ms)")
        print(f"    Policy chose: {policy['regime']} ({mode_results['policy']['mean']:.2f}ms)")
        print(f"    Policy optimal: {'YES ✓' if policy_is_best else 'NO ✗'}")

        # Welch t-test: policy vs empirical best (if different)
        if best_mode != "policy":
            _, p = sp_stats.ttest_ind(
                mode_results["policy"]["lats"],
                mode_results[best_mode]["lats"],
                equal_var=False
            )
            p_str = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"
        else:
            p_str = "N/A (policy IS best)"

        summary_rows.append({
            "config": f"({c_in},{c_out})",
            "regime": policy["regime"],
            "policy_mode": f"{'fused' if policy['fused'] else 'unfused'}_{policy['threads']}T",
            "policy_mean_ms": round(mode_results["policy"]["mean"], 4),
            "best_mode": best_mode,
            "best_mean_ms": round(mode_results[best_mode]["mean"], 4),
            "policy_is_optimal": policy_is_best,
            "p_value": p_str,
            "working_set": policy["working_set"],
            "gamma": gamma,
        })

    # Write raw log (append)
    out_path = os.path.join("raw_logs", "combined_policy.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\n[T8] Appended {len(raw_rows)} rows to {out_path}")

    # Write summary
    summary_path = os.path.join("summary", "combined_policy_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[T8] Summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="T8: Combined fusion+parallelism policy")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--width", type=int, default=14)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.7)
    args = parser.parse_args()
    run_combined_policy(n_runs=args.runs, warmup=args.warmup,
                         h=args.height, w=args.width,
                         n_threads=args.threads, gamma=args.gamma)


if __name__ == "__main__":
    main()
