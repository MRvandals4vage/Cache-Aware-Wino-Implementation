#!/usr/bin/env python3
"""
Task E — Multi-core / fusion x threading ablation.

Benchmarks the cross product of {unfused, fused} x {1-thread, 4-thread}
for configs: (64,64), (128,128), and fused-only for (32,64).
n=1000 each, monotonic clock, H'=W'=56.

Writes:
  raw_logs/jetson_ablation.csv
  summary/jetson_ablation_summary.csv

MUST be run on the Jetson Nano device.
"""
import os, sys, time, datetime, csv, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fused_winograd_kernel import FusedWinogradKernel
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.locality_scheduler import LocalityScheduler

# (c_in, c_out) -> list of (fused, threads) combinations to benchmark
ABLATION_MATRIX = {
    (64,  64):  [("unfused", False, 1), ("unfused", False, 4),
                 ("fused",   True,  1), ("fused",   True,  4)],
    (128, 128): [("unfused", False, 1), ("unfused", False, 4),
                 ("fused",   True,  1), ("fused",   True,  4)],
    (32,  64):  [("fused",   True,  1), ("fused",   True,  4)],
}

# Paper's claimed values for the key comparison in the paper:
# 33.11% for JN (64,64) 4-thread fused vs 4-thread unfused
PAPER_CLAIMS = {
    ("fused_vs_unfused_4T", 64, 64): 33.11,  # %
}


def run_ablation(n_runs=1000, warmup=20, h=56, w=56,
                 raw_path="raw_logs/jetson_ablation.csv",
                 summary_path="summary/jetson_ablation_summary.csv"):
    from scipy import stats as sp_stats
    ts = datetime.datetime.now().isoformat()
    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler()
    scheduler = LocalityScheduler()

    all_raw, config_results = [], {}

    print(f"[E] Jetson Ablation — fusion x threading")
    print(f"    n={n_runs}, warmup={warmup}, H'={h}, W'={w}")

    for (c_in, c_out), combos in ABLATION_MATRIX.items():
        decision = tiler.select_best_tile(c_in, c_out)
        td = decision["selected_tile"]["tile"]
        tile_name = decision["selected_tile"]["name"]
        tasks = scheduler.generate_tile_tasks(h, w, td, c_in, c_out)
        ordered = scheduler.group_tasks_by_channel_locality(tasks)

        np.random.seed(42)
        input_tile = np.random.randn(c_in, td, td).astype(np.float32)
        U = np.random.randn(c_out, c_in, td, td).astype(np.float32)

        print(f"\n  Config ({c_in},{c_out}), tile={tile_name}:")
        config_results[(c_in, c_out)] = {}

        for mode_name, fused, threads in combos:
            run_fn = kernel.run_fused if fused else kernel.run_non_fused
            label = f"{mode_name}_{threads}T"

            # Warmup
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
                    "mode": mode_name, "fused": fused, "threads": threads,
                    "run_id": run_id, "latency_ms": ms,
                })

            config_results[(c_in, c_out)][label] = lats
            mean = float(np.mean(lats))
            ci95 = 1.96 * float(np.std(lats, ddof=1)) / np.sqrt(len(lats))
            print(f"    {label}: mean={mean:.4f}ms CI95=±{ci95:.4f}ms")

    # Write raw
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    with open(raw_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_raw[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(all_raw)
    print(f"\n[E] Raw: {raw_path} ({len(all_raw)} rows)")

    # Build summary with pairwise t-tests
    summary_rows = []
    for (c_in, c_out), modes in config_results.items():
        for label, lats in modes.items():
            mean = float(np.mean(lats))
            ci95 = 1.96 * float(np.std(lats, ddof=1)) / np.sqrt(len(lats))

            # Compare to baseline (unfused_1T) if available
            bl = modes.get("unfused_1T")
            if bl and label != "unfused_1T":
                _, p = sp_stats.ttest_ind(lats, bl, equal_var=False)
                p_str = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"
                improvement = (float(np.mean(bl)) - mean) / float(np.mean(bl)) * 100.0
            else:
                p_str = "N/A"
                improvement = "N/A"

            row = {
                "config": f"({c_in},{c_out})", "c_in": c_in, "c_out": c_out,
                "mode": label, "n": len(lats),
                "mean_ms": round(mean, 4), "ci95_ms": round(ci95, 4),
                "p_vs_unfused_1T": p_str,
                "improvement_vs_unfused_1T_pct": round(improvement, 4) if isinstance(improvement, float) else improvement,
            }
            # Add paper claim comparison for the key metric
            paper_key = ("fused_vs_unfused_4T", c_in, c_out)
            if label == "fused_4T" and paper_key in PAPER_CLAIMS:
                unfused_4T = modes.get("unfused_4T")
                if unfused_4T:
                    _, p2 = sp_stats.ttest_ind(lats, unfused_4T, equal_var=False)
                    imp_vs_unf4T = (float(np.mean(unfused_4T)) - mean) / float(np.mean(unfused_4T)) * 100.0
                    paper_val = PAPER_CLAIMS[paper_key]
                    pct_diff = round((imp_vs_unf4T - paper_val) / abs(paper_val) * 100, 2)
                    row["paper_fused_4T_vs_unfused_4T_pct"] = paper_val
                    row["measured_fused_4T_vs_unfused_4T_pct"] = round(imp_vs_unf4T, 4)
                    row["pct_diff_from_paper"] = pct_diff
                    row["match"] = "MATCH" if abs(pct_diff) < 5.0 else "MISMATCH"

            summary_rows.append(row)

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        all_keys = []
        for r in summary_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[E] Summary: {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--height", type=int, default=56)
    p.add_argument("--width", type=int, default=56)
    args = p.parse_args()
    run_ablation(n_runs=args.runs, warmup=args.warmup, h=args.height, w=args.width)

if __name__ == "__main__":
    main()
