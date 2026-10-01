#!/usr/bin/env python3
"""
Task M — Raspberry Pi 4 main microbenchmark suite, ALL 7 configs, m=4 (fixed).

Baseline non-fused Winograd vs fused, single-threaded.
n>=30, monotonic clock.

Configs: (16,32),(32,16),(32,32),(32,64),(64,32),(64,64),(128,128)
m=4 fixed -> tile_dim=6 (F(4,3))

Paper claim: 63.91% improvement for (128,128) m=4 on RPi4.

Writes:
  raw_logs/rpi4_main_microbench.csv
  summary/rpi4_main_microbench_summary.csv

MUST be run on the Raspberry Pi 4 device.
"""
import os, sys, time, datetime, csv, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fused_winograd_kernel import FusedWinogradKernel
from src.locality_scheduler import LocalityScheduler

CONFIGS = [
    {"c_in": 16,  "c_out": 32},
    {"c_in": 32,  "c_out": 16},
    {"c_in": 32,  "c_out": 32},
    {"c_in": 32,  "c_out": 64},
    {"c_in": 64,  "c_out": 32},
    {"c_in": 64,  "c_out": 64},
    {"c_in": 128, "c_out": 128},
]

FIXED_M = 4
TILE_DIM = FIXED_M + 2  # F(4,3): tile_dim=6

# Paper claims — only for MATCH/MISMATCH comparison
PAPER_CLAIMS = {
    (128, 128): {"improvement_pct": 63.91},
}


def run_rpi4_suite(n_runs=30, warmup=10, h=14, w=14,
                   raw_path="raw_logs/rpi4_main_microbench.csv",
                   summary_path="summary/rpi4_main_microbench_summary.csv"):
    from scipy import stats as sp_stats
    ts = datetime.datetime.now().isoformat()
    kernel = FusedWinogradKernel()
    scheduler = LocalityScheduler()

    print(f"[M] RPi4 Main Microbenchmark — ALL 7 configs, m={FIXED_M} (tile_dim={TILE_DIM})")
    print(f"    n={n_runs}, warmup={warmup}, H'={h}, W'={w}")

    all_raw, summary_rows = [], []

    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        print(f"\n  Config ({c_in},{c_out}):")

        tasks = scheduler.generate_tile_tasks(h, w, TILE_DIM, c_in, c_out)
        ordered = scheduler.group_tasks_by_channel_locality(tasks)

        np.random.seed(42)
        input_tile = np.random.randn(c_in, TILE_DIM, TILE_DIM).astype(np.float32)
        U = np.random.randn(c_out, c_in, TILE_DIM, TILE_DIM).astype(np.float32)

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
                    "timestamp": ts, "platform": "rpi4",
                    "c_in": c_in, "c_out": c_out,
                    "h": h, "w": w, "m": FIXED_M, "tile_dim": TILE_DIM,
                    "method": method, "threads": 1,
                    "run_id": run_id, "latency_ms": ms,
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

        paper = PAPER_CLAIMS.get((c_in, c_out), {})
        p_imp = paper.get("improvement_pct")
        pct_diff = round((improvement - p_imp) / abs(p_imp) * 100, 2) if p_imp is not None else "N/A"
        match = ("MATCH" if isinstance(pct_diff, float) and abs(pct_diff) < 5.0
                 else "MISMATCH" if isinstance(pct_diff, float) else "NO_PAPER_CLAIM")

        summary_rows.append({
            "config": f"({c_in},{c_out})", "c_in": c_in, "c_out": c_out,
            "m": FIXED_M, "tile_dim": TILE_DIM, "n": n_runs,
            "baseline_mean_ms": round(float(np.mean(bl)), 4),
            "baseline_ci95_ms": round(1.96 * float(np.std(bl, ddof=1)) / np.sqrt(len(bl)), 4),
            "fused_mean_ms": round(float(np.mean(fu)), 4),
            "fused_ci95_ms": round(1.96 * float(np.std(fu, ddof=1)) / np.sqrt(len(fu)), 4),
            "improvement_pct": round(improvement, 4),
            "p_value": p_str,
            "significance": "SIGNIFICANT" if p < 0.05 else "NOT_SIGNIFICANT",
            "paper_improvement_pct": p_imp if p_imp is not None else "N/A",
            "pct_diff_from_paper": pct_diff,
            "match": match,
        })
        print(f"    improvement={improvement:.2f}% | paper={p_imp}% | {match}")

    # Write raw
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    with open(raw_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_raw[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(all_raw)
    print(f"\n[M] Raw: {raw_path} ({len(all_raw)} rows)")

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[M] Summary: {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--height", type=int, default=14)
    p.add_argument("--width", type=int, default=14)
    p.add_argument("--raw", default="raw_logs/rpi4_main_microbench.csv")
    p.add_argument("--summary", default="summary/rpi4_main_microbench_summary.csv")
    args = p.parse_args()
    run_rpi4_suite(n_runs=args.runs, warmup=args.warmup, h=args.height, w=args.width,
                   raw_path=args.raw, summary_path=args.summary)

if __name__ == "__main__":
    main()
