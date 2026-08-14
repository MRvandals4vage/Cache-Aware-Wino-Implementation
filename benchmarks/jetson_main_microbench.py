#!/usr/bin/env python3
"""
Task C — Jetson Nano main microbenchmark suite, ALL 7 configs.

Baseline direct conv (non-fused Winograd) vs fused Winograd, single-threaded.
n=1000, 20 warm-up iterations, monotonic clock (CLOCK_MONOTONIC via time.monotonic()).
H'=W'=56 for all configs per paper protocol.

Configs: (16,32),(32,16),(32,32),(32,64),(64,32),(64,64),(128,128)

Writes:
  raw_logs/jetson_main_microbench.csv  — every individual run
  summary/jetson_main_microbench_summary.csv — mean/CI95/p-value per config/method

MUST be run on the Jetson Nano device.
"""
import os, sys, time, datetime, csv, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fused_winograd_kernel import FusedWinogradKernel
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
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

# Paper's claimed values: (c_in, c_out) -> (baseline_ms, fused_ms, improvement_pct)
# These are ONLY used for the MATCH/MISMATCH comparison — never used for any measurement.
PAPER_CLAIMS = {
    (16,  32):  {"baseline_ms": None, "fused_ms": None, "improvement_pct": None},
    (32,  16):  {"baseline_ms": None, "fused_ms": None, "improvement_pct": None},
    (32,  32):  {"baseline_ms": None, "fused_ms": None, "improvement_pct": None},
    (32,  64):  {"baseline_ms": None, "fused_ms": None, "improvement_pct": None},
    (64,  32):  {"baseline_ms": None, "fused_ms": None, "improvement_pct": None},
    (64,  64):  {"baseline_ms": None, "fused_ms": None, "improvement_pct": None},
    (128, 128): {"baseline_ms": 860.65, "fused_ms": 807.97, "improvement_pct": -6.52},
}


def run_config(kernel, tiler, scheduler, c_in, c_out, h, w, n_runs, warmup, ts):
    decision = tiler.select_best_tile(c_in, c_out)
    td = decision["selected_tile"]["tile"]
    tile_name = decision["selected_tile"]["name"]
    tasks = scheduler.generate_tile_tasks(h, w, td, c_in, c_out)
    ordered = scheduler.group_tasks_by_channel_locality(tasks)

    np.random.seed(42)
    input_tile = np.random.randn(c_in, td, td).astype(np.float32)
    U = np.random.randn(c_out, c_in, td, td).astype(np.float32)

    rows = []
    results = {}

    for method, fused in [("baseline_nonfused", False), ("fused", True)]:
        run_fn = kernel.run_fused if fused else kernel.run_non_fused
        # Full warmup — all tasks exercised
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
            rows.append({
                "timestamp": ts, "c_in": c_in, "c_out": c_out,
                "h": h, "w": w, "tile": tile_name, "method": method,
                "threads": 1, "run_id": run_id, "latency_ms": ms,
            })

        results[method] = lats
        mean = np.mean(lats)
        ci95 = 1.96 * np.std(lats, ddof=1) / np.sqrt(len(lats))
        print(f"      {method}: mean={mean:.4f}ms CI95=±{ci95:.4f}ms")

    return rows, results


def run_suite(n_runs=1000, warmup=20, h=56, w=56,
              raw_path="raw_logs/jetson_main_microbench.csv",
              summary_path="summary/jetson_main_microbench_summary.csv",
              only_128=False):
    from scipy import stats as sp_stats
    ts = datetime.datetime.now().isoformat()
    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler()
    scheduler = LocalityScheduler()

    all_raw, all_summary = [], []

    configs_to_run = [cfg for cfg in CONFIGS if cfg["c_in"] == 128 and cfg["c_out"] == 128] if only_128 else CONFIGS

    print(f"[C] Jetson Nano Main Microbenchmark — {'(128,128) ONLY' if only_128 else 'ALL 7 configs'}")
    print(f"    n={n_runs}, warmup={warmup}, H'={h}, W'={w}")

    for cfg in configs_to_run:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        print(f"\n  Config ({c_in},{c_out}):")
        rows, results = run_config(kernel, tiler, scheduler,
                                   c_in, c_out, h, w, n_runs, warmup, ts)
        all_raw.extend(rows)

        bl_lats = results["baseline_nonfused"]
        fu_lats = results["fused"]
        bl_mean = float(np.mean(bl_lats))
        fu_mean = float(np.mean(fu_lats))
        bl_ci95 = 1.96 * float(np.std(bl_lats, ddof=1)) / np.sqrt(len(bl_lats))
        fu_ci95 = 1.96 * float(np.std(fu_lats, ddof=1)) / np.sqrt(len(fu_lats))
        _, p = sp_stats.ttest_ind(fu_lats, bl_lats, equal_var=False)
        p_str = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"
        improvement = (bl_mean - fu_mean) / bl_mean * 100.0

        paper = PAPER_CLAIMS.get((c_in, c_out), {})
        p_imp = paper.get("improvement_pct")
        pct_diff_imp = round((improvement - p_imp) / abs(p_imp) * 100, 2) if p_imp is not None else "N/A"
        match = ("MATCH" if isinstance(pct_diff_imp, float) and abs(pct_diff_imp) < 5.0
                 else "MISMATCH" if isinstance(pct_diff_imp, float) else "NO_PAPER_CLAIM")

        all_summary.append({
            "config": f"({c_in},{c_out})", "c_in": c_in, "c_out": c_out,
            "n": n_runs,
            "baseline_mean_ms": round(bl_mean, 4), "baseline_ci95_ms": round(bl_ci95, 4),
            "fused_mean_ms": round(fu_mean, 4), "fused_ci95_ms": round(fu_ci95, 4),
            "improvement_pct": round(improvement, 4),
            "p_value": p_str,
            "significance": "SIGNIFICANT" if p < 0.05 else "NOT_SIGNIFICANT",
            "paper_improvement_pct": p_imp if p_imp is not None else "N/A",
            "pct_diff_from_paper": pct_diff_imp,
            "match": match,
        })
        print(f"    improvement={improvement:.2f}% | paper={p_imp}% | {match}")

    # Write raw (append)
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    with open(raw_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_raw[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(all_raw)
    print(f"\n[C] Raw: {raw_path} ({len(all_raw)} rows)")

    # Write summary (overwrite if running all, append or update if only_128)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    if only_128 and os.path.exists(summary_path) and os.path.getsize(summary_path) > 0:
        # Read existing summary rows and update/append (128,128)
        existing_rows = []
        with open(summary_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for r in reader:
                if r.get("config") != "(128,128)":
                    existing_rows.append(r)
        existing_rows.extend(all_summary)
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames or list(all_summary[0].keys()))
            w.writeheader()
            w.writerows(existing_rows)
    else:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
            w.writeheader()
            w.writerows(all_summary)
    print(f"[C] Summary: {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--height", type=int, default=56)
    p.add_argument("--width", type=int, default=56)
    p.add_argument("--only-128", action="store_true", help="Run only (128,128) config")
    p.add_argument("--raw", default="raw_logs/jetson_main_microbench.csv")
    p.add_argument("--summary", default="summary/jetson_main_microbench_summary.csv")
    args = p.parse_args()
    run_suite(n_runs=args.runs, warmup=args.warmup, h=args.height, w=args.width,
              raw_path=args.raw, summary_path=args.summary, only_128=args.only_128)

if __name__ == "__main__":
    main()
