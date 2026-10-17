#!/usr/bin/env python3
"""
Rebuild the combined e2e summary table from per-model raw CSVs.
Needed because e2e_benchmark.py overwrites summary/e2e_comparison_table.csv
on every invocation -- running models as separate processes means only the
last model's summary survives there. This recomputes the same stats
(mean, std, CI95, p-value vs cachewinograd, pct improvement) from the
raw per-run latency data in raw_logs/, which IS preserved per model.
"""
import glob
import os
import csv
import numpy as np
from scipy import stats as sp_stats

RAW_DIR = "raw_logs"
OUT_PATH = "summary/e2e_full_comparison_table.csv"

MODEL_DISPLAY = {
    "alexnet": "AlexNet",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "vgg16": "VGG16",
}

def load_raw(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    pattern = os.path.join(RAW_DIR, "e2e_*_rpi4.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched {pattern}")
        return

    summary_rows = []

    for path in files:
        fname = os.path.basename(path)
        # e2e_<model>_rpi4.csv
        model_name = fname[len("e2e_"):-len("_rpi4.csv")]
        display = MODEL_DISPLAY.get(model_name, model_name)

        rows = load_raw(path)
        if not rows:
            print(f"  {fname}: empty, skipping")
            continue

        by_baseline = {}
        for row in rows:
            bl = row["baseline"]
            by_baseline.setdefault(bl, []).append(float(row["latency_ms"]))

        cw_lats = by_baseline.get("cachewinograd")
        cw_mean = float(np.mean(cw_lats)) if cw_lats else None

        for bl_name, lats in by_baseline.items():
            mean_ms = float(np.mean(lats))
            std_ms = float(np.std(lats, ddof=1)) if len(lats) > 1 else 0.0
            ci95 = 1.96 * std_ms / np.sqrt(len(lats)) if len(lats) > 1 else 0.0

            p_value = "N/A"
            improvement = "N/A"
            if bl_name != "cachewinograd" and cw_lats and len(cw_lats) > 1 and len(lats) > 1:
                _, p = sp_stats.ttest_ind(cw_lats, lats, equal_var=False)
                p_value = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"
                improvement = round(((mean_ms - cw_mean) / mean_ms) * 100.0, 2)

            summary_rows.append({
                "model": display,
                "platform": "rpi4",
                "baseline": bl_name,
                "mean_ms": round(mean_ms, 4),
                "CI95": round(ci95, 4),
                "std_ms": round(std_ms, 4),
                "p_value": p_value,
                "pct_improvement": improvement,
                "n_runs": len(lats),
            })

        print(f"  {fname}: {len(rows)} rows -> {len(by_baseline)} baselines "
              f"({', '.join(f'{k} n={len(v)}' for k, v in by_baseline.items())})")

    if not summary_rows:
        print("No summary rows produced.")
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nWrote {len(summary_rows)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()

