#!/usr/bin/env python3
"""
T1 — Generate paper figures from CSV data.

Produces exactly:
  paper_fig_microbench_improvement.png
  paper_fig_microbench_ci.png
  paper_fig_end_to_end_latency.png

All data is read from CSV files on disk.  Nothing is hardcoded.
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLORS = {
    "baseline":    "#3B82F6",
    "unfused_mt":  "#10B981",
    "fused_st":    "#F59E0B",
    "fused_mt":    "#EF4444",
    "cachewinograd": "#8B5CF6",
    "onnxruntime":   "#6366F1",
    "tvm":           "#14B8A6",
    "autotvm":       "#F97316",
    "armcl":         "#EC4899",
    "naive_wino":    "#64748B",
    "nonfused_wino": "#94A3B8",
}

HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


# ===================================================================
# Figure 1: Microbenchmark Improvement (%)
# ===================================================================
def plot_microbench_improvement(processed_csv: str, out_path: str) -> bool:
    if not os.path.exists(processed_csv):
        print(f"[T1] SKIP  {out_path} — source CSV not found: {processed_csv}")
        return False

    df = pd.read_csv(processed_csv)
    if df.empty:
        return False

    # Build labels
    df["Config"] = df.apply(lambda r: f"{r['C_in']}×{r['C_out']}", axis=1)
    fused_col = "Fused" if "Fused" in df.columns else "fused"
    mc_col = "MultiCore" if "MultiCore" in df.columns else "threads"

    def _mode_label(row):
        fused = str(row[fused_col]) in ("True", "Yes", "1")
        multi = str(row[mc_col]) in ("True", "Yes", "4")
        if not fused and not multi:
            return "Baseline (unfused, 1T)"
        elif not fused and multi:
            return "Unfused, 4T"
        elif fused and not multi:
            return "Fused, 1T"
        else:
            return "Fused, 4T"

    df["Mode"] = df.apply(_mode_label, axis=1)

    imp_col = "Improvement_vs_Baseline_pct"
    # Filter only rows that have improvement data
    df_imp = df[df[imp_col].apply(lambda v: str(v) not in ("N/A", "nan", ""))].copy()
    df_imp[imp_col] = df_imp[imp_col].astype(float)

    configs = df["Config"].unique()
    modes = [m for m in ["Unfused, 4T", "Fused, 1T", "Fused, 4T"] if m in df_imp["Mode"].unique()]
    x = np.arange(len(configs))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    mode_colors = [COLORS["unfused_mt"], COLORS["fused_st"], COLORS["fused_mt"]]
    for i, m in enumerate(modes):
        vals = []
        for c in configs:
            row = df_imp[(df_imp["Config"] == c) & (df_imp["Mode"] == m)]
            vals.append(float(row.iloc[0][imp_col]) if not row.empty else 0.0)
        ax.bar(x + (i - len(modes) / 2 + 0.5) * width, vals, width,
               label=m, color=mode_colors[i % len(mode_colors)],
               hatch=HATCHES[i % len(HATCHES)], edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Latency Improvement over Non-Fused Baseline (%)")
    ax.set_ylabel("Improvement (%)")
    ax.set_xlabel("Workload (Cin × Cout)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[T1] SAVED {out_path}")
    return True


# ===================================================================
# Figure 2: Microbenchmark CI (mean + error bars)
# ===================================================================
def plot_microbench_ci(processed_csv: str, out_path: str) -> bool:
    if not os.path.exists(processed_csv):
        print(f"[T1] SKIP  {out_path} — source CSV not found: {processed_csv}")
        return False

    df = pd.read_csv(processed_csv)
    if df.empty:
        return False

    df["Config"] = df.apply(lambda r: f"{r['C_in']}×{r['C_out']}", axis=1)
    fused_col = "Fused" if "Fused" in df.columns else "fused"
    mc_col = "MultiCore" if "MultiCore" in df.columns else "threads"

    def _mode_label(row):
        fused = str(row[fused_col]) in ("True", "Yes", "1")
        multi = str(row[mc_col]) in ("True", "Yes", "4")
        if not fused and not multi:
            return "Baseline (unfused, 1T)"
        elif not fused and multi:
            return "Unfused, 4T"
        elif fused and not multi:
            return "Fused, 1T"
        else:
            return "Fused, 4T"

    df["Mode"] = df.apply(_mode_label, axis=1)
    configs = df["Config"].unique()
    modes = sorted(df["Mode"].unique())
    x = np.arange(len(configs))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    mode_colors = [COLORS["baseline"], COLORS["unfused_mt"], COLORS["fused_st"], COLORS["fused_mt"]]

    for i, m in enumerate(modes):
        vals, errs = [], []
        for c in configs:
            row = df[(df["Config"] == c) & (df["Mode"] == m)]
            if not row.empty:
                vals.append(float(row.iloc[0]["Mean_Latency_ms"]))
                ci_col = "CI95_ms" if "CI95_ms" in df.columns else "Conf_Interval_95"
                errs.append(float(row.iloc[0][ci_col]) if ci_col in df.columns else 0)
            else:
                vals.append(0)
                errs.append(0)
        ax.bar(x + (i - len(modes) / 2 + 0.5) * width, vals, width,
               yerr=errs, capsize=3, label=m,
               color=mode_colors[i % len(mode_colors)],
               hatch=HATCHES[i % len(HATCHES)], edgecolor="white",
               linewidth=0.5, alpha=0.85)

    ax.set_title("Mean Latency with 95% Confidence Intervals")
    ax.set_ylabel("Latency (ms) ± 95% CI")
    ax.set_xlabel("Workload (Cin × Cout)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[T1] SAVED {out_path}")
    return True


# ===================================================================
# Figure 3: End-to-End Latency Comparison
# ===================================================================
def plot_e2e_latency(e2e_csv: str, out_path: str) -> bool:
    if not os.path.exists(e2e_csv):
        print(f"[T1] SKIP  {out_path} — source CSV not found: {e2e_csv}")
        return False

    df = pd.read_csv(e2e_csv)
    if df.empty:
        return False

    # Expect columns: model, platform, baseline, mean_ms, CI95, p_value, pct_improvement
    required = {"model", "baseline", "mean_ms"}
    if not required.issubset(set(df.columns)):
        # Try alternate column names
        col_map = {}
        for c in df.columns:
            lc = c.lower().replace(" ", "_")
            if "model" in lc:
                col_map[c] = "model"
            elif "baseline" in lc or "backend" in lc:
                col_map[c] = "baseline"
            elif "mean" in lc and "ms" in lc:
                col_map[c] = "mean_ms"
            elif "ci95" in lc or "ci_95" in lc:
                col_map[c] = "CI95"
        df = df.rename(columns=col_map)
        if not required.issubset(set(df.columns)):
            print(f"[T1] SKIP  {out_path} — CSV columns don't match expected schema: {list(df.columns)}")
            return False

    models = sorted(df["model"].unique())
    baselines = sorted(df["baseline"].unique())
    x = np.arange(len(models))
    width = 0.8 / max(len(baselines), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    bl_colors = list(COLORS.values())

    for i, bl in enumerate(baselines):
        vals, errs = [], []
        for m in models:
            row = df[(df["model"] == m) & (df["baseline"] == bl)]
            if not row.empty:
                vals.append(float(row.iloc[0]["mean_ms"]))
                errs.append(float(row.iloc[0]["CI95"]) if "CI95" in df.columns else 0)
            else:
                vals.append(0)
                errs.append(0)
        ax.bar(x + (i - len(baselines) / 2 + 0.5) * width, vals, width,
               yerr=errs if any(e > 0 for e in errs) else None,
               capsize=2, label=bl,
               color=bl_colors[i % len(bl_colors)],
               hatch=HATCHES[i % len(HATCHES)],
               edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_title("End-to-End Inference Latency by Model and Backend")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_xlabel("CNN Architecture")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[T1] SAVED {out_path}")
    return True


# ===================================================================
# Entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="T1: Generate paper figures from CSV data")
    parser.add_argument("--microbench-csv", default="artifacts/processed/paper_table_microbench.csv")
    parser.add_argument("--e2e-csv", default="summary/e2e_comparison_table.csv")
    parser.add_argument("--plot-dir", default="artifacts/plots")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    ok1 = plot_microbench_improvement(
        args.microbench_csv,
        os.path.join(args.plot_dir, "paper_fig_microbench_improvement.png"),
    )
    ok2 = plot_microbench_ci(
        args.microbench_csv,
        os.path.join(args.plot_dir, "paper_fig_microbench_ci.png"),
    )
    ok3 = plot_e2e_latency(
        args.e2e_csv,
        os.path.join(args.plot_dir, "paper_fig_end_to_end_latency.png"),
    )

    generated = sum([ok1, ok2, ok3])
    print(f"\n[T1] Generated {generated}/3 figures.")
    if generated < 2:
        print("[T1] WARNING: Some figures could not be generated due to missing CSV data.")
        print("     Run the corresponding benchmark tasks first to produce the raw data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
