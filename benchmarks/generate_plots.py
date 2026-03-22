#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_microbenchmarks(processed_csv, out_dir):
    if not os.path.exists(processed_csv):
        print(f"File {processed_csv} not found, skipping microbenchmark plots.")
        return
        
    df = pd.read_csv(processed_csv)
    if df.empty:
        return
        
    df["Config"] = df.apply(lambda row: f"{row['C_in']}x{row['C_out']}", axis=1)
    
    # 1. Latency Plot (paper_fig_microbench_latency.png)
    plt.figure(figsize=(10, 6))
    df["Mode"] = df.apply(lambda row: f"Fused={row['Fused']}, Threads={'Multi' if row['MultiCore'] else 'Single'}", axis=1)
    
    sns.barplot(data=df, x="Config", y="Mean_Latency_ms", hue="Mode", palette="viridis")
    plt.title("Microbenchmark Configuration Latency")
    plt.ylabel("Mean Latency (ms)")
    plt.xlabel("Layer Configuration (C_in x C_out)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_fig_microbench_latency.png"), dpi=300)
    plt.close()
    
    # 2. Improvement Bar Plot (paper_fig_microbench_improvement.png)
    imp_df = df[df["Improvement_vs_Baseline_pct"] != "N/A"].copy()
    if not imp_df.empty:
        imp_df["Improvement_vs_Baseline_pct"] = imp_df["Improvement_vs_Baseline_pct"].astype(float)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp_df, x="Config", y="Improvement_vs_Baseline_pct", hue="Mode", palette="coolwarm")
        plt.axhline(0, color='black', linewidth=1)
        plt.title("Percentage Improvement over Baseline (Higher is Better)")
        plt.ylabel("Improvement (%)")
        plt.xlabel("Layer Configuration (C_in x C_out)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "paper_fig_microbench_improvement.png"), dpi=300)
        plt.close()

def plot_e2e(processed_csv, out_dir):
    if not os.path.exists(processed_csv):
        print(f"File {processed_csv} not found, skipping E2E plots.")
        return
        
    df = pd.read_csv(processed_csv)
    if df.empty:
        return
        
    # 1. End to End Latency (paper_fig_end_to_end_latency.png)
    plt.figure(figsize=(8, 5))
    if not df.empty and "Model" in df.columns:
        sns.barplot(data=df, x="Model", y="Mean_Latency_ms", hue="Model", palette="Set2", legend=False)
    plt.title("End-to-End Latency by Model")
    plt.ylabel("Mean Latency (ms)")
    plt.xlabel("CNN Architecture")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_fig_end_to_end_latency.png"), dpi=300)
    plt.close()
    
    # 2. End to End FPS (paper_fig_end_to_end_fps.png)
    if "FPS" in df.columns and "Model" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Model", y="FPS", hue="Model", palette="Set3", legend=False)
        plt.title("End-to-End Throughput by Model")
        plt.ylabel("Frames Per Second (FPS)")
        plt.xlabel("CNN Architecture")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "paper_fig_end_to_end_fps.png"), dpi=300)
        plt.close()

def plot_autotiling(processed_csv, out_dir):
    if not os.path.exists(processed_csv):
        print(f"File {processed_csv} not found, skipping Autotiling plots.")
        return
        
    df = pd.read_csv(processed_csv)
    if df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Workload",
        y="Reuse_Score",
        size="Selected_Tile",
        sizes=(100, 500),
        hue="Selected_Tile",
        palette="deep",
        legend="full"
    )
    plt.title("Autotiler Policy vs Strategy Reuse Score")
    plt.ylabel("Reuse Score")
    plt.xlabel("Workload")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_fig_autotiling_decisions.png"), dpi=300)
    plt.close()

def run_all(processed_dir="artifacts/processed", plot_dir="artifacts/plots"):
    os.makedirs(plot_dir, exist_ok=True)
    plot_microbenchmarks(os.path.join(processed_dir, "paper_table_microbench.csv"), plot_dir)
    plot_e2e(os.path.join(processed_dir, "paper_table_end_to_end.csv"), plot_dir)
    plot_autotiling(os.path.join(processed_dir, "paper_table_autotiling.csv"), plot_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="artifacts/processed")
    parser.add_argument("--plot-dir", default="artifacts/plots")
    args = parser.parse_args()
    run_all(args.processed_dir, args.plot_dir)
