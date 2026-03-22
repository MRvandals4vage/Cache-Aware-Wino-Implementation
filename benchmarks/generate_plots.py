#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(processed_dir="artifacts/processed", plot_dir="artifacts/plots"):
    os.makedirs(plot_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    if not csv_files:
        print(f"No processed CSV files found in {processed_dir}")
        return
        
    for pf in csv_files:
        try:
            df = pd.read_csv(pf)
            if df.empty or "mean_latency_ms" not in df.columns:
                print(f"Skipping {pf} - no plottable data.")
                continue
                
            # Filter and prepare strings for x-axis labels
            df["config"] = df.apply(lambda row: f"{row['c_in']}x{row['c_out']} th:{row['threads']}", axis=1)
            
            # Plot 1: Fused vs Non-Fused Latency Comparison
            if "fused" in df.columns:
                plt.figure(figsize=(10, 6))
                
                # Plot bar chart for mean latency, split by 'fused'
                sns.barplot(
                    data=df, 
                    x="config", 
                    y="mean_latency_ms", 
                    hue="fused",
                    errorbar=None,
                    palette="muted"
                )
                
                plt.title("Fused vs Non-Fused Kernel Latency")
                plt.ylabel("Mean Latency (ms)")
                plt.xlabel("Configuration (C_in x C_out) Threads")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plot_file = os.path.join(plot_dir, "fused_vs_nonfused_" + os.path.basename(pf).replace(".csv", ".png"))
                plt.savefig(plot_file, dpi=300)
                plt.close()
                print(f"Generated plot: {plot_file}")
                
            # Plot 2: Speedup Distribution using Boxplot proxy
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df,
                x="config",
                y="p_value",
                hue="fused"
            )
            plt.axhline(0.05, ls="--", color="red", label="Significance Threshold (0.05)")
            plt.title("Statistical Significance (Welch t-test P-value vs Baseline)")
            plt.ylabel("P-value")
            plt.xlabel("Configuration")
            plt.yscale("log")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plot_file = os.path.join(plot_dir, "statistical_significance_" + os.path.basename(pf).replace(".csv", ".png"))
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Generated plot: {plot_file}")
            
        except Exception as e:
            print(f"Error plotting {pf}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate plots from processed benchmark results")
    parser.add_argument("--processed-dir", default="artifacts/processed")
    parser.add_argument("--plot-dir", default="artifacts/plots")
    args = parser.parse_args()
    
    generate_plots(args.processed_dir, args.plot_dir)
