import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plots():
    csv_path = "artifacts/processed/microbenchmark_results.csv"
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Plot 1: Mean Latency Bar Chart showing configuration differences
    plt.figure(figsize=(10, 6))
    
    df_plot = df[df['MultiCore'] == False].copy()
    df_plot['Config_name'] = df_plot.apply(lambda r: f"{r['C_in']}x{r['C_out']}", axis=1)
    
    x_labels = df_plot['Config_name'].unique()
    x = np.arange(len(x_labels))
    width = 0.35
    
    fused_vals = df_plot[df_plot['Fused'] == True].sort_values('Config_name')['Mean_Latency_ms'].values
    nofused_vals = df_plot[df_plot['Fused'] == False].sort_values('Config_name')['Mean_Latency_ms'].values
    
    plt.bar(x - width/2, nofused_vals, width, label='Baseline')
    plt.bar(x + width/2, fused_vals, width, label='Fused Winograd')
    
    plt.xticks(x, x_labels)
    plt.title('Fused vs Non-Fused Cache-Aware Winograd Latency (Single Core)')
    plt.ylabel('Mean Latency (ms)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/fused_vs_baseline_latency.png')
    
    # Create an identifier column for detailed plotting
    df['Config'] = df.apply(lambda r: f"Cin={r['C_in']} Cout={r['C_out']} Fused={r['Fused']} Multi={r['MultiCore']}", axis=1)
    
    # Plot 2: Detailed error bar plot of all configs
    plt.figure(figsize=(12, 8))
    x = np.arange(len(df))
    plt.errorbar(x, df['Mean_Latency_ms'], yerr=df['Conf_Interval_95'], fmt='o', capsize=5, capthick=2)
    plt.xticks(x, df['Config'], rotation=90)
    plt.ylabel('Latency (ms)')
    plt.title('Winograd Latency Across All Configurations (with 95% Confidence Intervals)')
    plt.tight_layout()
    plt.savefig('artifacts/config_latency_ci_plot.png')
    
    print("Plots generated successfully in artifacts/")

if __name__ == '__main__':
    generate_plots()
