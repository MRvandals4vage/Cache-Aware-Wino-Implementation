"""
Unified Research Graph Generation Script
Generates comparison plots for DRAM, Energy, and MAC Efficiency.
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os
from benchmark import BenchmarkRunner  # type: ignore


def get_data():
    """Gathers data across models and strategies for plotting."""
    models = ["resnet18", "vgg16", "alexnet", "resnet34"]
    strategies = ["Baseline", "Naive Winograd", "Cache-Aware", "TVM Model"]
    
    all_data = []
    for model in models:
        for strat in strategies:
            runner = BenchmarkRunner(mode=strat, model_name=model)
            res = runner.run(average_power_mw=5000.0) # Assume 5W for Jetson Nano
            # Standardizing keys for plot consistency
            all_data.append({
                "Model": model,
                "Strategy": strat,
                "DRAM": res["Bytes"] / 4,
                "Energy": res["Energy (mJ)"],
                "MACs": res["MACs"],
                "Latency": res["time_ms"]
            })
    return all_data

def plot_comparison():
    data = get_data()
    models = sorted(list(set(d["Model"] for d in data)))
    strategies = ["Baseline", "Naive Winograd", "Cache-Aware", "TVM Model"]
    
    # Use a clean, neat style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
        
    colors = plt.cm.tab10.colors # Use distinct colors
    
    # Pre-calculate positions for 4 models to avoid congestion/overlap
    x = np.arange(len(strategies))
    width = 0.2
    # Offsets for 4 items: -0.3, -0.1, 0.1, 0.3
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    def format_plot(title, ylabel, log_scale=False):
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, pad=15, fontweight='bold')
        plt.xticks(x, strategies, fontsize=11)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        if log_scale:
            plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

    # 1. DRAM Accesses Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        dram_vals = [d["DRAM"] for d in data if d["Model"] == model]
        plt.bar(x + offsets[i], dram_vals, width, label=model.capitalize(), color=colors[i], edgecolor='black', linewidth=0.5)
    
    format_plot('DRAM Traffic Analysis: Comparison of Scheduling Strategies', 'DRAM Accesses', log_scale=True)
    plt.savefig('dram_comparison_unified.png', dpi=300, bbox_inches='tight')
    print("Generated: dram_comparison_unified.png")

    # 2. Energy Consumption Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        energy_vals = [d["Energy"] for d in data if d["Model"] == model]
        plt.bar(x + offsets[i], energy_vals, width, label=model.capitalize(), color=colors[i], edgecolor='black', linewidth=0.5)
    
    format_plot('Energy Consumption per Inference', 'Energy (mJ)')
    plt.savefig('energy_comparison_unified.png', dpi=300, bbox_inches='tight')
    print("Generated: energy_comparison_unified.png")

    # 3. MAC Efficiency (MACs per Joule) (Bar Chart)
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        efficiency = [float(d["MACs"]) / (float(d["Energy"])/1000) if float(d["Energy"]) > 0 else 0.0 for d in data if d["Model"] == model]
        # Replace 0s with a small epsilon to avoid log scale warnings
        efficiency = [e if e > 0 else 1e-5 for e in efficiency]
        plt.bar(x + offsets[i], efficiency, width, label=model.capitalize(), color=colors[i], edgecolor='black', linewidth=0.5)
    
    format_plot('Energy Efficiency (MACs/J)', 'MACs / Joule', log_scale=True)
    plt.savefig('mac_reduction_unified.png', dpi=300, bbox_inches='tight')
    print("Generated: mac_reduction_unified.png")

    # 4. Latency Trend Across Optimizations (Line Graph)
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        latency_vals = [d["Latency"] for d in data if d["Model"] == model]
        plt.plot(x, latency_vals, marker='o', linewidth=2.5, markersize=8, label=model.capitalize(), color=colors[i])
    
    format_plot('Inference Latency Trend Across Optimizations', 'Latency (ms)')
    plt.savefig('latency_trend_line.png', dpi=300, bbox_inches='tight')
    print("Generated: latency_trend_line.png")

    # 5. Energy vs Latency Trade-off (Line Graph)
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        energy_vals = [d["Energy"] for d in data if d["Model"] == model]
        plt.plot(x, energy_vals, marker='s', linewidth=2.5, markersize=8, linestyle='--', label=model.capitalize(), color=colors[i])
    
    format_plot('Energy Consumption Trend Across Optimizations', 'Energy (mJ)')
    plt.savefig('energy_trend_line.png', dpi=300, bbox_inches='tight')
    print("Generated: energy_trend_line.png")

if __name__ == "__main__":
    print("Running unified graph generator...")
    plot_comparison()
    print("All comparison graphs generated successfully.")
