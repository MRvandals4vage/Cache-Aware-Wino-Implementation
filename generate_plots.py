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
    
    # 1. DRAM Accesses Comparison
    plt.figure(figsize=(12, 7))
    x = np.arange(len(strategies))
    width = 0.35
    
    for i, model in enumerate(models):
        dram_vals = [d["DRAM"] for d in data if d["Model"] == model]
        plt.bar(x + (i * width) - width/2, dram_vals, width, label=f'Model: {model}')
    
    plt.ylabel('DRAM Accesses')
    plt.title('DRAM Traffic Analysis: Comparison of Scheduling Strategies')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('dram_comparison_unified.png', dpi=300)
    print("Generated: dram_comparison_unified.png")

    # 2. Energy Consumption Comparison
    plt.figure(figsize=(12, 7))
    for i, model in enumerate(models):
        energy_vals = [d["Energy"] for d in data if d["Model"] == model]
        plt.bar(x + (i * width) - width/2, energy_vals, width, label=f'Model: {model}')
    
    plt.ylabel('Energy (mJ)')
    plt.title('Energy Consumption per Inference')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('energy_comparison_unified.png', dpi=300)
    print("Generated: energy_comparison_unified.png")

    # 3. MAC Efficiency (MACs per Joule)
    plt.figure(figsize=(12, 7))
    for i, model in enumerate(models):
        efficiency = [float(d["MACs"]) / (float(d["Energy"])/1000) if float(d["Energy"]) > 0 else 0 for d in data if d["Model"] == model]
        plt.bar(x + (i * width) - width/2, efficiency, width, label=f'Model: {model}')
    
    plt.ylabel('MACs / Joule')
    plt.yscale('log')
    plt.title('Energy Efficiency (MACs/J) - Log Scale')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    # Save a second version specifically for the readme requirement
    plt.savefig('mac_reduction_unified.png', dpi=300)
    print("Generated: mac_reduction_unified.png")

if __name__ == "__main__":
    print("Running unified graph generator...")
    plot_comparison()
    print("All comparison graphs generated successfully.")
