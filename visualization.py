import matplotlib.pyplot as plt
import numpy as np

def generate_bench_plots(results, model_name=""):
    """
    Generates research-grade plots for:
    1. Latency (ms) vs Strategy
    2. Energy (mJ) vs Strategy
    3. Memory Traffic (Bytes) vs Strategy
    """
    strategies = [r["Strategy"] for r in results if r["Model"] == model_name or not model_name]
    if not strategies:
        print("No results to plot.")
        return

    # Filenames as requested
    latency_fn = 'latency_comparison.png'
    energy_fn = 'energy_comparison.png'
    memory_fn = 'memory_traffic_comparison.png'

    # Color mapping for strategies
    color_map = {
        "Baseline": "#3498db",      # Blue
        "Naive Winograd": "#e74c3c",# Red
        "Cache-Aware": "#2ecc71",   # Green
        "TVM Model": "#9b59b6"      # Purple
    }

    unique_models = sorted(list(set(r["Model"] for r in results)))
    x = np.arange(len(unique_models))
    width = 0.2
    strats = ["Baseline", "Naive Winograd", "Cache-Aware", "TVM Model"]

    # 1. Latency Comparison
    plt.figure(figsize=(12, 7))
    for i, strat in enumerate(strats):
        vals = [next((r["time_ms"] for r in results if r["Model"] == m and r["Strategy"] == strat or r["Strategy"].startswith(strat)), 0) for m in unique_models]
        plt.bar(x + (i*width) - (len(strats)-1)*width/2, vals, width, label=strat, color=color_map.get(strat))
    plt.xticks(x, unique_models)
    plt.ylabel('Latency (ms)')
    plt.title('Hardware-Measured Inference Latency across Architectures')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(latency_fn, dpi=300)
    plt.close()

    # 2. Energy Consumption Comparison
    plt.figure(figsize=(12, 7))
    for i, strat in enumerate(strats):
        vals = [next((r["Energy (mJ)"] for r in results if r["Model"] == m and r["Strategy"] == strat or r["Strategy"].startswith(strat)), 0) for m in unique_models]
        plt.bar(x + (i*width) - (len(strats)-1)*width/2, vals, width, label=strat, color=color_map.get(strat))
    plt.xticks(x, unique_models)
    plt.ylabel('Energy (mJ)')
    plt.title('Total Inference Energy (mW * ms / 1000)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(energy_fn, dpi=300)
    plt.close()

    # 3. Memory Traffic Comparison
    plt.figure(figsize=(12, 7))
    for i, strat in enumerate(strats):
        # We now plot 'Bytes' which is the emergent metric
        vals = [next((r["Bytes"] for r in results if r["Model"] == m and r["Strategy"] == strat or r["Strategy"].startswith(strat)), 0) for m in unique_models]
        plt.bar(x + (i*width) - (len(strats)-1)*width/2, vals, width, label=strat, color=color_map.get(strat))
    plt.xticks(x, unique_models)
    plt.ylabel('Memory Traffic (Total Bytes Traversed)')
    plt.title('Memory Traffic Comparison via Algorithmic Trace')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(memory_fn, dpi=300)
    plt.close()

    print(f"Research-grade graphs ({latency_fn}, {energy_fn}, {memory_fn}) generated.")
