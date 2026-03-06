import matplotlib.pyplot as plt
import numpy as np

def generate_bench_plots(results, model_name=""):
    """
    Generates research-grade plots for:
    1. DRAM Access vs Strategy
    2. Energy vs Strategy
    3. MAC Reduction vs Strategy
    """
    strategies = [r["Strategy"] for r in results if r["Model"] == model_name or not model_name]
    if not strategies:
        print("No results to plot.")
        return

    # Data extraction
    dram = [r["DRAM"] for r in results if r["Model"] == model_name or not model_name]
    energy = [r["Total Energy (mJ)"] for r in results if r["Model"] == model_name or not model_name]
    macs = [r["MACs"] for r in results if r["Model"] == model_name or not model_name]
    
    # 1. DRAM Access vs Strategy
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, dram, color='skyblue')
    plt.title(f'DRAM Accesses vs Execution Strategy ({model_name})')
    plt.ylabel('DRAM Accesses')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('dram_comparison.png')
    plt.close()
    
    # 2. Energy vs Strategy
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, energy, color='salmon')
    plt.title(f'Total Energy Consumption vs Execution Strategy ({model_name})')
    plt.ylabel('Energy (mJ)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('energy_breakdown.png')
    plt.close()

    # 3. MAC Reduction vs Strategy (Standardized to Baseline)
    baseline_macs = macs[0] if macs else 1
    mac_ratio = [m / baseline_macs for m in macs]
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, mac_ratio, color='lightgreen')
    plt.title(f'MAC Operations Ratio vs Strategy ({model_name})')
    plt.ylabel('Normalized MACs')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('mac_reduction.png')
    plt.close()

    print(f"Research-grade graphs (dram_comparison.png, energy_breakdown.png, mac_reduction.png) generated for {model_name}.")

if __name__ == "__main__":
    # Test plotting
    mock_results = [{"Strategy": "Baseline", "Model": "resnet18", "DRAM": 1e8, "Energy(mJ)": 50, "MACs": 1e9}]
    generate_bench_plots(mock_results, "resnet18")
