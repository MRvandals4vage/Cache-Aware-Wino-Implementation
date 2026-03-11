import argparse
from benchmark import BenchmarkRunner
from visualization import generate_bench_plots

from power_monitor import JetsonPowerMonitor

def run_model_benchmarks(model_name):
    print(f"\nRunning benchmarks for model: {model_name}")
    print("-" * 40)
    modes = ["Baseline", "Naive Winograd", "Cache-Aware", "TVM Model"]
    results = []
    
    # Initialize Power Monitor
    power_monitor = JetsonPowerMonitor()
    
    for mode in modes:
        print(f"  Mode: {mode:<20}")
        
        # Start power monitoring
        power_monitor.start_monitoring()
        
        runner = BenchmarkRunner(mode=mode, model_name=model_name)
        
        # runner.run() now includes warmup, algorithmic trace, and measurement loops
        res = runner.run() 
        
        power_avgs = power_monitor.stop_monitoring()
        
        # Update result with measured power
        res["average_power_mw"] = power_avgs["total"]
        # Re-calculate energy with final power average
        from energy_model import EnergyModel
        em = EnergyModel()
        res["Energy (mJ)"] = em.calculate_energy(res["average_power_mw"], res["time_ms"])
        res["efficiency"] = em.calculate_efficiency(res["MACs"], res["Energy (mJ)"])
        
        results.append(res)
    
    # Generate Plots for this model
    generate_bench_plots(results, model_name)
    return results

def save_markdown_table(all_results):
    with open("benchmark_results_measured.md", "w") as f:
        f.write("# Measurement-Driven CNN Benchmarking Results\n\n")
        f.write("| Architecture   | Mode                | Latency (ms) | FPS    | MACs (Measured) | DRAM (Est/Run) | Power (mW) | Energy (mJ) | MACs/J    |\n")
        f.write("| :------------- | :------------------ | :----------: | :----: | :------------: | :------------: | :--------: | :---------: | :-------: |\n")
        for r in all_results:
            # Note: r['Bytes'] is returned by BenchmarkRunner now
            f.write(f"| {r['Model']:<14} | {r['Strategy']:<19} | {r['time_ms']:>12.2f} | {r['throughput_fps']:>6.1f} | {int(r['MACs']):>14,} | {int(r['Bytes']):>14,} | {r['average_power_mw']:>10.1f} | {r['Energy (mJ)']:>11.2f} | {r['efficiency']:>9.2e} |\n")
    print("\nSummary table saved to benchmark_results_measured.md")

def main():
    parser = argparse.ArgumentParser(description="Edge AI Convolution Benchmarking")
    parser.add_argument("--model", type=str, choices=["resnet18", "vgg16", "alexnet", "resnet34", "all"], default="all", help="Model to benchmark")
    args = parser.parse_args()

    models_to_run = ["resnet18", "vgg16", "alexnet", "resnet34"] if args.model == "all" else [args.model]
    
    all_combined_results = []
    
    print("="*60)
    print("Edge AI Benchmarking: Multi-Architecture Evaluation")
    print("Evaluating Winograd Convolution Memory Scheduling Strategies")
    print("="*60)

    for model_name in models_to_run:
        model_results = run_model_benchmarks(model_name)
        all_combined_results.extend(model_results)

    save_markdown_table(all_combined_results)
    
    # Generate Architecture Comparison Plots
    print("\nGenerating Architecture Comparison Graphs...")
    generate_bench_plots(all_combined_results)
    
    print("\nBenchmarking completed successfully.")

if __name__ == "__main__":
    main()
