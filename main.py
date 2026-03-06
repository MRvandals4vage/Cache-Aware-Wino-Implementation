import argparse
from benchmark import BenchmarkRunner
from visualization import generate_bench_plots

def run_model_benchmarks(model_name):
    print(f"\nRunning benchmarks for model: {model_name}")
    print("-" * 40)
    modes = ["Baseline", "Naive Winograd", "Cache-Aware", "TVM Model"]
    results = []
    for mode in modes:
        print(f"  Mode: {mode:<20}")
        runner = BenchmarkRunner(mode=mode, model_name=model_name)
        res = runner.run()
        results.append(res)
    
    # Generate Plots for this model
    generate_bench_plots(results, model_name)
    return results

def save_markdown_table(all_results):
    with open("benchmark_results.md", "w") as f:
        f.write("# Convolution Strategy Benchmarking Results\n\n")
        f.write("| Model          | Mode                | Time (ms) | MACs         | DRAM Accesses | Energy (mJ) | MACs/J    |\n")
        f.write("| :------------- | :------------------ | :-------: | :----------: | :-----------: | :---------: | :-------: |\n")
        for r in all_results:
            f.write(f"| {r['Model']:<14} | {r['Strategy']:<19} | {r['time_ms']:>9.2f} | {int(r['MACs']):>12,} | {int(r['DRAM']):>13,} | {r['Total Energy (mJ)']:>11.2f} | {r['efficiency']:>9.2e} |\n")
    print("\nSummary table saved to benchmark_results.md")

def main():
    parser = argparse.ArgumentParser(description="Edge AI Convolution Benchmarking")
    parser.add_argument("--model", type=str, choices=["resnet18", "mobilenetv2", "all"], default="all", help="Model to benchmark")
    args = parser.parse_args()

    models_to_run = ["resnet18", "mobilenetv2"] if args.model == "all" else [args.model]
    
    all_combined_results = []
    
    print("="*60)
    print("Edge AI Benchmarking: ResNet-18 vs MobileNetV2")
    print("Evaluating Winograd Convolution Memory Scheduling Strategies")
    print("="*60)

    for model_name in models_to_run:
        model_results = run_model_benchmarks(model_name)
        all_combined_results.extend(model_results)

    save_markdown_table(all_combined_results)
    print("\nBenchmarking completed successfully.")

if __name__ == "__main__":
    main()
