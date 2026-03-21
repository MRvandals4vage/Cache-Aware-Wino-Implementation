#!/bin/bash
set -e

mkdir -p artifacts/logs
LOG_FILE="artifacts/logs/raspberry_pi_benchmark.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo "Raspberry Pi Benchmark Runner"
echo "Date: $(date)"
echo "========================================"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ensure Python dependencies are found
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Collecting hardware features..."
python3 runtime_cache_probe.py || echo "Warning: Cache probe failed to initialize completely."

echo "Running Cache-Aware Fused Winograd Microbenchmarks..."
# This executes the core workload that tests L1 fusion and Pi parameters without strict dependency on PyTorch
python3 benchmarks/microbenchmarks.py

echo "Generating visualizations using available plotters..."
python3 generate_microbenchmark_plots.py || echo "Warning: Optional matplotlib dependencies not fully met. Proceeding without plots."

echo "========================================"
echo "Benchmark Finished. Check artifacts/ processed for metrics."
echo "========================================"
