#!/bin/bash
set -e

# Rebuild the C extension natively if possible
echo "Compiling fused_winograd C extension..."
gcc -O3 -shared -fPIC -o fused_winograd.so fused_winograd.c || echo "Native compilation failed. Python fallback will be used."

echo "Running microbenchmarks..."
python3 benchmarks/microbenchmarks.py

echo "Generating plots..."
python3 generate_microbenchmark_plots.py

echo "Microbenchmarking complete. Results saved to artifacts/"
