#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p artifacts/logs
LOG_FILE="artifacts/logs/mac_benchmark.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "========================================"
echo "macOS Benchmark Runner"
echo "Date: $(date)"
echo "Repo: ${ROOT_DIR}"
echo "========================================"

python3 benchmarks/run_all_benchmarks.py --mode micro --runs 30 --warmup 10 --paper-assets all
python3 tools/compare_edge_backends.py --backends project,onnxruntime --runs 20 --warmup 5 --height 4 --width 4

echo "========================================"
echo "macOS benchmark run finished"
echo "Artifacts: ${ROOT_DIR}/artifacts"
echo "========================================"
