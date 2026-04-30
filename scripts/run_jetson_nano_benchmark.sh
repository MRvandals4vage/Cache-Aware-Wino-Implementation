#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p artifacts/logs
LOG_FILE="artifacts/logs/jetson_nano_benchmark.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

BACKENDS="${BACKENDS:-project,onnxruntime,tvm,autotvm}"
if [ -n "${ARMCL_COMMAND:-}" ]; then
    BACKENDS="${BACKENDS},armcl"
fi
COMPARE_ARGS=(
  --backends "${BACKENDS}"
  --runs 20
  --warmup 5
  --height 4
  --width 4
  --tvm-target "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a57 -mattr=+neon"
)
if [ -n "${ARMCL_COMMAND:-}" ]; then
    COMPARE_ARGS+=(--armcl-command "${ARMCL_COMMAND}")
fi

echo "========================================"
echo "Jetson Nano Benchmark Runner"
echo "Date: $(date)"
echo "Repo: ${ROOT_DIR}"
echo "Backends: ${BACKENDS}"
echo "========================================"

python3 benchmarks/run_all_benchmarks.py --mode micro --runs 30 --warmup 10 --paper-assets all
python3 tools/compare_edge_backends.py "${COMPARE_ARGS[@]}"

echo "========================================"
echo "Jetson Nano benchmark run finished"
echo "Artifacts: ${ROOT_DIR}/artifacts"
echo "========================================"
