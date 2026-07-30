#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] run_raspberry_pi_benchmark.sh failed with exit code $exit_code" >&2
    fi
}
trap cleanup EXIT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p artifacts/logs
LOG_FILE="artifacts/logs/raspberry_pi_benchmark.log"
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
  --tvm-target "llvm -mattr=+neon"
)
if [ -n "${ARMCL_COMMAND:-}" ]; then
    COMPARE_ARGS+=(--armcl-command "${ARMCL_COMMAND}")
fi

echo "========================================"
echo "Raspberry Pi Benchmark Runner"
echo "Date: $(date)"
echo "Repo: ${ROOT_DIR}"
echo "Backends: ${BACKENDS}"
echo "========================================"

python3 benchmarks/run_all_benchmarks.py --mode micro --runs 30 --warmup 10 --paper-assets all
python3 tools/compare_edge_backends.py "${COMPARE_ARGS[@]}"

echo "========================================"
echo "Raspberry Pi benchmark run finished"
echo "Artifacts: ${ROOT_DIR}/artifacts"
echo "========================================"
