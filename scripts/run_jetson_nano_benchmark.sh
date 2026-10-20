#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] run_jetson_nano_benchmark.sh failed with exit code $exit_code" >&2
    fi
}
trap cleanup EXIT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p artifacts/logs
LOG_FILE="artifacts/logs/jetson_nano_benchmark.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

BACKENDS="${BACKENDS:-project,onnxruntime,tvm,autotvm,armcl}"
ARMCL_COMMAND="${ARMCL_COMMAND:-${ROOT_DIR}/tools/run_armcl.sh {c_in} {c_out} {height} {width} {kernel} {runs} {warmup}}"
export ARMCL_COMMAND

COMPARE_ARGS=(
  --backends "${BACKENDS}"
  --runs 20
  --warmup 5
  --height 4
  --width 4
  --tvm-target "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a57 -mattr=+neon"
  --armcl-command "${ARMCL_COMMAND}"
)

echo "========================================"
echo "Jetson Nano Benchmark Runner"
echo "Date: $(date)"
echo "Repo: ${ROOT_DIR}"
echo "Backends: ${BACKENDS}"
echo "========================================"

# 1. Run core microbenchmarks
python3 benchmarks/run_all_benchmarks.py --mode micro --runs 40 --warmup 10 --paper-assets all

# 2. Run backend comparisons (Project, ORT, TVM, AutoTVM, ArmCL)
python3 tools/compare_edge_backends.py "${COMPARE_ARGS[@]}"

# 3. Run TVM, AutoTVM, and ArmCL statistical rigor suite (T3)
echo "========================================"
echo "Running TVM, AutoTVM & ArmCL Statistical Rigor (Task D)..."
echo "========================================"
python3 benchmarks/tvm_armcl_rigor.py --runs 50 --warmup 10 --armcl-command "${ARMCL_COMMAND}"

echo "========================================"
echo "Jetson Nano benchmark run finished"
echo "Artifacts: ${ROOT_DIR}/artifacts"
echo "========================================"
