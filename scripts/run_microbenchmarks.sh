#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] run_microbenchmarks.sh failed with exit code $exit_code" >&2
    fi
}
trap cleanup EXIT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "Running unified microbenchmark pipeline..."
python3 benchmarks/run_all_benchmarks.py --mode micro --runs 30 --warmup 10 --paper-assets all

echo "Microbenchmarking complete. Results saved to artifacts/"
