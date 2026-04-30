#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo
echo "Jetson Nano Python environment ready."
echo "Optional backends:"
echo "  TVM/AutoTVM: install TVM separately inside the same venv."
echo "  ARMCL: set ARMCL_COMMAND to a wrapper that prints LATENCY_MS=<value>."
echo "Run with: bash scripts/run_jetson_nano_benchmark.sh"
