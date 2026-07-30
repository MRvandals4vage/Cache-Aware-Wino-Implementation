#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash "${ROOT_DIR}/scripts/setup_device_dependencies.sh"

echo
echo "Jetson Nano setup complete."
echo "Run benchmarks with: make verify-all (or bash scripts/run_jetson_nano_benchmark.sh)"
