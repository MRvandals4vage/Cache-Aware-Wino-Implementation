#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash "${ROOT_DIR}/scripts/setup_device_dependencies.sh"

echo
echo "Raspberry Pi setup complete."
echo "Run benchmarks with: make verify-all (or bash scripts/run_raspberry_pi_benchmark.sh)"
