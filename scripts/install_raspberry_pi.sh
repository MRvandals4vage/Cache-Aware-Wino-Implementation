#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] install_raspberry_pi.sh failed with exit code $exit_code" >&2
    fi
}
trap cleanup EXIT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash "${ROOT_DIR}/scripts/setup_device_dependencies.sh"

echo
echo "Raspberry Pi setup complete."
echo "Run benchmarks with: make verify-all (or bash scripts/run_raspberry_pi_benchmark.sh)"
