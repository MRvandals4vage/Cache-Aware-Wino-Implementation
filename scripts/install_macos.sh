#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] install_macos.sh failed with exit code $exit_code" >&2
    fi
}
trap cleanup EXIT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo
echo "macOS environment ready."
echo "Activate with: source venv/bin/activate"
echo "Run with: bash scripts/run_mac_benchmark.sh"
