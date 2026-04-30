#!/bin/bash
set -euo pipefail

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
