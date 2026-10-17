#!/usr/bin/env bash
# ==============================================================================
# scripts/analyze_all.sh
# Shell wrapper around scripts/analyze_all.py
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 scripts/analyze_all.py "$@"
