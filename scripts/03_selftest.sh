#!/usr/bin/env bash
# ==============================================================================
# scripts/03_selftest.sh
# Smoke test harness gate:
# - Correctness verification (max abs error, L2 error)
# - Null test (n=200 alternated, establishes board noise floor)
# - Thread engagement check
# - Path assertion validation
# Must pass on any board prior to launching full benchmark execution.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 tools/selftest_suite.py
echo "[03_selftest] PASSED."
