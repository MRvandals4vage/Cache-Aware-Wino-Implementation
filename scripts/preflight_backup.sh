#!/usr/bin/env bash
# ==============================================================================
# Pre-flight backup: copy raw_logs/ -> raw_logs_prior/
#                        summary/   -> summary_prior/
# Run this FIRST, before any benchmarks, to preserve old results.
# Constraint #2: never overwrite prior data.
# ==============================================================================
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] preflight_backup.sh failed with exit code $exit_code" >&2
    fi
}
trap cleanup EXIT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Pre-flight backup ==="

for dir in raw_logs summary; do
    prior="${dir}_prior"
    if [ -d "${dir}" ] && [ "$(ls -A ${dir} 2>/dev/null | grep -v '\.gitkeep')" ]; then
        if [ -d "${prior}" ]; then
            ts=$(date +%Y%m%d_%H%M%S)
            echo "  ${prior}/ already exists — rotating to ${prior}_${ts}/"
            mv "${prior}" "${prior}_${ts}"
        fi
        echo "  Copying ${dir}/ -> ${prior}/"
        cp -r "${dir}" "${prior}"
        echo "  Done. $(find ${dir} -type f | wc -l | tr -d ' ') files backed up."
    else
        echo "  ${dir}/ is empty or missing — nothing to back up."
        mkdir -p "${prior}"
    fi
done

echo ""
echo "=== Backup complete. Safe to start benchmarks. ==="
echo "    raw_logs_prior/ and summary_prior/ preserved."
