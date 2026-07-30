#!/usr/bin/env bash
# ==============================================================================
# Archive Old Raw Benchmark Data & Reset Fresh Directories
# For Jetson Nano & Raspberry Pi Runs
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OLD_RAWS_DIR="${ROOT_DIR}/old_raws/archive_${TIMESTAMP}"

echo "======================================================================"
echo "          CacheWinograd: Archive Old Raws & Reset Directories          "
echo "======================================================================"
echo "Archive destination: ${OLD_RAWS_DIR}"
echo

# 1. Create target archive subdirectories
mkdir -p "${OLD_RAWS_DIR}/raw_logs"
mkdir -p "${OLD_RAWS_DIR}/artifacts_raw"
mkdir -p "${OLD_RAWS_DIR}/summary"
mkdir -p "${OLD_RAWS_DIR}/processed"
mkdir -p "${OLD_RAWS_DIR}/plots"
mkdir -p "${OLD_RAWS_DIR}/logs"

# Function to move directory contents safely if non-empty
archive_dir_contents() {
    local src_dir="$1"
    local dest_dir="$2"
    if [ -d "${src_dir}" ] && [ "$(ls -A "${src_dir}" 2>/dev/null)" ]; then
        echo "  [Archive] Moving files from ${src_dir} -> ${dest_dir}..."
        cp -r "${src_dir}"/* "${dest_dir}/" 2>/dev/null || true
        rm -rf "${src_dir:?}"/*
    else
        echo "  [Archive] Directory ${src_dir} is empty or missing, skipping."
    fi
}

# 2. Archive existing raw and processed outputs
archive_dir_contents "raw_logs" "${OLD_RAWS_DIR}/raw_logs"
archive_dir_contents "artifacts/raw" "${OLD_RAWS_DIR}/artifacts_raw"
archive_dir_contents "summary" "${OLD_RAWS_DIR}/summary"
archive_dir_contents "artifacts/processed" "${OLD_RAWS_DIR}/processed"
archive_dir_contents "artifacts/plots" "${OLD_RAWS_DIR}/plots"
archive_dir_contents "artifacts/logs" "${OLD_RAWS_DIR}/logs"

# 3. Clean up root old raw copies if any exist
if [ -d "old_raws" ]; then
    echo "  [Archive] Active archive stored under: ${ROOT_DIR}/old_raws"
fi

# 4. Re-create fresh, empty test directories for upcoming runs
mkdir -p raw_logs summary artifacts/raw artifacts/processed artifacts/plots artifacts/logs

# Ensure a fresh blank placeholder in raw_logs and artifacts/raw
touch raw_logs/.gitkeep summary/.gitkeep artifacts/raw/.gitkeep artifacts/processed/.gitkeep artifacts/plots/.gitkeep artifacts/logs/.gitkeep

echo
echo "======================================================================"
echo "Archive Complete!"
echo "  - All prior raw logs & summaries preserved in: ${OLD_RAWS_DIR}"
echo "  - Clean, fresh folders ready for device benchmarks:"
echo "      * raw_logs/"
echo "      * summary/"
echo "      * artifacts/raw/"
echo "      * artifacts/processed/"
echo "      * artifacts/plots/"
echo "      * artifacts/logs/"
echo "======================================================================"
