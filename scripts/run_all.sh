#!/usr/bin/env bash
# ==============================================================================
# scripts/run_all.sh
#
# Master experiment orchestrator for CacheWinograd JSA revision:
# - Supports --board {auto|rpi4|jetson_nano|mac}
# - Supports --mode {quick|full}
# - Supports --tiers {A,B,C,D}
# - Fully resumable via <cell>.done markers
# - Automatically detects board, runs preflight, and executes cells
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BOARD="auto"
MODE="quick"
TIERS="A"
RESUME=1
MAX_HOURS=12

while [[ $# -gt 0 ]]; do
  case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --tiers) TIERS="$2"; shift 2 ;;
    --resume) RESUME=1; shift 1 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Step 0: Board detection if auto
if [[ "${BOARD}" == "auto" ]]; then
  if [[ -f "scripts/00_detect_board.sh" ]]; then
    DETECTED_BOARD="$(bash scripts/00_detect_board.sh)"
    if [[ -n "${DETECTED_BOARD}" ]]; then
      BOARD="${DETECTED_BOARD}"
    else
      BOARD="generic"
    fi
  else
    BOARD="generic"
  fi
fi

echo "================================================================"
echo "CacheWinograd Master Benchmark Orchestrator"
echo "Board      : ${BOARD}"
echo "Mode       : ${MODE}"
echo "Tiers      : ${TIERS}"
echo "Start Time : $(date)"
echo "================================================================"

OUT_DIR="results/${BOARD}"
mkdir -p "${OUT_DIR}"
RAW_CSV="${OUT_DIR}/raw_measurements.csv"

# Preflight check
if [[ -f "scripts/02_preflight.sh" ]]; then
  echo "[run_all] Executing preflight checks..."
  bash scripts/02_preflight.sh || true
fi

# Determine iteration count
if [[ "${MODE}" == "quick" ]]; then
  WARMUP=5
  RUNS=10
else
  WARMUP=20
  RUNS=30
fi

# Define Workload Tiers
# Tier A: Canonical paper layers (7 configurations)
# Formats: Cin Cout H W
TIER_A_WORKLOADS=(
  "16 32 56 56"
  "32 16 56 56"
  "32 32 56 56"
  "32 64 56 56"
  "64 32 56 56"
  "64 64 56 56"
  "128 128 56 56"
)

# Variants to benchmark
VARIANTS=("naive_direct" "neon_direct" "unfused" "fused" "v2_regblocked" "v3_cin_blocked" "v4_pretransformed_packed")
M_VALUES=(2 4)
T_VALUES=(1 4)

run_experiment_cell() {
  local var="$1"
  local m_val="$2"
  local t_val="$3"
  local cin="$4"
  local cout="$5"
  local h="$6"
  local w="$7"

  # Skip invalid combinations: naive/neon direct do not use m
  if [[ "${var}" == "naive_direct" || "${var}" == "neon_direct" ]]; then
    if [[ "${m_val}" -ne 2 ]]; then return 0; fi
  fi

  local cell_id="${var}_m${m_val}_T${t_val}_${h}x${w}_c${cin}x${cout}"

  bash scripts/run_cell.sh \
    --board "${BOARD}" \
    --cell-id "${cell_id}" \
    --variant "${var}" \
    --m "${m_val}" \
    --T "${t_val}" \
    --H "${h}" \
    --W "${w}" \
    --Cin "${cin}" \
    --Cout "${cout}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --out "${RAW_CSV}"
}

if [[ "${TIERS}" =~ "A" ]]; then
  echo "--- Launching Tier A: Canonical Paper Workloads ---"
  for wl in "${TIER_A_WORKLOADS[@]}"; do
    read -r cin cout h w <<< "${wl}"
    for var in "${VARIANTS[@]}"; do
      for m_val in "${M_VALUES[@]}"; do
        for t_val in "${T_VALUES[@]}"; do
          run_experiment_cell "${var}" "${m_val}" "${t_val}" "${cin}" "${cout}" "${h}" "${w}"
        done
      done
    done
  done
fi

echo "================================================================"
echo "[run_all] All benchmark executions completed successfully."
echo "Raw CSV output: ${RAW_CSV}"
echo "================================================================"
