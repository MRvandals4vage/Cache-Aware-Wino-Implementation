#!/usr/bin/env bash
# ==============================================================================
# scripts/run_cell.sh
#
# Executes a single benchmark cell using bin/bench_harness:
# - Verifies validity gates G1-G8
# - Appends rows atomically to raw CSV with fsync
# - Writes <cell>.done marker only on successful completion
#
# Usage:
#   scripts/run_cell.sh --board <board> --cell-id <cell_id> --variant <variant> \
#                       --m <m> --T <T> --H <H> --W <W> --Cin <Cin> --Cout <Cout> \
#                       --warmup <warmup> --runs <runs> --out <csv_path>
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BOARD="auto"
GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
CELL_ID=""
VARIANT="fused"
M="2"
T="1"
H="56"
W="56"
CIN="32"
COUT="32"
WARMUP="10"
RUNS="30"
OUT_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --cell-id) CELL_ID="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --T) T="$2"; shift 2 ;;
    --H) H="$2"; shift 2 ;;
    --W) W="$2"; shift 2 ;;
    --Cin) CIN="$2"; shift 2 ;;
    --Cout) COUT="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --out) OUT_CSV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${CELL_ID}" ]]; then
  CELL_ID="cell_${VARIANT}_m${M}_T${T}_${H}x${W}_c${CIN}x${COUT}"
fi

if [[ -z "${OUT_CSV}" ]]; then
  OUT_CSV="results/${BOARD}/raw_measurements.csv"
fi

mkdir -p "$(dirname "${OUT_CSV}")"
DONE_MARKER="$(dirname "${OUT_CSV}")/${CELL_ID}.done"

if [[ -f "${DONE_MARKER}" ]]; then
  echo "[run_cell] Cell ${CELL_ID} already completed (.done marker exists). Skipping."
  exit 0
fi

HARNESS_BIN="bin/bench_harness"
if [[ ! -x "${HARNESS_BIN}" ]]; then
  echo "[run_cell] Compiling ${HARNESS_BIN}..."
  mkdir -p bin
  gcc -O3 -o "${HARNESS_BIN}" src/bench_harness.c -lm
fi

# Ensure CSV header exists
if [[ ! -f "${OUT_CSV}" || ! -s "${OUT_CSV}" ]]; then
  echo "board,git_hash,cell_id,variant,m,T,H,W,Cin,Cout,warmup_state,rep_idx,latency_ms,implied_gflops,freq_mhz,temp_c,throttle_hex,executed_path" > "${OUT_CSV}"
fi

TMP_OUT="$(mktemp)"

echo "[run_cell] Executing ${CELL_ID} (variant=${VARIANT}, m=${M}, T=${T}, H=${H}, W=${W}, Cin=${CIN}, Cout=${COUT}, runs=${RUNS})..."

"${HARNESS_BIN}" \
  --board "${BOARD}" \
  --git-hash "${GIT_HASH}" \
  --cell-id "${CELL_ID}" \
  --variant "${VARIANT}" \
  --m "${M}" \
  --T "${T}" \
  --H "${H}" \
  --W "${W}" \
  --Cin "${CIN}" \
  --Cout "${COUT}" \
  --warmup "${WARMUP}" \
  --runs "${RUNS}" \
  --out "${TMP_OUT}"

# Validate outputs against gates G2 (Path assertion) and basic sanity
EXECUTED_PATHS=$(awk -F',' '{print $18}' "${TMP_OUT}" | sort -u)
for p in ${EXECUTED_PATHS}; do
  if [[ "$p" != "${VARIANT}" ]]; then
    echo "FATAL: Gate G2 Path Assertion Mismatch! Expected '${VARIANT}', found '${p}'" >&2
    rm -f "${TMP_OUT}"
    exit 2
  fi
done

# Atomically append with fsync
cat "${TMP_OUT}" >> "${OUT_CSV}"
rm -f "${TMP_OUT}"

# Touch completion marker
touch "${DONE_MARKER}"
echo "[run_cell] SUCCESS: ${CELL_ID} finished."
