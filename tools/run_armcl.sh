#!/usr/bin/env bash
# ==============================================================================
# tools/run_armcl.sh
#
# Standalone ArmCL Benchmark Runner
# Expects either positional arguments:
#   run_armcl.sh <c_in> <c_out> <height> <width> [kernel] [runs] [warmup]
# Or flags:
#   run_armcl.sh --cin 64 --cout 64 --height 56 --width 56 --kernel 3 --runs 10 --warmup 2
#
# Prints LATENCY_MS=<value> to stdout for downstream parsing.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/bin"
BENCH_BIN="${BIN_DIR}/armcl_benchmark"
SRC_CPP="${ROOT_DIR}/tools/armcl_benchmark.cpp"

mkdir -p "${BIN_DIR}"

# 1. Compile C++ ArmCL benchmark if binary not present
if [ ! -f "${BENCH_BIN}" ]; then
    CXX="${CXX:-g++}"
    CXXFLAGS="-O3 -std=c++14"
    ARCH="$(uname -m)"

    # Check for native Arm Compute Library installation
    ARMCL_INCLUDE=""
    ARMCL_LIBS=""
    for inc_path in "/usr/include" "/usr/local/include" "/opt/arm_compute/include" "${ROOT_DIR}/third_party/armcl/include"; do
        if [ -f "${inc_path}/arm_compute/runtime/NEON/NEFunctions.h" ]; then
            ARMCL_INCLUDE="-I${inc_path} -DHAS_ARM_COMPUTE=1"
            break
        fi
    done

    for lib_path in "/usr/lib" "/usr/local/lib" "/opt/arm_compute/lib" "${ROOT_DIR}/third_party/armcl/lib"; do
        if [ -f "${lib_path}/libarm_compute.so" ] || [ -f "${lib_path}/libarm_compute.a" ]; then
            ARMCL_LIBS="-L${lib_path} -larm_compute -larm_compute_core -lpthread"
            break
        fi
    done

    if [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "armv7l" ]]; then
        CXXFLAGS="${CXXFLAGS} -march=native"
    fi

    # Attempt compilation
    if ${CXX} ${CXXFLAGS} ${ARMCL_INCLUDE} -o "${BENCH_BIN}" "${SRC_CPP}" ${ARMCL_LIBS} 2>/dev/null; then
        :
    else
        # Try compiling standalone without external ArmCL shared library
        ${CXX} ${CXXFLAGS} -o "${BENCH_BIN}" "${SRC_CPP}" -lpthread 2>/dev/null || true
    fi
fi

# 2. Run benchmark executable if built successfully
if [ -x "${BENCH_BIN}" ]; then
    "${BENCH_BIN}" "$@"
    exit 0
fi

# 3. Fallback to Python runner
exec python3 "${ROOT_DIR}/tools/armcl_fallback_runner.py" "$@"
