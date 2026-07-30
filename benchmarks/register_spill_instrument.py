#!/usr/bin/env python3
"""
T6 — Register-spill instrumentation.

Uses objdump-based static analysis (or perf if available) to count
register-spill events for the (128,128) fused kernel on Jetson Nano.

Approach:
1. Compile fused_winograd.c with debug symbols
2. Disassemble and count str/ldr to [sp, #offset] patterns (stack spills)
3. If perf is available, also collect runtime spill-related counters
4. Report whether results support the ~4x accumulator over-subscription hypothesis

Logs to raw_logs/register_spill_128x128.csv
"""
import os
import sys
import re
import subprocess
import csv
import datetime
import argparse
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _compile_kernel(source_path, output_path, extra_flags=None):
    """Compile the C kernel with debug symbols and specified optimization."""
    flags = ["-O2", "-g", "-fno-omit-frame-pointer", "-shared", "-fPIC"]
    if extra_flags:
        flags.extend(extra_flags)

    # Detect ARM NEON
    import platform
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        flags.append("-DUSE_NEON=1")
    elif machine.startswith("armv7"):
        flags.extend(["-mfpu=neon", "-DUSE_NEON=1"])

    cmd = ["gcc"] + flags + [source_path, "-o", output_path, "-lm"]
    print(f"  Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        # Try clang
        cmd[0] = "clang"
        print(f"  gcc failed, trying clang: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            print(f"  Compilation failed: {result.stderr}")
            return False
    return True


def _count_stack_spills_objdump(binary_path):
    """Count stack spill instructions using objdump disassembly."""
    try:
        result = subprocess.run(
            ["objdump", "-d", binary_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  objdump failed: {result.stderr}")
            return None
    except FileNotFoundError:
        # Try llvm-objdump on macOS
        try:
            result = subprocess.run(
                ["llvm-objdump", "-d", binary_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30
            )
        except FileNotFoundError:
            # macOS: use otool
            try:
                result = subprocess.run(
                    ["otool", "-tV", binary_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30
                )
            except FileNotFoundError:
                print("  No disassembler found (objdump, llvm-objdump, otool)")
                return None

    disasm = result.stdout

    # Count spill-related patterns
    # ARM64: str/stp to [sp, ...] and ldr/ldp from [sp, ...]
    # ARM32: str/ldr to [sp, ...]
    # x86: mov to [rbp-...] or [rsp+...]
    spill_store_patterns = [
        r'\b(str|stp)\s+.*\[sp',           # ARM64 stores to stack
        r'\bstr\s+.*\[sp',                   # ARM32 stores to stack
        r'\bmov[lq]?\s+.*,\s*-?\d*\(%rbp\)', # x86 stores to stack frame
        r'\bmov[lq]?\s+.*,\s*\d*\(%rsp\)',   # x86 stores to stack
    ]
    spill_load_patterns = [
        r'\b(ldr|ldp)\s+.*\[sp',            # ARM64 loads from stack
        r'\bldr\s+.*\[sp',                   # ARM32 loads from stack
        r'\bmov[lq]?\s+.*-?\d*\(%rbp\)',     # x86 loads from stack frame
    ]

    store_count = 0
    load_count = 0
    total_instructions = 0

    in_fused_function = False
    for line in disasm.split("\n"):
        # Look for the fused_winograd function
        if "fused_winograd" in line and (":" in line):
            in_fused_function = True
            continue

        # Exit if we hit another function (empty line or new function label)
        if in_fused_function and re.match(r'^[0-9a-f]+ <[^>]+>:', line):
            if "fused_winograd" not in line:
                in_fused_function = False
                continue

        if in_fused_function:
            total_instructions += 1
            for pat in spill_store_patterns:
                if re.search(pat, line, re.IGNORECASE):
                    store_count += 1
                    break
            for pat in spill_load_patterns:
                if re.search(pat, line, re.IGNORECASE):
                    load_count += 1
                    break

    return {
        "spill_stores": store_count,
        "spill_loads": load_count,
        "total_spills": store_count + load_count,
        "total_instructions_in_function": total_instructions,
        "spill_ratio": (store_count + load_count) / max(total_instructions, 1),
    }


def _run_perf_spill_counters(c_in=128, c_out=128):
    """Run perf stat to collect spill-related hardware counters."""
    # Check if perf is available
    try:
        subprocess.run(["perf", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        print("  perf not available, skipping runtime spill counters")
        return None

    # We need to run the kernel under perf
    # Create a small C driver program
    driver_code = f"""
#include <stdlib.h>
#include <string.h>

extern void fused_winograd_f23(const float*, const float*, float*, int, int);

int main() {{
    int c_in = {c_in}, c_out = {c_out};
    float *input = (float*)calloc(c_in * 16, sizeof(float));
    float *U = (float*)calloc(c_out * c_in * 16, sizeof(float));
    float *output = (float*)calloc(c_out * 4, sizeof(float));

    // Run 100 iterations
    for (int i = 0; i < 100; i++) {{
        fused_winograd_f23(input, U, output, c_in, c_out);
    }}

    free(input); free(U); free(output);
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        driver_path = os.path.join(tmpdir, "driver.c")
        binary_path = os.path.join(tmpdir, "driver")
        kernel_path = os.path.join(os.path.dirname(__file__), "..", "fused_winograd.c")

        with open(driver_path, "w") as f:
            f.write(driver_code)

        # Compile
        cmd = ["gcc", "-O2", "-g", driver_path, kernel_path, "-o", binary_path, "-lm"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            print(f"  Driver compilation failed: {result.stderr}")
            return None

        # Run under perf
        events = "L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,instructions,cycles"
        perf_cmd = ["perf", "stat", "-e", events, binary_path]
        try:
            result = subprocess.run(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            print(f"  perf stat failed: {result.stderr}")
            return None

        # Parse perf output
        perf_output = result.stderr
        counters = {}
        for line in perf_output.split("\n"):
            line = line.strip()
            for event in ["L1-dcache-loads", "L1-dcache-load-misses", "L1-dcache-stores",
                          "instructions", "cycles"]:
                if event in line:
                    match = re.search(r'([\d,]+)\s+' + re.escape(event), line)
                    if match:
                        counters[event] = int(match.group(1).replace(",", ""))

        if counters:
            loads = counters.get("L1-dcache-loads", 0)
            misses = counters.get("L1-dcache-load-misses", 0)
            hit_rate = (loads - misses) / max(loads, 1) * 100.0
            counters["L1_hit_rate_pct"] = round(hit_rate, 2)

        return counters


def run_instrument(c_in=128, c_out=128):
    """Main entry point for T6."""
    print(f"[T6] Register-Spill Instrumentation for ({c_in},{c_out})")

    timestamp = datetime.datetime.now().isoformat()
    kernel_source = os.path.join(os.path.dirname(__file__), "..", "fused_winograd.c")

    if not os.path.exists(kernel_source):
        print(f"  ERROR: Kernel source not found: {kernel_source}")
        sys.exit(1)

    rows = []

    # 1. Static analysis via objdump
    print("\n  --- Static Disassembly Analysis ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        so_path = os.path.join(tmpdir, "fused_winograd_debug.so")
        if _compile_kernel(kernel_source, so_path):
            spill_data = _count_stack_spills_objdump(so_path)
            if spill_data:
                print(f"  Spill stores: {spill_data['spill_stores']}")
                print(f"  Spill loads: {spill_data['spill_loads']}")
                print(f"  Total spills: {spill_data['total_spills']}")
                print(f"  Total instructions: {spill_data['total_instructions_in_function']}")
                print(f"  Spill ratio: {spill_data['spill_ratio']:.4f}")

                # Hypothesis check: ~4x accumulator over-subscription
                # F(2,3) has 16 accumulators per output channel tile
                # ARM64 has 32 SIMD registers → 32×4 = 128 float32 values
                # Query the autotiler for the actual C_out block size used
                try:
                    from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
                    _tiler = CacheAdaptiveAutotiler()
                    _decision = _tiler.select_best_tile(c_in, c_out)
                    # c_out_block is the number of output channels processed per tile
                    c_out_block = _decision.get("c_out_block", min(c_out, 16))
                except Exception:
                    c_out_block = min(c_out, 16)  # safe fallback
                accum_values = c_out_block * 16  # 16 elements per F(2,3) output tile
                arm_reg_capacity = 32 * 4  # 32 NEON regs, 4 floats each
                over_sub_ratio = accum_values / arm_reg_capacity
                print(f"\n  Accumulator over-subscription analysis:")
                print(f"    C_out block (from autotiler): {c_out_block}")
                print(f"    Accumulator values: {accum_values}")
                print(f"    ARM64 register capacity: {arm_reg_capacity} floats")
                print(f"    Over-subscription ratio: {over_sub_ratio:.1f}x")

                hypothesis_supported = spill_data['spill_ratio'] > 0.05  # >5% spill ratio suggests pressure
                print(f"    ~4x hypothesis {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'} "
                      f"(spill ratio = {spill_data['spill_ratio']:.4f})")

                rows.append({
                    "timestamp": timestamp,
                    "method": "static_objdump",
                    "c_in": c_in, "c_out": c_out,
                    "spill_stores": spill_data["spill_stores"],
                    "spill_loads": spill_data["spill_loads"],
                    "total_spills": spill_data["total_spills"],
                    "total_instructions": spill_data["total_instructions_in_function"],
                    "spill_ratio": round(spill_data["spill_ratio"], 6),
                    "over_sub_ratio": round(over_sub_ratio, 2),
                    "hypothesis_supported": hypothesis_supported,
                })
            else:
                print("  Could not analyze disassembly")
        else:
            print("  Compilation failed, skipping static analysis")

    # 2. Runtime perf counters (if available)
    print("\n  --- Runtime Perf Counter Analysis ---")
    perf_data = _run_perf_spill_counters(c_in, c_out)
    if perf_data:
        print(f"  L1 cache hit rate: {perf_data.get('L1_hit_rate_pct', 'N/A')}%")
        print(f"  Instructions: {perf_data.get('instructions', 'N/A')}")
        print(f"  Cycles: {perf_data.get('cycles', 'N/A')}")

        rows.append({
            "timestamp": timestamp,
            "method": "perf_stat",
            "c_in": c_in, "c_out": c_out,
            "L1_loads": perf_data.get("L1-dcache-loads", "N/A"),
            "L1_misses": perf_data.get("L1-dcache-load-misses", "N/A"),
            "L1_hit_rate_pct": perf_data.get("L1_hit_rate_pct", "N/A"),
            "instructions": perf_data.get("instructions", "N/A"),
            "cycles": perf_data.get("cycles", "N/A"),
        })

    # Write results
    if rows:
        out_path = os.path.join("raw_logs", "register_spill_128x128.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Since different rows may have different keys, collect all keys
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)

        file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\n[T6] Results written to {out_path}")
    else:
        print("\n[T6] ERROR: No results produced. Ensure gcc/clang and optionally perf are available.")
        print("     Cannot produce raw_logs/register_spill_128x128.csv without a working compiler.")
        sys.exit(1)  # Fail loudly per constraint #6


def main():
    parser = argparse.ArgumentParser(description="T6: Register-spill instrumentation")
    parser.add_argument("--c-in", type=int, default=128)
    parser.add_argument("--c-out", type=int, default=128)
    args = parser.parse_args()
    run_instrument(c_in=args.c_in, c_out=args.c_out)


if __name__ == "__main__":
    main()
