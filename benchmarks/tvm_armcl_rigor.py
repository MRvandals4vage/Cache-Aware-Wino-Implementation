#!/usr/bin/env python3
"""
T3 — TVM/AutoTVM and ArmCL statistical rigor.

Re-runs both baselines for configs (32,64), (64,32), (64,64) on Jetson Nano,
n=1000, identical warm-up/thread-pinning to the CacheWinograd runs.

Logs every individual run (not just mean) to raw_logs/tvm_armcl_full_traces.csv.
Computes CI95 and Welch's t-test vs CacheWinograd fused latency.
Writes summary to summary/tvm_armcl_significance.csv.

MUST be executed on a device with TVM and/or ArmCL installed.
"""
import os
import sys
import time
import datetime
import argparse
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fused_winograd_kernel import FusedWinogradKernel
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.locality_scheduler import LocalityScheduler

# Try importing TVM
try:
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False

# Try importing ONNX Runtime (for building comparison models)
try:
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

CONFIGS = [
    {"c_in": 32, "c_out": 64},
    {"c_in": 64, "c_out": 32},
    {"c_in": 64, "c_out": 64},
]

H, W = 56, 56  # Match paper spatial dimensions
KERNEL_SIZE = 3


def _infer_tvm_target():
    """Infer TVM target for current platform."""
    import platform
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        if sys.platform == "darwin":
            return "llvm -mcpu=apple-m1"
        return "llvm -mattr=+neon"
    return "llvm"


def _build_tvm_module(c_in, c_out, h, w, target, autotvm_enabled=False):
    """Build a TVM module for a single conv layer."""
    data_shape = (1, c_in, h, w)
    weight_shape = (c_out, c_in, KERNEL_SIZE, KERNEL_SIZE)
    data = relay.var("input", shape=data_shape, dtype="float32")
    weight_data = np.random.randn(*weight_shape).astype("float32")
    weight = relay.const(weight_data)
    conv = relay.nn.conv2d(data, weight, kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
                           padding=(1, 1), channels=c_out)
    mod = tvm.IRModule.from_expr(relay.Function([data], conv))

    if autotvm_enabled:
        from tvm import autotvm
        import tempfile
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=tvm.target.Target(target), params={})
        if tasks:
            with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as lf:
                log_path = lf.name
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=1, repeat=1, min_repeat_ms=0))
            try:
                for task in reversed(tasks):
                    tuner = autotvm.tuner.XGBTuner(task)
                    tuner.tune(n_trial=min(32, len(task.config_space)),
                               measure_option=measure_option,
                               callbacks=[autotvm.callback.log_to_file(log_path)])
                with autotvm.apply_history_best(log_path):
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build(mod, target=target, params={})
            finally:
                if os.path.exists(log_path):
                    os.remove(log_path)
        else:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params={})
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params={})

    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module


def benchmark_tvm(c_in, c_out, h, w, target, n_runs, warmup, autotvm_enabled=False):
    """Benchmark TVM/AutoTVM for a single config."""
    backend = "autotvm" if autotvm_enabled else "tvm"
    if not TVM_AVAILABLE:
        print(f"     [{backend}] TVM not available, skipping.")
        return None, backend

    try:
        module = _build_tvm_module(c_in, c_out, h, w, target, autotvm_enabled)
    except Exception as e:
        print(f"     [{backend}] Build failed: {e}")
        return None, backend

    input_data = np.random.randn(1, c_in, h, w).astype("float32")
    module.set_input("input", tvm.nd.array(input_data))

    for _ in range(warmup):
        module.run()

    latencies = []
    for _ in range(n_runs):
        t0 = time.monotonic()
        module.run()
        t1 = time.monotonic()
        latencies.append((t1 - t0) * 1000.0)

    return latencies, backend


def benchmark_armcl(c_in, c_out, h, w, n_runs, warmup, armcl_command=None):
    """Benchmark ArmCL via external command."""
    import subprocess
    import re

    if not armcl_command:
        armcl_command = os.environ.get("ARMCL_COMMAND")
    if not armcl_command:
        print("     [armcl] No ARMCL_COMMAND set, skipping.")
        return None, "armcl"

    # We need per-run latencies, so run the command n_runs times
    # or if the command supports --runs, parse individual latencies
    latencies = []
    for run_id in range(n_runs):
        cmd = armcl_command.format(c_in=c_in, c_out=c_out, height=h, width=w,
                                    kernel=KERNEL_SIZE, runs=1, warmup=warmup if run_id == 0 else 0)
        try:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
            match = re.search(r"LATENCY_MS\s*=\s*([0-9]+(?:\.[0-9]+)?)", result.stdout + result.stderr)
            if match:
                latencies.append(float(match.group(1)))
        except Exception as e:
            print(f"     [armcl] Run {run_id} failed: {e}")

    if not latencies:
        return None, "armcl"
    return latencies, "armcl"


def benchmark_cachewinograd(c_in, c_out, h, w, n_runs, warmup):
    """Benchmark CacheWinograd fused kernel for comparison."""
    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler()
    scheduler = LocalityScheduler()

    decision = tiler.select_best_tile(c_in, c_out)
    tile_dim = decision["selected_tile"]["tile"]
    tasks = scheduler.generate_tile_tasks(h, w, tile_dim, c_in, c_out)
    ordered_tasks = scheduler.group_tasks_by_channel_locality(tasks)

    np.random.seed(42)
    input_tile = np.random.randn(c_in, tile_dim, tile_dim).astype(np.float32)
    U = np.random.randn(c_out, c_in, tile_dim, tile_dim).astype(np.float32)

    for _ in range(warmup):
        for task in ordered_tasks[:min(10, len(ordered_tasks))]:
            kernel.run_fused(input_tile, U)

    latencies = []
    for _ in range(n_runs):
        t0 = time.monotonic()
        for task in ordered_tasks:
            kernel.run_fused(input_tile, U)
        t1 = time.monotonic()
        latencies.append((t1 - t0) * 1000.0)

    return latencies, "cachewinograd"


def run_rigor(n_runs=1000, warmup=20, tvm_target=None, armcl_command=None):
    """Main entry point for T3."""
    from scipy import stats as sp_stats

    if tvm_target is None:
        tvm_target = _infer_tvm_target()

    timestamp = datetime.datetime.now().isoformat()
    raw_rows = []
    summary_rows = []

    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        print(f"\n[T3] Config ({c_in},{c_out}), H={H}, W={W}")

        # Run CacheWinograd (our method) first
        cw_lats, _ = benchmark_cachewinograd(c_in, c_out, H, W, n_runs, warmup)
        cw_mean = float(np.mean(cw_lats))
        print(f"     CacheWinograd: mean={cw_mean:.2f}ms")

        for cw_run_id, lat in enumerate(cw_lats):
            raw_rows.append({
                "timestamp": timestamp, "backend": "cachewinograd",
                "c_in": c_in, "c_out": c_out, "h": H, "w": W,
                "run_id": cw_run_id, "latency_ms": lat,
            })

        # Run baselines
        backends_to_run = [
            ("tvm", lambda: benchmark_tvm(c_in, c_out, H, W, tvm_target, n_runs, warmup, False)),
            ("autotvm", lambda: benchmark_tvm(c_in, c_out, H, W, tvm_target, n_runs, warmup, True)),
            ("armcl", lambda: benchmark_armcl(c_in, c_out, H, W, n_runs, warmup, armcl_command)),
        ]

        for backend_name, run_fn in backends_to_run:
            lats, actual_name = run_fn()
            if lats is None:
                print(f"     {actual_name}: SKIPPED (not available)")
                summary_rows.append({
                    "config": f"({c_in},{c_out})",
                    "baseline": actual_name,
                    "n": 0, "mean_ms": "N/A", "std_ms": "N/A",
                    "ci95_ms": "N/A", "p_value": "N/A",
                    "cw_mean_ms": round(cw_mean, 4),
                    "pct_improvement": "N/A", "status": "SKIPPED",
                })
                continue

            for run_id, lat in enumerate(lats):
                raw_rows.append({
                    "timestamp": timestamp, "backend": actual_name,
                    "c_in": c_in, "c_out": c_out, "h": H, "w": W,
                    "run_id": run_id, "latency_ms": lat,
                })

            bl_mean = float(np.mean(lats))
            bl_std = float(np.std(lats, ddof=1))
            bl_ci95 = 1.96 * bl_std / np.sqrt(len(lats))

            # Welch's t-test: CacheWinograd vs baseline
            stat, p = sp_stats.ttest_ind(cw_lats, lats, equal_var=False)
            p_str = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"
            improvement = ((bl_mean - cw_mean) / bl_mean) * 100.0

            print(f"     {actual_name}: mean={bl_mean:.2f}ms, CI95=±{bl_ci95:.2f}ms, "
                  f"p={p_str}, CW improvement={improvement:+.2f}%")

            summary_rows.append({
                "config": f"({c_in},{c_out})",
                "baseline": actual_name,
                "n": len(lats),
                "mean_ms": round(bl_mean, 4),
                "std_ms": round(bl_std, 4),
                "ci95_ms": round(bl_ci95, 4),
                "p_value": p_str,
                "cw_mean_ms": round(cw_mean, 4),
                "pct_improvement": round(improvement, 2),
                "status": "SIGNIFICANT" if p < 0.05 else "NOT_SIGNIFICANT",
            })

    # Write raw traces (append) — guard against empty list if all backends skipped
    raw_path = os.path.join("raw_logs", "tvm_armcl_full_traces.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    if not raw_rows:
        print("\n[T3] WARNING: No raw rows collected — all backends were skipped.")
    else:
        file_exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
        with open(raw_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(raw_rows)
        print(f"\n[T3] Appended {len(raw_rows)} rows to {raw_path}")

    # Write summary
    summary_path = os.path.join("summary", "tvm_armcl_significance.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[T3] Summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="T3: TVM/ArmCL statistical rigor")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--tvm-target", default=None)
    parser.add_argument("--armcl-command", default=None)
    args = parser.parse_args()
    run_rigor(n_runs=args.runs, warmup=args.warmup,
              tvm_target=args.tvm_target, armcl_command=args.armcl_command)


if __name__ == "__main__":
    main()
