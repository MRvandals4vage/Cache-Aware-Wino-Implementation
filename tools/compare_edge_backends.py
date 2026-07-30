#!/usr/bin/env python3

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

try:
    from dataclasses import dataclass
except ImportError:
    def dataclass(cls):
        return cls

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fused_winograd_kernel import FusedWinogradKernel
from src.runtime_cache_probe import build_platform_descriptor


@dataclass
class BenchmarkResult:
    backend: str
    status: str
    avg_latency_ms: Optional[float]
    std_latency_ms: Optional[float]
    runs: int
    notes: str = ""

    def as_row(self, workload: Dict[str, int], platform_desc: Dict[str, object]) -> Dict[str, object]:
        row: Dict[str, object] = {
            "backend": self.backend,
            "status": self.status,
            "avg_latency_ms": self.avg_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "runs": self.runs,
            "notes": self.notes,
        }
        row.update(workload)
        row["platform"] = platform_desc.get("os", "unknown")
        row["architecture"] = platform_desc.get("architecture", "unknown")
        row["cpu_model"] = platform_desc.get("cpu_model", "unknown")
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the fused Winograd kernel against optional TVM, AutoTVM, and ARMCL backends."
    )
    parser.add_argument(
        "--backends",
        default="project,onnxruntime",
        help="Comma-separated backends: project,onnxruntime,tvm,autotvm,armcl",
    )
    parser.add_argument("--runs", type=int, default=30, help="Measured iterations per backend")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per backend")
    parser.add_argument("--c-in", type=int, default=64, help="Input channels")
    parser.add_argument("--c-out", type=int, default=64, help="Output channels")
    parser.add_argument("--height", type=int, default=4, help="Input height")
    parser.add_argument("--width", type=int, default=4, help="Input width")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "comparisons"),
        help="Directory for CSV/Markdown results",
    )
    parser.add_argument(
        "--tvm-target",
        default=None,
        help="Optional TVM target string. Defaults to an inferred target for the current machine.",
    )
    parser.add_argument(
        "--autotvm-trials",
        type=int,
        default=32,
        help="Maximum tuning trials per AutoTVM task",
    )
    parser.add_argument(
        "--armcl-command",
        default=os.environ.get("ARMCL_COMMAND"),
        help="Shell command that prints LATENCY_MS=<value>. Use placeholders like {c_in}, {c_out}, {runs}.",
    )
    return parser.parse_args()


def infer_tvm_target() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        if sys.platform == "darwin":
            return "llvm -mcpu=apple-m1"
        return "llvm -mattr=+neon"
    if machine.startswith("armv7"):
        return "llvm -mtriple=armv7l-linux-gnueabihf -mattr=+neon"
    return "llvm"


def benchmark_project(args: argparse.Namespace) -> BenchmarkResult:
    if args.kernel != 3:
        return BenchmarkResult("project", "skipped", None, None, 0, "The fused Winograd path only supports 3x3 kernels.")

    kernel = FusedWinogradKernel()
    input_tile = np.random.randn(args.c_in, 4, 4).astype(np.float32)
    weights = np.random.randn(args.c_out, args.c_in, 4, 4).astype(np.float32)

    for _ in range(args.warmup):
        kernel.run_fused(input_tile, weights)

    latencies: List[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        kernel.run_fused(input_tile, weights)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return BenchmarkResult(
        "project",
        "ok",
        statistics.mean(latencies),
        statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        len(latencies),
        "Fused Winograd F(2,3) tile microbenchmark.",
    )


def build_single_conv_onnx_model(path: Path, args: argparse.Namespace) -> None:
    output_h = args.height - args.kernel + 1
    output_w = args.width - args.kernel + 1
    if output_h <= 0 or output_w <= 0:
        raise ValueError("Input height/width must be at least as large as the kernel size.")
    input_tensor = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, args.c_in, args.height, args.width])
    output_tensor = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, args.c_out, output_h, output_w])
    weights = np.random.randn(args.c_out, args.c_in, args.kernel, args.kernel).astype(np.float32)
    weight_initializer = onh.from_array(weights, name="weight")
    conv_node = oh.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        kernel_shape=[args.kernel, args.kernel],
        strides=[1, 1],
    )
    graph = oh.make_graph([conv_node], "single_conv", [input_tensor], [output_tensor], [weight_initializer])
    model = oh.make_model(graph, producer_name="cache-aware-wino")
    onnx.checker.check_model(model)
    onnx.save(model, path)


def benchmark_onnxruntime(args: argparse.Namespace) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="edge-wino-onnx-") as tmpdir:
        model_path = Path(tmpdir) / "single_conv.onnx"
        build_single_conv_onnx_model(model_path, args)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_data = np.random.randn(1, args.c_in, args.height, args.width).astype(np.float32)

        for _ in range(args.warmup):
            session.run(None, {input_name: input_data})

        latencies: List[float] = []
        for _ in range(args.runs):
            start = time.perf_counter()
            session.run(None, {input_name: input_data})
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    return BenchmarkResult(
        "onnxruntime",
        "ok",
        statistics.mean(latencies),
        statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        len(latencies),
        "Single Conv ONNX Runtime CPU benchmark.",
    )


def _build_tvm_module(args: argparse.Namespace, autotvm_enabled: bool):
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor

    data_shape = (1, args.c_in, args.height, args.width)
    weight_shape = (args.c_out, args.c_in, args.kernel, args.kernel)
    data = relay.var("input", shape=data_shape, dtype="float32")
    weight_data = np.random.randn(*weight_shape).astype("float32")
    weight = relay.const(weight_data)
    conv = relay.nn.conv2d(
        data,
        weight,
        kernel_size=(args.kernel, args.kernel),
        padding=(1, 1),
        channels=args.c_out,
    )
    mod = tvm.IRModule.from_expr(relay.Function([data], conv))
    params = {}
    target = args.tvm_target or infer_tvm_target()

    if autotvm_enabled:
        from tvm import autotvm

        tasks = autotvm.task.extract_from_program(mod["main"], target=tvm.target.Target(target), params=params)
        if not tasks:
            raise RuntimeError("AutoTVM could not extract any tuning tasks.")

        with tempfile.NamedTemporaryFile(prefix="edge-wino-autotvm-", suffix=".log", delete=False) as log_file:
            tuning_log = log_file.name

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=1, repeat=1, min_repeat_ms=0),
        )
        try:
            for task in reversed(tasks):
                tuner = autotvm.tuner.XGBTuner(task)
                tuner.tune(
                    n_trial=min(args.autotvm_trials, len(task.config_space)),
                    measure_option=measure_option,
                    callbacks=[autotvm.callback.log_to_file(tuning_log)],
                )

            with autotvm.apply_history_best(tuning_log):
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(mod, target=target, params=params)
        finally:
            if os.path.exists(tuning_log):
                os.remove(tuning_log)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return tvm, module


def benchmark_tvm(args: argparse.Namespace, autotvm_enabled: bool) -> BenchmarkResult:
    backend_name = "autotvm" if autotvm_enabled else "tvm"
    try:
        tvm, module = _build_tvm_module(args, autotvm_enabled)
    except ImportError as exc:
        return BenchmarkResult(backend_name, "skipped", None, None, 0, f"TVM is not installed: {exc}")
    except Exception as exc:
        return BenchmarkResult(backend_name, "error", None, None, 0, str(exc))

    input_data = np.random.randn(1, args.c_in, args.height, args.width).astype("float32")
    module.set_input("input", tvm.nd.array(input_data))

    for _ in range(args.warmup):
        module.run()

    latencies: List[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        module.run()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    note = "Single Conv TVM benchmark."
    if autotvm_enabled:
        note = f"Single Conv AutoTVM benchmark with up to {args.autotvm_trials} trials per task."
    return BenchmarkResult(
        backend_name,
        "ok",
        statistics.mean(latencies),
        statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        len(latencies),
        note,
    )


def benchmark_armcl(args: argparse.Namespace) -> BenchmarkResult:
    if not args.armcl_command:
        return BenchmarkResult(
            "armcl",
            "skipped",
            None,
            None,
            0,
            "Set ARMCL_COMMAND or pass --armcl-command. The command must print LATENCY_MS=<value>.",
        )

    command = args.armcl_command.format(
        c_in=args.c_in,
        c_out=args.c_out,
        height=args.height,
        width=args.width,
        kernel=args.kernel,
        runs=args.runs,
        warmup=args.warmup,
    )
    completed = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    combined_output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    if completed.returncode != 0:
        return BenchmarkResult("armcl", "error", None, None, 0, combined_output or f"Command failed: {command}")

    match = re.search(r"LATENCY_MS\s*=\s*([0-9]+(?:\.[0-9]+)?)", combined_output)
    if not match:
        return BenchmarkResult(
            "armcl",
            "error",
            None,
            None,
            0,
            "ARMCL command did not emit LATENCY_MS=<value>.",
        )

    return BenchmarkResult(
        "armcl",
        "ok",
        float(match.group(1)),
        0.0,
        args.runs,
        "External ARM Compute Library wrapper command.",
    )


def write_outputs(
    results: List[BenchmarkResult],
    args: argparse.Namespace,
    workload: Dict[str, int],
    platform_desc: Dict[str, object],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"backend_comparison_{timestamp}.csv"
    md_path = output_dir / f"backend_comparison_{timestamp}.md"
    json_path = output_dir / f"backend_comparison_{timestamp}.json"

    rows = [result.as_row(workload, platform_desc) for result in results]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    successful = [result for result in results if result.status == "ok" and result.avg_latency_ms is not None]
    best_latency = min((result.avg_latency_ms for result in successful), default=None)
    markdown_lines = [
        "# Backend Comparison",
        "",
        f"- Platform: {platform_desc.get('os', 'unknown')} / {platform_desc.get('architecture', 'unknown')}",
        f"- CPU: {platform_desc.get('cpu_model', 'unknown')}",
        f"- Workload: C_in={args.c_in}, C_out={args.c_out}, H={args.height}, W={args.width}, K={args.kernel}",
        f"- Runs: {args.runs}, Warmup: {args.warmup}",
        "",
        "| Backend | Status | Avg Latency (ms) | Std (ms) | Relative to Best | Notes |",
        "| :------ | :----- | ---------------: | -------: | ---------------: | :---- |",
    ]
    for result in results:
        relative = "N/A"
        if best_latency is not None and result.avg_latency_ms not in (None, 0) and result.status == "ok":
            relative = f"{result.avg_latency_ms / best_latency:.2f}x"
        markdown_lines.append(
            f"| {result.backend} | {result.status} | "
            f"{result.avg_latency_ms if result.avg_latency_ms is not None else 'N/A'} | "
            f"{result.std_latency_ms if result.std_latency_ms is not None else 'N/A'} | "
            f"{relative} | {result.notes} |"
        )

    md_path.write_text("\n".join(markdown_lines) + "\n")
    json_path.write_text(json.dumps({"platform": platform_desc, "workload": workload, "results": rows}, indent=2) + "\n")

    print(f"Saved CSV comparison to {csv_path}")
    print(f"Saved Markdown summary to {md_path}")
    print(f"Saved JSON summary to {json_path}")


def main() -> None:
    args = parse_args()
    workload = {
        "c_in": args.c_in,
        "c_out": args.c_out,
        "height": args.height,
        "width": args.width,
        "kernel": args.kernel,
    }
    platform_desc = build_platform_descriptor()
    backend_names = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    backend_map: Dict[str, Callable[[argparse.Namespace], BenchmarkResult]] = {
        "project": benchmark_project,
        "onnxruntime": benchmark_onnxruntime,
        "tvm": lambda parsed_args: benchmark_tvm(parsed_args, autotvm_enabled=False),
        "autotvm": lambda parsed_args: benchmark_tvm(parsed_args, autotvm_enabled=True),
        "armcl": benchmark_armcl,
    }

    results: List[BenchmarkResult] = []
    for backend in backend_names:
        if backend not in backend_map:
            results.append(BenchmarkResult(backend, "skipped", None, None, 0, "Unknown backend."))
            continue
        print(f"Running backend: {backend}")
        results.append(backend_map[backend](args))

    if not results:
        raise SystemExit("No backends selected.")

    write_outputs(results, args, workload, platform_desc)


if __name__ == "__main__":
    main()
