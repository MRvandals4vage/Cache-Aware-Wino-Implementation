#!/usr/bin/env python3
"""
T7 — Energy per-run distribution.

Modifies the tegrastats-based energy harness so it logs power samples
per individual run rather than a single aggregate.

Recomputes CI95 for the VGG16 end-to-end energy number.
Logs to raw_logs/energy_per_run_vgg16.csv.

MUST be executed on a Jetson Nano (uses INA3221 power sensors).
"""
import os
import sys
import time
import datetime
import threading
import argparse
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class PerRunPowerSampler:
    """Samples power per individual inference run on Jetson Nano.

    Uses INA3221 sensors via sysfs, or tegrastats as fallback,
    or psutil-based proxy on non-Jetson systems.
    """
    INA3221_BASE = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio_device"
    RAILS = {
        "total": "in_power0_input",
        "cpu": "in_power1_input",
        "gpu": "in_power2_input",
    }

    def __init__(self, sample_interval_ms=5):
        self.sample_interval = sample_interval_ms / 1000.0
        self._is_jetson = os.path.exists(self.INA3221_BASE)
        self._sampling = False
        self._samples = []
        self._thread = None

    def _read_power_mw(self):
        """Read total board power in milliwatts."""
        if self._is_jetson:
            path = os.path.join(self.INA3221_BASE, self.RAILS["total"])
            try:
                with open(path, "r") as f:
                    return float(f.read().strip())
            except Exception:
                return 0.0
        else:
            # Proxy: use psutil CPU usage
            import psutil
            cpu_pct = psutil.cpu_percent(interval=None)
            return 2500.0 + cpu_pct * 30.0  # rough proxy

    def _sample_loop(self):
        while self._sampling:
            self._samples.append({
                "time_ns": time.perf_counter_ns(),
                "power_mw": self._read_power_mw(),
            })
            time.sleep(self.sample_interval)

    def start(self):
        self._samples = []
        self._sampling = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._sampling = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return list(self._samples)


def run_energy_benchmark(n_runs=50, warmup=10, model_name="vgg16"):
    """Run energy-per-run benchmark for VGG16."""
    from scipy import stats as sp_stats

    print(f"[T7] Energy per-run distribution — {model_name}")

    # Import E2E benchmark components
    try:
        from benchmarks.e2e_benchmark import (
            benchmark_cachewinograd_e2e,
            benchmark_onnxruntime_e2e,
            _get_conv3x3_layers,
        )
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from e2e_benchmark import (
            benchmark_cachewinograd_e2e,
            benchmark_onnxruntime_e2e,
            _get_conv3x3_layers,
        )

    conv_layers = _get_conv3x3_layers(model_name)
    if not conv_layers:
        print(f"  ERROR: No conv layers found for {model_name}")
        sys.exit(1)

    sampler = PerRunPowerSampler(sample_interval_ms=5)
    timestamp = datetime.datetime.now().isoformat()
    raw_rows = []

    from src.fused_winograd_kernel import FusedWinogradKernel
    from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
    from src.locality_scheduler import LocalityScheduler

    kernel = FusedWinogradKernel()
    tiler = CacheAdaptiveAutotiler()

    # Prepare layer data
    layer_data = []
    for layer in conv_layers:
        decision = tiler.select_best_tile(layer["c_in"], layer["c_out"])
        td = decision["selected_tile"]["tile"]
        scheduler = LocalityScheduler()
        tasks = scheduler.generate_tile_tasks(layer["h_in"], layer["w_in"], td,
                                               layer["c_in"], layer["c_out"])
        ordered_tasks = scheduler.group_tasks_by_channel_locality(tasks)
        input_tile = np.random.randn(layer["c_in"], td, td).astype(np.float32)
        U = np.random.randn(layer["c_out"], layer["c_in"], td, td).astype(np.float32)
        layer_data.append({
            "tasks": ordered_tasks, "input_tile": input_tile, "U": U,
            "stride": layer.get("stride", 1),
        })

    # Warmup — full coverage across all layers to warm the entire working set
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        for ld in layer_data:
            for task in ld["tasks"]:
                kernel.run_fused(ld["input_tile"], ld["U"])

    # Measurement with per-run power sampling
    print(f"  Running {n_runs} measured iterations with power sampling...")
    for run_id in range(n_runs):
        sampler.start()
        t0 = time.perf_counter()

        total_ms = 0.0
        for ld in layer_data:
            lt0 = time.perf_counter()
            if ld["stride"] > 1:
                for task in ld["tasks"]:
                    kernel.run_non_fused(ld["input_tile"], ld["U"])
            else:
                for task in ld["tasks"]:
                    kernel.run_fused(ld["input_tile"], ld["U"])
            lt1 = time.perf_counter()
            total_ms += (lt1 - lt0) * 1000.0

        t1 = time.perf_counter()
        samples = sampler.stop()

        duration_s = t1 - t0
        duration_ms = duration_s * 1000.0

        # Compute energy from power samples
        if samples and len(samples) > 1:
            power_values = [s["power_mw"] for s in samples]
            avg_power_mw = float(np.mean(power_values))
            energy_mj = avg_power_mw * duration_s  # mW * s = mJ
        else:
            avg_power_mw = 0.0
            energy_mj = 0.0

        raw_rows.append({
            "timestamp": timestamp,
            "model": model_name,
            "run_id": run_id,
            "latency_ms": round(duration_ms, 4),
            "layer_sum_ms": round(total_ms, 4),
            "avg_power_mw": round(avg_power_mw, 4),
            "energy_mj": round(energy_mj, 4),
            "n_power_samples": len(samples),
        })

        if (run_id + 1) % 10 == 0:
            print(f"    Run {run_id + 1}/{n_runs}: {duration_ms:.1f}ms, {avg_power_mw:.0f}mW, {energy_mj:.2f}mJ")

    # Write raw log (append)
    out_path = os.path.join("raw_logs", "energy_per_run_vgg16.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\n[T7] Appended {len(raw_rows)} rows to {out_path}")

    # Write summary
    energies = [r["energy_mj"] for r in raw_rows]
    latencies = [r["latency_ms"] for r in raw_rows]

    mean_energy = float(np.mean(energies))
    std_energy = float(np.std(energies, ddof=1))
    ci95_energy = 1.96 * std_energy / np.sqrt(len(energies))
    mean_lat = float(np.mean(latencies))

    summary_path = os.path.join("summary", "energy_vgg16_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "n", "mean_energy_mj", "std_energy_mj", "ci95_energy_mj",
            "mean_latency_ms", "mean_power_mw", "is_jetson",
        ])
        writer.writeheader()
        writer.writerow({
            "model": model_name,
            "n": len(energies),
            "mean_energy_mj": round(mean_energy, 4),
            "std_energy_mj": round(std_energy, 4),
            "ci95_energy_mj": round(ci95_energy, 4),
            "mean_latency_ms": round(mean_lat, 4),
            "mean_power_mw": round(float(np.mean([r["avg_power_mw"] for r in raw_rows])), 4),
            "is_jetson": os.path.exists(PerRunPowerSampler.INA3221_BASE),
        })
    print(f"[T7] Summary written to {summary_path}")
    print(f"     Energy: {mean_energy:.2f} ± {ci95_energy:.2f} mJ (CI95)")


def main():
    parser = argparse.ArgumentParser(description="T7: Energy per-run distribution")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model", default="vgg16")
    args = parser.parse_args()
    run_energy_benchmark(n_runs=args.runs, warmup=args.warmup, model_name=args.model)


if __name__ == "__main__":
    main()
