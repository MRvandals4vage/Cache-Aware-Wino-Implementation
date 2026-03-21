import time
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fused_winograd_kernel import FusedWinogradKernel
from cache_adaptive_autotiler import CacheAdaptiveAutotiler
from multicore_scheduler import MulticoreScheduler
from runtime_cache_probe import RuntimeCacheProbe

class HardwareCounter:
    def __init__(self):
        self.cmd = ["perf", "stat", "-e", "L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_miss", "-p", str(os.getpid())]
        self.process = None

    def start(self):
        try:
            self.process = subprocess.Popen(self.cmd, stderr=subprocess.PIPE, universal_newlines=True)
        except Exception:
            self.process = None

    def stop(self):
        if self.process:
            self.process.terminate()
            stdout, stderr = self.process.communicate()
            return stderr
        return ""

def generate_data(C_in, C_out, tile_h, tile_w):
    # F(2,3) -> tile size 4
    input_tile = np.random.randn(C_in, tile_h, tile_w).astype(np.float32)
    U = np.random.randn(C_out, C_in, tile_h, tile_w).astype(np.float32)
    return input_tile, U

def execute_baseline(input_tile, U):
    # Baseline direct convolution simulation
    # Actually just simple matmuls to simulate non-fused winograd steps
    BT = np.array([[ 1,  0, -1,  0], [ 0,  1,  1,  0], [ 0, -1,  1,  0], [ 0,  1,  0, -1]], dtype=np.float32)
    AT = np.array([[1,  1,  1,  0], [0,  1, -1, -1]], dtype=np.float32)
    
    # Non fused simulates writing full V and reading it back
    V = np.matmul(np.matmul(BT, input_tile), BT.T)
    V_store = np.copy(V)
    
    M = U * V_store[np.newaxis, ...]
    M_sum = np.sum(M, axis=1) # intermediate storage
    M_sum_store = np.copy(M_sum)
    
    Y = np.matmul(np.matmul(AT, M_sum_store), AT.T)
    return Y

def _worker_baseline(args):
    inp, U = args
    return execute_baseline(inp, U)

def _worker_fused(args):
    inp, U = args
    kernel = FusedWinogradKernel(use_c_ext=True)
    return kernel.execute(inp, U)

from hardware_telemetry import HardwareTelemetry

def run_microbenchmark():
    np.random.seed(42)
    
    # We will ensure artifacts directories exist
    os.makedirs("artifacts/raw", exist_ok=True)
    os.makedirs("artifacts/processed", exist_ok=True)

    configs = {
        "C_in": [32, 128],
        "C_out": [32, 128],
        "fused": [False, True],
        "multi_core": [False, True]
    }
    
    runs = 1000
    warmup = 20
    
    results = []
    
    # Probe platform
    probe = RuntimeCacheProbe()
    probe.save_descriptor("artifacts/platform_descriptor.json")
    print(f"Platform: {probe.get_info().get('platform', 'unknown')} | CPU: {probe.get_info().get('cpu_model', 'unknown')}")
    
    hw_perf = HardwareTelemetry(use_perf=True)
    tiler = CacheAdaptiveAutotiler()
    
    keys, values = zip(*configs.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    baseline_latencies = None
    
    # Raw latencies data
    all_raw_latencies = []
    
    for cfg in combinations:
        c_in, c_out = cfg["C_in"], cfg["C_out"]
        fused, multi = cfg["fused"], cfg["multi_core"]
        
        tile_cfg = tiler.select_best_tile(c_in, c_out)
        
        num_tiles = 100
        tasks = [generate_data(c_in, c_out, 4, 4) for _ in range(num_tiles)]
        
        scheduler = MulticoreScheduler(mode="multi" if multi else "single", num_threads=4 if multi else 1)
        worker = _worker_fused if fused else _worker_baseline
        
        for _ in range(warmup):
            scheduler.execute_tasks(tasks[:10], worker)
            
        latencies = []
        
        hw_perf.start()
        for i in range(runs):
            t0 = time.perf_counter()
            scheduler.execute_tasks(tasks, worker)
            t1 = time.perf_counter()
            duration_ms = (t1 - t0) * 1000.0
            latencies.append(duration_ms)
            
            # Save raw metric
            all_raw_latencies.append({
                "C_in": c_in, "C_out": c_out, "Fused": fused, "Multi": multi, 
                "Run_ID": i, "Latency_ms": duration_ms
            })
            
        perf_output = hw_perf.stop()
        
        mean_lat = np.mean(latencies)
        median_lat = np.median(latencies)
        std_lat = np.std(latencies, ddof=1)
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)
        ci = 1.96 * (std_lat / np.sqrt(runs))
        
        if not fused and not multi:
            baseline_latencies = latencies
            test_used = "N/A"
            p_value = 1.0
            effect_direction = "N/A"
        else:
            if baseline_latencies and np.var(latencies) > 0 and np.var(baseline_latencies) > 0:
                test_used = "Welch t-test"
                stat_val, p_value = stats.ttest_ind(latencies, baseline_latencies, equal_var=False)
                effect_direction = "Faster" if mean_lat < np.mean(baseline_latencies) else "Slower"
            else:
                test_used = "N/A"
                p_value = 1.0
                effect_direction = "N/A"
                
        results.append({
            "Platform": probe.get_info().get("platform", "Unknown"),
            "C_in": c_in,
            "C_out": c_out,
            "Tile": tile_cfg["name"],
            "Fused": fused,
            "MultiCore": multi,
            "Mean_Latency_ms": mean_lat,
            "Median_Latency_ms": median_lat,
            "StdDev_ms": std_lat,
            "Conf_Interval_95": ci,
            "Min_ms": min_lat,
            "Max_ms": max_lat,
            "Test_Used": test_used,
            "P_Value_vs_Baseline": p_value,
            "Effect_Direction": effect_direction,
            "HW_Perf_Output": "Logged" if perf_output.get("perf_stat") else "Unsupported",
            "Pi_Temp": perf_output.get("pi_temp", "N/A"),
            "Pi_Clock": perf_output.get("pi_clock_arm", "N/A"),
            "Pi_Throttled": perf_output.get("pi_throttled", "N/A"),
            "CPU_Percent_End": perf_output.get("cpu_percent", "N/A")
        })
        print(f"[{c_in}x{c_out} Fused={fused} Multi={multi}] Mean: {mean_lat:.2f}ms CI: ±{ci:.2f}ms | p_val: {p_value:.3e}")
        
    df_raw = pd.DataFrame(all_raw_latencies)
    df_raw.to_csv("artifacts/raw/microbenchmark_raw_latencies.csv", index=False)
    
    df_proc = pd.DataFrame(results)
    df_proc.to_csv("artifacts/processed/microbenchmark_results.csv", index=False)
    print("Saved datasets to artifacts/raw/ and artifacts/processed/")

if __name__ == "__main__":
    run_microbenchmark()
