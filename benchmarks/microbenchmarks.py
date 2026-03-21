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

def run_microbenchmark():
    np.random.seed(42)
    
    configs = {
        "C_in": [32, 128],
        "C_out": [32, 128],
        "fused": [False, True],
        "multi_core": [False, True]
    }
    
    runs = 1000
    warmup = 20
    
    results = []
    
    # Try perf stat
    hw_perf = HardwareCounter()
    
    # Autotiler for choosing
    tiler = CacheAdaptiveAutotiler()
    
    keys, values = zip(*configs.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    baseline_latencies = None
    
    for cfg in combinations:
        c_in, c_out = cfg["C_in"], cfg["C_out"]
        fused, multi = cfg["fused"], cfg["multi_core"]
        
        # Pick best tile based on autotiler
        tile_cfg = tiler.select_best_tile(c_in, c_out)
        tile_size = tile_cfg["tile"]
        
        # We generate a workload of say 100 tiles to simulate a layer
        num_tiles = 100
        tasks = [generate_data(c_in, c_out, 4, 4) for _ in range(num_tiles)] # Hardcode F23 for evaluation
        
        scheduler = MulticoreScheduler(mode="multi" if multi else "single", num_threads=4 if multi else 1)
        worker = _worker_fused if fused else _worker_baseline
        
        # Warmup
        for _ in range(warmup):
            scheduler.execute_tasks(tasks[:10], worker)
            
        latencies = []
        
        # Full runs
        for _ in range(runs):
            t0 = time.perf_counter()
            scheduler.execute_tasks(tasks, worker)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        ci = 1.96 * (std_lat / np.sqrt(runs))
        
        if not fused and not multi:
            baseline_latencies = latencies
            p_value = 1.0 # vs itself
        else:
            if baseline_latencies:
                _, p_value = stats.ttest_ind(latencies, baseline_latencies, equal_var=False)
            else:
                p_value = 1.0
                
        results.append({
            "C_in": c_in,
            "C_out": c_out,
            "Tile": tile_cfg["name"],
            "Fused": fused,
            "MultiCore": multi,
            "Mean_Latency_ms": mean_lat,
            "StdDev_ms": std_lat,
            "Conf_Interval_95": ci,
            "P_Value_vs_Baseline": p_value
        })
        print(f"[{c_in}x{c_out} Fused={fused} Multi={multi}] Latency: {mean_lat:.2f}ms ± {ci:.2f}ms (p_val={p_value:.3f})")
        
    df = pd.DataFrame(results)
    df.to_csv("microbenchmark_results.csv", index=False)
    print("Saved microbenchmark_results.csv")

if __name__ == "__main__":
    run_microbenchmark()
