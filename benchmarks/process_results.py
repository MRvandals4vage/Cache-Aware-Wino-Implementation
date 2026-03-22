#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

def process_results(raw_dir="artifacts/raw", processed_dir="artifacts/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"No raw CSV files found in {raw_dir}")
        return
        
    for rf in csv_files:
        try:
            df = pd.read_csv(rf)
            
            if df.empty or "latency_ms" not in df.columns:
                print(f"Skipping {rf} - no latency_ms data.")
                continue
                
            groups = df.groupby(["c_in", "c_out", "h", "w", "tile_dim", "fused", "threads"])
            processed_data = []
            
            # For Welch's t-test, we need a baseline. Let's assume baseline is fused=False, threads=1
            baseline_dict = {}
            for name, group in groups:
                if not group["fused"].iloc[0] and group["threads"].iloc[0] == 1:
                    baseline_key = (group["c_in"].iloc[0], group["c_out"].iloc[0])
                    baseline_dict[baseline_key] = group["latency_ms"].values
                    
            for name, group in groups:
                c_in, c_out, h, w, tile_dim, fused, threads = name
                latencies = group["latency_ms"].values
                n_runs = len(latencies)
                
                mean_lat = np.mean(latencies)
                median_lat = np.median(latencies)
                std_lat = np.std(latencies, ddof=1) if n_runs > 1 else 0.0
                min_lat = np.min(latencies)
                max_lat = np.max(latencies)
                
                # 95% Confidence interval
                ci_95 = 1.96 * (std_lat / np.sqrt(n_runs)) if n_runs > 0 else 0.0
                
                # Welch t-test against baseline
                baseline_key = (c_in, c_out)
                p_value = 1.0
                test_used = "N/A"
                if fused or threads > 1:
                    if baseline_key in baseline_dict:
                        base_lats = baseline_dict[baseline_key]
                        if len(base_lats) > 1 and len(latencies) > 1:
                            stat_val, p = stats.ttest_ind(latencies, base_lats, equal_var=False)
                            p_value = float(p)
                            test_used = "Welch t-test"
                            
                processed_data.append({
                    "c_in": c_in,
                    "c_out": c_out,
                    "h": h,
                    "w": w,
                    "tile_dim": tile_dim,
                    "fused": fused,
                    "threads": threads,
                    "runs": n_runs,
                    "mean_latency_ms": mean_lat,
                    "median_latency_ms": median_lat,
                    "std_dev_ms": std_lat,
                    "min_latency_ms": min_lat,
                    "max_latency_ms": max_lat,
                    "ci_95_ms": ci_95,
                    "p_value": p_value,
                    "test_used": test_used
                })
                
            out_df = pd.DataFrame(processed_data)
            out_file = os.path.join(processed_dir, "processed_" + os.path.basename(rf))
            out_df.to_csv(out_file, index=False)
            print(f"Processed {rf} -> {out_file}")
            
        except Exception as e:
            print(f"Error processing {rf}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process raw benchmark CSVs")
    parser.add_argument("--raw-dir", default="artifacts/raw")
    parser.add_argument("--out-dir", default="artifacts/processed")
    args = parser.parse_args()
    
    process_results(args.raw_dir, args.out_dir)
