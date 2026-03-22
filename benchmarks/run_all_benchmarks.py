#!/usr/bin/env python3
import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runtime_cache_probe import build_platform_descriptor, save_platform_descriptor
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.fused_winograd_kernel import FusedWinogradKernel
from src.locality_scheduler import LocalityScheduler

# Import ONNX execution from root
try:
    from export_onnx import export_models
    from onnx_inference import run_onnx_inference
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from process_results import run_all as process_paper_results
from generate_plots import run_all as generate_paper_plots

def generate_random_tasks(scheduler, c_in, c_out, h, w, tile_dim):
    return scheduler.generate_tile_tasks(h, w, tile_dim, c_in, c_out)

def setup_directories(base_dir):
    dirs = {
        "raw": os.path.join(base_dir, "raw"),
        "processed": os.path.join(base_dir, "processed"),
        "logs": os.path.join(base_dir, "logs"),
        "plots": os.path.join(base_dir, "plots")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def parse_args():
    parser = argparse.ArgumentParser(description="Cache-Aware Winograd Benchmark CLI")
    parser.add_argument("--mode", choices=["micro", "end-to-end"], default="micro", help="Benchmark mode")
    parser.add_argument("--tile", type=int, choices=[4, 6, 8], default=None, help="Force tile size (overrides autotiler)")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for multi-core")
    parser.add_argument("--fused", action="store_true", help="Enable fused execution")
    parser.add_argument("--runs", type=int, default=10, help="Number of measurement runs")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "alexnet", "vgg16", "all"], help="Model to benchmark (for end-to-end mode)")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Base directory for artifacts output")
    parser.add_argument("--paper-assets", choices=["all", "tables", "figures", "none"], default="none", help="Generate paper-ready assets")
    
    # We use type=str and check lower() to support 'true'/'false' correctly
    parser.add_argument("--export-latex", type=str, choices=["true", "false"], default="true", help="Export LaTeX tables alongside CSVs")
    
    return parser.parse_args()

def run_micro(args, dirs, autotiler, platform_desc):
    print("Running Microbenchmarks...")
    
    configs = [
        {"c_in": 16, "c_out": 32, "h": 14, "w": 14},
        {"c_in": 32, "c_out": 64, "h": 14, "w": 14},
    ]
    
    kernel = FusedWinogradKernel()
    scheduler = LocalityScheduler()
    
    raw_data = []
    
    for cfg in configs:
        c_in, c_out, h, w = cfg["c_in"], cfg["c_out"], cfg["h"], cfg["w"]
        
        if args.tile is not None:
            tile_dim = args.tile
            print(f"Using forced tile size: {tile_dim}")
        else:
            decision = autotiler.select_best_tile(c_in, c_out)
            tile_dim = decision["selected_tile"]["tile"]
            autotiler.save_autotiling_decision(decision, os.path.join(dirs["logs"], "autotiling_decisions.json"))
            print(f"Autotiler selected tile size {tile_dim} config for {c_in}x{c_out}")
            
        tasks = generate_random_tasks(scheduler, c_in, c_out, h, w, tile_dim)
        ordered_tasks = scheduler.group_tasks_by_channel_locality(tasks)
        
        if args.threads > 1:
            plan = scheduler.schedule_multi_core(ordered_tasks, num_cores=args.threads)
        else:
            plan = scheduler.schedule_single_core(ordered_tasks)
            
        # Provide dummy data
        input_tile = np.random.randn(c_in, 4, 4).astype(np.float32)
        U = np.random.randn(c_out, c_in, 4, 4).astype(np.float32)
        
        # Warmup
        for _ in range(args.warmup):
            for task_item in plan:
                if args.fused:
                    kernel.run_fused(input_tile, U)
                else:
                    kernel.run_non_fused(input_tile, U)
                    
        # Measurement
        for run_id in range(args.runs):
            t0 = time.perf_counter()
            for task_item in plan:
                if args.fused:
                    kernel.run_fused(input_tile, U)
                else:
                    kernel.run_non_fused(input_tile, U)
            t1 = time.perf_counter()
            
            duration_ms = (t1 - t0) * 1000.0
            
            raw_data.append({
                "mode": "micro",
                "c_in": c_in,
                "c_out": c_out,
                "h": h, "w": w,
                "tile_dim": tile_dim,
                "fused": args.fused,
                "threads": args.threads,
                "run_id": run_id,
                "latency_ms": duration_ms
            })
            
    df = pd.DataFrame(raw_data)
    out_file = os.path.join(dirs["raw"], f"micro_runs_fused_{args.fused}_th_{args.threads}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {len(raw_data)} raw samples to {out_file}")

def run_e2e(args, dirs):
    print(f"Running End-to-End Benchmarks (ONNX Baseline)...")
    if not ONNX_AVAILABLE:
        print("ONNX modules not available or failed to import. Ensure torch and onnxruntime are installed.")
        pd.DataFrame([{"mode": "end-to-end", "status": "unsupported", "reason": "missing_dependencies"}]).to_csv(
            os.path.join(dirs["raw"], "e2e_runs_error.csv"), index=False
        )
        return
        
    models_to_test = ["resnet18", "resnet34", "alexnet", "vgg16"] if args.model == "all" else [args.model]
    
    # Run export script to ensure .onnx files exist
    export_models()
    
    for model in models_to_test:
        model_path = f"{model}.onnx"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping...")
            continue
            
        print(f"Benchmarking {model}...")
        try:
            result = run_onnx_inference(model_path, num_iterations=args.runs, num_warmup=args.warmup)
            
            raw_data = []
            for run_id in range(len(result["latencies_ms"])):
                raw_data.append({
                    "mode": "end-to-end",
                    "model": model,
                    "run_id": run_id,
                    "latency_ms": result["latencies_ms"][run_id],
                    "throughput_fps": result["throughputs_fps"][run_id]
                })
                
            df = pd.DataFrame(raw_data)
            out_file = os.path.join(dirs["raw"], f"e2e_runs_{model}.csv")
            df.to_csv(out_file, index=False)
            print(f"Saved {len(raw_data)} e2e samples to {out_file}")
            
        except Exception as e:
            print(f"Failed ONNX execution for {model}: {e}")

def main():
    args = parse_args()
    dirs = setup_directories(args.out_dir)
    
    # 1. Platform Descriptor
    platform_desc = build_platform_descriptor()
    desc_path = os.path.join(dirs["logs"], "platform_descriptor.json")
    save_platform_descriptor(platform_desc, desc_path)
    print(f"Generated platform descriptor at {desc_path}")
    
    # 2. Autotiler
    autotiler = CacheAdaptiveAutotiler(platform_descriptor=platform_desc)
    
    if args.mode == "micro":
        run_micro(args, dirs, autotiler, platform_desc)
    else:
        run_e2e(args, dirs)
        
    export_latex = (args.export_latex.lower() == "true")
    
    # 3. Post-Process Paper Assets
    if args.paper_assets in ["all", "tables"]:
        print("\n--- Generating Paper Tables ---")
        process_paper_results(dirs["raw"], dirs["processed"], dirs["logs"], export_latex)
        
    if args.paper_assets in ["all", "figures"]:
        print("\n--- Generating Paper Figures ---")
        if args.paper_assets == "figures":
            # If ONLY figures requested, ensure stats exist first by running a quiet process
            process_paper_results(dirs["raw"], dirs["processed"], dirs["logs"], export_latex=False)
        generate_paper_plots(dirs["processed"], dirs["plots"])

if __name__ == "__main__":
    main()
