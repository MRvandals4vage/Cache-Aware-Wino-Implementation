#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd
import numpy as np
from scipy import stats

def export_latex_table(df, file_path, caption="Auto-generated LaTeX Table"):
    """Exports a pandas DataFrame to a booktabs LaTeX table."""
    try:
        # Format types safely for LaTeX
        latex_df = df.copy()
        
        # Format explicitly
        for col in latex_df.columns:
            if latex_df[col].dtype == 'float64':
                latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "N/A")
            elif latex_df[col].dtype == 'bool':
                latex_df[col] = latex_df[col].apply(lambda x: "Yes" if x else "No")
                
        # Escape column names
        latex_df.columns = [str(c).replace("_", "\\_") for c in latex_df.columns]
        
        # Generate standard table using whatever to_latex is available
        try:
            # Pandas >= 2.0
            table_content = latex_df.style.hide(axis="index").to_latex(hrules=True, column_format="l" * len(latex_df.columns))
            # The styler escape logic is handled inherently or via subsets, but let's assume it's clean enough
        except Exception:
            # Pandas < 2.0 or no Jinja2
            table_content = latex_df.to_latex(index=False, escape=True, column_format="l" * len(latex_df.columns))
            # Convert default hlines to booktabs if not already
            table_content = table_content.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule").replace("\\bottomrule", "\\bottomrule")
            if "\\toprule" not in table_content:
                table_content = table_content.replace("\\hline", "\\toprule", 1)  # Top
                table_content = table_content.replace("\\hline", "\\midrule", 1)  # Middle
                table_content = table_content.replace("\\hline", "\\bottomrule")  # Bottom
            
        with open(file_path, "w") as f:
            f.write("% " + caption + "\n")
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write(table_content)
            f.write(f"\\caption{{{caption}}}\n")
            f.write("\\end{table*}\n")
    except Exception as e:
        print(f"Failed to generate latex table {file_path}: {e}")

def process_microbenchmarks(raw_dir, out_dir, export_latex):
<<<<<<< HEAD
    csv_files = glob.glob(os.path.join(raw_dir, "micro_runs_*.csv"))
    if not csv_files:
        print("No microbenchmark raw files found.")
=======
    csv_file = os.path.join(raw_dir, "microbenchmark_raw_latencies.csv")
    if not os.path.exists(csv_file):
        print(f"No microbenchmark raw file found at {csv_file}.")
        return False
        
    try:
        alldf = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return False
        
    if alldf.empty or "latency_ms" not in alldf.columns:
        print("Empty or invalid microbenchmark dataset.")
        return False
        
    groups = alldf.groupby(["c_in", "c_out", "tile_dim", "fused", "threads"])
    
    # Identify baseline (fused=False, threads=1)
    baseline_dict = {}
    for name, group in groups:
        if not group["fused"].iloc[0] and group["threads"].iloc[0] == 1:
            baseline_key = (group["c_in"].iloc[0], group["c_out"].iloc[0])
            baseline_dict[baseline_key] = group["latency_ms"].values

    processed_data = []
    
    for name, group in groups:
        c_in, c_out, tile_dim, fused, threads = name
        latencies = group["latency_ms"].values
        n_runs = len(latencies)
        
        if n_runs < 30:
            print(f"ERROR: Microbenchmark paper table generation failed. Row C_in={c_in}, C_out={c_out}, Fused={fused}, Threads={threads} has {n_runs} runs. Minimum 30 required.")
            return False
        
        mean_lat = float(np.mean(latencies))
        median_lat = float(np.median(latencies))
        std_lat = float(np.std(latencies, ddof=1)) if n_runs > 1 else 0.0
        ci_95 = float(1.96 * (std_lat / np.sqrt(n_runs))) if n_runs > 0 else 0.0
        
        p_value = "N/A"
        effect_dir = "N/A"
        pct_improvement = "N/A"
        
        baseline_key = (c_in, c_out)
        if fused or threads > 1:
            if baseline_key in baseline_dict:
                base_lats = baseline_dict[baseline_key]
                base_mean = float(np.mean(base_lats))
                if len(base_lats) > 1 and len(latencies) > 1:
                    stat_val, p = stats.ttest_ind(latencies, base_lats, equal_var=False)
                    p_value = "< 1e-10" if p < 1e-10 else float(p)
                    effect_dir = "Faster" if mean_lat < base_mean else "Slower"
                    imp = ((base_mean - mean_lat) / base_mean) * 100.0
                    pct_improvement = round(imp, 2)

        processed_data.append({
            "C_in": int(c_in),
            "C_out": int(c_out),
            "Tile": f"F({int(tile_dim)},3)",
            "Fused": bool(fused),
            "MultiCore": bool(threads > 1),
            "Mean_Latency_ms": round(mean_lat, 4),
            "Median_Latency_ms": round(median_lat, 4),
            "StdDev_ms": round(std_lat, 4),
            "CI95_ms": round(ci_95, 4),
            "P_Value_vs_Baseline": p_value,
            "Effect_Direction": effect_dir,
            "Improvement_vs_Baseline_pct": pct_improvement,
            "Runs": int(n_runs)
        })
        
    if len(processed_data) < 28:
        print(f"ERROR: Microbenchmark paper table generation failed. Expected 28 total combinations, found {len(processed_data)}.")
        return False
        
    out_df = pd.DataFrame(processed_data).sort_values(by=["C_in", "C_out", "MultiCore", "Fused"])
    csv_path = os.path.join(out_dir, "paper_table_microbench.csv")
    out_df.to_csv(csv_path, index=False)
    
    if export_latex:
        tex_path = os.path.join(out_dir, "paper_table_microbench.tex")
        export_latex_table(out_df, tex_path, "Microbenchmark Custom Winograd Mechanism Evaluation")
    return True

def process_e2e(raw_dir, out_dir, logs_dir, export_latex):
    csv_files = glob.glob(os.path.join(raw_dir, "e2e_runs_*.csv"))
    # Filter out the error dummy file if it exists
    csv_files = [f for f in csv_files if "e2e_runs_error.csv" not in f]
    
    if not csv_files:
        print("No valid end-to-end raw files found.")
>>>>>>> e528b05 (centralizing outputs)
        return False
        
    df_list = []
    for rf in csv_files:
        try:
            df = pd.read_csv(rf)
            if not df.empty and "latency_ms" in df.columns:
                df_list.append(df)
        except Exception:
            pass
            
    if not df_list:
        return False
        
<<<<<<< HEAD
    alldf = pd.concat(df_list, ignore_index=True)
    groups = alldf.groupby(["c_in", "c_out", "tile_dim", "fused", "threads"])
    
    # Identify baseline (fused=False, threads=1)
    baseline_dict = {}
    for name, group in groups:
        if not group["fused"].iloc[0] and group["threads"].iloc[0] == 1:
            baseline_key = (group["c_in"].iloc[0], group["c_out"].iloc[0])
            baseline_dict[baseline_key] = group["latency_ms"].values

    processed_data = []
    
    for name, group in groups:
        c_in, c_out, tile_dim, fused, threads = name
        latencies = group["latency_ms"].values
=======
    platform_data = {"os": "Unknown"}
    try:
        with open(os.path.join(logs_dir, "platform_descriptor.json"), "r") as f:
            platform_data = json.load(f)
    except Exception:
        pass
        
    alldf = pd.concat(df_list, ignore_index=True)
    
    # Truthfulness Check: ensure exactly 4 models exist
    unique_models = alldf["model"].unique()
    if len(unique_models) < 4:
        print(f"ERROR: End-to-End paper table generation failed. Required 4 models, but only found {len(unique_models)}: {unique_models}. Run with --model all")
        return False
        
    groups = alldf.groupby(["model"])
    processed_data = []
    
    model_name_mapping = {
        "resnet18": "ResNet-18",
        "resnet34": "ResNet-34",
        "alexnet": "AlexNet",
        "vgg16": "VGG16"
    }
    
    raw_os = platform_data.get("os", "Unknown")
    cpu_model = str(platform_data.get("cpu_model", "")).lower()
    
    # Resolve platform name for paper
    paper_platform = raw_os
    if raw_os == "Darwin":
        paper_platform = "macOS (Apple Silicon)"
    elif raw_os == "Linux":
        if "bcm" in cpu_model or "raspberry" in cpu_model:
            paper_platform = "Raspberry Pi"
            if "bcm2711" in cpu_model:
                paper_platform = "Raspberry Pi 4"
            elif "bcm2712" in cpu_model:
                paper_platform = "Raspberry Pi 5"
        elif "tegra" in cpu_model or "cortex-a57" in cpu_model:
            paper_platform = "Jetson Nano"
    
    for name, group in groups:
        raw_model = name[0]
        mapped_model = model_name_mapping.get(raw_model, raw_model)
        
        latencies = group["latency_ms"].values
        fps_vals = group["throughput_fps"].values if "throughput_fps" in group.columns else []
>>>>>>> e528b05 (centralizing outputs)
        n_runs = len(latencies)
        
        mean_lat = float(np.mean(latencies))
        median_lat = float(np.median(latencies))
        std_lat = float(np.std(latencies, ddof=1)) if n_runs > 1 else 0.0
        ci_95 = float(1.96 * (std_lat / np.sqrt(n_runs))) if n_runs > 0 else 0.0
<<<<<<< HEAD
        
        p_value = "N/A"
        effect_dir = "N/A"
        pct_improvement = "N/A"
        
        baseline_key = (c_in, c_out)
        if fused or threads > 1:
            if baseline_key in baseline_dict:
                base_lats = baseline_dict[baseline_key]
                base_mean = float(np.mean(base_lats))
                if len(base_lats) > 1 and len(latencies) > 1:
                    stat_val, p = stats.ttest_ind(latencies, base_lats, equal_var=False)
                    # Don't print hard 0.0 for p-values underflow
                    p_value = "< 1e-10" if p < 1e-10 else float(p)
                    effect_dir = "Faster" if mean_lat < base_mean else "Slower"
                    imp = ((base_mean - mean_lat) / base_mean) * 100.0
                    pct_improvement = round(imp, 2)

        processed_data.append({
            "C_in": int(c_in),
            "C_out": int(c_out),
            "Tile": int(tile_dim),
            "Fused": bool(fused),
            "MultiCore": bool(threads > 1),
=======
        mean_fps = float(np.mean(fps_vals)) if len(fps_vals) > 0 else (1000.0 / mean_lat if mean_lat > 0 else 0.0)
        
        processed_data.append({
            "Model": mapped_model,
            "Platform": paper_platform,
>>>>>>> e528b05 (centralizing outputs)
            "Mean_Latency_ms": round(mean_lat, 4),
            "Median_Latency_ms": round(median_lat, 4),
            "StdDev_ms": round(std_lat, 4),
            "CI95_ms": round(ci_95, 4),
<<<<<<< HEAD
            "P_Value_vs_Baseline": p_value,
            "Effect_Direction": effect_dir,
            "Improvement_vs_Baseline_pct": pct_improvement,
            "Runs": int(n_runs)
        })
        
    out_df = pd.DataFrame(processed_data).sort_values(by=["C_in", "C_out", "MultiCore", "Fused"])
    csv_path = os.path.join(out_dir, "paper_table_microbench.csv")
    out_df.to_csv(csv_path, index=False)
    
    if export_latex:
        tex_path = os.path.join(out_dir, "paper_table_microbench.tex")
        export_latex_table(out_df, tex_path, "Microbenchmark Custom Winograd Mechanism Evaluation")
    return True

def process_e2e(raw_dir, out_dir, logs_dir, export_latex):
    csv_files = glob.glob(os.path.join(raw_dir, "e2e_runs_*.csv"))
    # Filter out the error dummy file if it exists
    csv_files = [f for f in csv_files if "e2e_runs_error.csv" not in f]
    
    if not csv_files:
        print("No valid end-to-end raw files found.")
        return False
        
    df_list = []
    for rf in csv_files:
        try:
            df = pd.read_csv(rf)
            if not df.empty and "latency_ms" in df.columns:
                df_list.append(df)
        except Exception:
            pass
            
    if not df_list:
        return False
        
    platform_data = {"os": "Unknown"}
    try:
        with open(os.path.join(logs_dir, "platform_descriptor.json"), "r") as f:
            platform_data = json.load(f)
    except Exception:
        pass
        
    alldf = pd.concat(df_list, ignore_index=True)
    
    # Truthfulness Check: ensure exactly 4 models exist
    unique_models = alldf["model"].unique()
    if len(unique_models) < 4:
        print(f"ERROR: End-to-End paper table generation failed. Required 4 models, but only found {len(unique_models)}: {unique_models}. Run with --model all")
        return False
        
    groups = alldf.groupby(["model"])
    processed_data = []
    
    model_name_mapping = {
        "resnet18": "ResNet-18",
        "resnet34": "ResNet-34",
        "alexnet": "AlexNet",
        "vgg16": "VGG16"
    }
    
    raw_os = platform_data.get("os", "Unknown")
    cpu_model = str(platform_data.get("cpu_model", "")).lower()
    
    # Resolve platform name for paper
    paper_platform = raw_os
    if raw_os == "Darwin":
        paper_platform = "macOS (Apple Silicon)"
    elif raw_os == "Linux":
        if "bcm" in cpu_model or "raspberry" in cpu_model:
            paper_platform = "Raspberry Pi"
            if "bcm2711" in cpu_model:
                paper_platform = "Raspberry Pi 4"
            elif "bcm2712" in cpu_model:
                paper_platform = "Raspberry Pi 5"
        elif "tegra" in cpu_model or "cortex-a57" in cpu_model:
            paper_platform = "Jetson Nano"
    
    for name, group in groups:
        raw_model = name[0]
        mapped_model = model_name_mapping.get(raw_model, raw_model)
        
        latencies = group["latency_ms"].values
        fps_vals = group["throughput_fps"].values if "throughput_fps" in group.columns else []
        n_runs = len(latencies)
        
        mean_lat = float(np.mean(latencies))
        median_lat = float(np.median(latencies))
        std_lat = float(np.std(latencies, ddof=1)) if n_runs > 1 else 0.0
        ci_95 = float(1.96 * (std_lat / np.sqrt(n_runs))) if n_runs > 0 else 0.0
        mean_fps = float(np.mean(fps_vals)) if len(fps_vals) > 0 else (1000.0 / mean_lat if mean_lat > 0 else 0.0)
        
        processed_data.append({
            "Model": mapped_model,
            "Platform": paper_platform,
            "Mean_Latency_ms": round(mean_lat, 4),
            "Median_Latency_ms": round(median_lat, 4),
            "StdDev_ms": round(std_lat, 4),
            "CI95_ms": round(ci_95, 4),
            "FPS": round(mean_fps, 2),
            "Runs": int(n_runs),
            "Warmup_Runs": 10 # Standard warmup mapped from prompt requirements
        })

    out_df = pd.DataFrame(processed_data).sort_values(by=["Model"])
    csv_path = os.path.join(out_dir, "paper_table_end_to_end.csv")
    out_df.to_csv(csv_path, index=False)
    
    if export_latex:
        tex_path = os.path.join(out_dir, "paper_table_end_to_end.tex")
        export_latex_table(out_df, tex_path, "End-to-End CNN Workload Characterization")
        
    return True

def process_autotiling(logs_dir, out_dir, export_latex):
    desc_path = os.path.join(logs_dir, "platform_descriptor.json")
    auto_path = os.path.join(logs_dir, "autotiling_decisions.json")
    
    if not os.path.exists(desc_path) or not os.path.exists(auto_path):
        return False
        
    try:
        with open(desc_path, "r") as f:
            platform_data = json.load(f)
        with open(auto_path, "r") as f:
            auto_data = json.load(f)
            
        processed_data = []
        for dec in auto_data:
            c_in = dec.get("c_in", 0)
            c_out = dec.get("c_out", 0)
            workload = f"C_in={c_in}, C_out={c_out}"
            selected = dec.get("selected_tile", {})
            
=======
            "FPS": round(mean_fps, 2),
            "Runs": int(n_runs),
            "Warmup_Runs": 10 # Standard warmup mapped from prompt requirements
        })

    out_df = pd.DataFrame(processed_data).sort_values(by=["Model"])
    csv_path = os.path.join(out_dir, "paper_table_end_to_end.csv")
    out_df.to_csv(csv_path, index=False)
    
    if export_latex:
        tex_path = os.path.join(out_dir, "paper_table_end_to_end.tex")
        export_latex_table(out_df, tex_path, "End-to-End CNN Workload Characterization")
        
    return True

def process_autotiling(logs_dir, out_dir, export_latex):
    desc_path = os.path.join(logs_dir, "platform_descriptor.json")
    auto_path = os.path.join(logs_dir, "autotiling_decisions.json")
    
    if not os.path.exists(desc_path) or not os.path.exists(auto_path):
        return False
        
    try:
        with open(desc_path, "r") as f:
            platform_data = json.load(f)
        with open(auto_path, "r") as f:
            auto_data = json.load(f)
            
        processed_data = []
        for dec in auto_data:
            c_in = dec.get("c_in", 0)
            c_out = dec.get("c_out", 0)
            workload = f"C_in={c_in}, C_out={c_out}"
            selected = dec.get("selected_tile", {})
            
>>>>>>> e528b05 (centralizing outputs)
            processed_data.append({
                "Platform": platform_data.get("os", "Unknown"),
                "CPU_Model": platform_data.get("cpu_model", "Unknown"),
                "L1D_Bytes": platform_data.get("l1d_size_bytes", "Unknown"),
                "L2_Bytes": platform_data.get("l2_size_bytes", "Unknown"),
                "Workload": workload,
                "Candidate_Tiles": "4, 6, 8",
                "Selected_Tile": int(selected.get("tile", 0)),
                "Alpha": 0.7,
                "Target_Cache": int(dec.get("l1_capacity", 0)),
                "Estimated_Working_Set_Bytes": int(dec.get("working_set", 0)),
                "Reuse_Score": round(float(dec.get("score", 0.0)), 4)
            })
            
        out_df = pd.DataFrame(processed_data)
        csv_path = os.path.join(out_dir, "paper_table_autotiling.csv")
        out_df.to_csv(csv_path, index=False)
        
        if export_latex:
            tex_path = os.path.join(out_dir, "paper_table_autotiling.tex")
            export_latex_table(out_df, tex_path, "Runtime Cache-Aware Tile Selection Decisions")
        return True
    except Exception as e:
        print(f"Error processing autotiling: {e}")
        return False

def run_all(raw_dir="artifacts/raw", out_dir="artifacts/processed", logs_dir="artifacts/logs", export_latex=True):
    os.makedirs(out_dir, exist_ok=True)
    m = process_microbenchmarks(raw_dir, out_dir, export_latex)
    e = process_e2e(raw_dir, out_dir, logs_dir, export_latex)
    a = process_autotiling(logs_dir, out_dir, export_latex)
    
    if not m and not e and not a:
        print("No outputs were processed. Run benchmarks first to generate raw data/logs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="artifacts/raw")
    parser.add_argument("--out-dir", default="artifacts/processed")
    parser.add_argument("--logs-dir", default="artifacts/logs")
    parser.add_argument("--export-latex", type=bool, default=True)
    args = parser.parse_args()
    run_all(args.raw_dir, args.out_dir, args.logs_dir, args.export_latex)
