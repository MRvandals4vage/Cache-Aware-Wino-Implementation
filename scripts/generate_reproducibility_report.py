#!/usr/bin/env python3
"""
Generate reproducibility_report.md

For every metric from the paper, lists:
  - Paper's claimed value
  - Freshly measured value (with n, CI95, p-value if applicable)
  - MATCH or MISMATCH (with %diff)
  - Path to the raw log file backing it

Exits non-zero if any required raw log is missing.
"""
import os
import sys
import csv
import json
import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Paper-claimed values
# ---------------------------------------------------------------------------
PAPER_METRICS = {
    # Microbenchmarks (Jetson Nano)
    "JN_64x64_1T_fused_improvement": {"value": 43.12, "unit": "%", "desc": "Jetson Nano (64,64) 1-thread fused improvement"},
    "RPi4_128x128_m4_improvement": {"value": 63.91, "unit": "%", "desc": "RPi4 (128,128) m=4 fused improvement"},
    "JN_64x64_4T_fused_vs_unfused": {"value": 33.11, "unit": "%", "desc": "JN (64,64) 4-thread fused vs 4-thread unfused"},
    "VGG16_JN_energy_improvement": {"value": 5.5, "unit": "%", "desc": "End-to-end energy improvement VGG16 JN"},
    "vs_TVM_AutoTVM": {"value": 29.5, "unit": "%", "desc": "CacheWinograd vs TVM/AutoTVM improvement"},
    "vs_ArmCL": {"value": 32.1, "unit": "%", "desc": "CacheWinograd vs ArmCL improvement"},
    "RPi4_32x16_max_regression_avoided": {"value": 107.18, "unit": "%", "desc": "Max regression avoided (RPi4 32x16)"},
    "scheduling_overhead_us": {"value": 20, "unit": "μs", "desc": "Scheduling overhead"},
    "scheduling_overhead_per_tile_ns": {"value": 5, "unit": "ns", "desc": "Per-tile scheduling overhead"},
    # (128,128) re-verification
    "JN_128x128_baseline_ms": {"value": 860.65, "unit": "ms", "desc": "JN (128,128) baseline non-fused"},
    "JN_128x128_fused_ms": {"value": 807.97, "unit": "ms", "desc": "JN (128,128) fused"},
    "JN_128x128_improvement": {"value": -6.52, "unit": "%", "desc": "JN (128,128) improvement (regression)"},
}

# Required raw log files (verify-all fails loudly if any are missing)
REQUIRED_LOGS = {
    "T2": "raw_logs/jetson_128x128_reverify.csv",
    "T3": "raw_logs/tvm_armcl_full_traces.csv",
    "T4": "raw_logs/rpi4_adaptive_m_sweep.csv",
    "T5_e2e": "summary/e2e_comparison_table.csv",
    "T6": "raw_logs/register_spill_128x128.csv",
    "T7": "raw_logs/energy_per_run_vgg16.csv",
    "microbench": "artifacts/raw/microbenchmark_raw_latencies.csv",
}

# Optional raw log files (reported as MISSING but do NOT cause non-zero exit)
OPTIONAL_LOGS = {
    "T8 (optional)": "raw_logs/combined_policy.csv",
}


def _read_csv(path):
    """Read CSV and return list of dicts."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _match_status(paper_val, measured_val, threshold_pct=5.0):
    """Determine MATCH/MISMATCH status."""
    if paper_val == 0:
        return "MATCH" if abs(measured_val) < 0.01 else "MISMATCH", 0.0
    pct_diff = ((measured_val - paper_val) / abs(paper_val)) * 100.0
    status = "MATCH" if abs(pct_diff) < threshold_pct else "MISMATCH"
    return status, pct_diff


def generate_report():
    """Generate the reproducibility report."""
    lines = []
    lines.append("# Reproducibility Report")
    lines.append(f"\nGenerated: {datetime.datetime.now().isoformat()}")
    lines.append(f"\n---\n")

    missing_logs = []
    all_results = []

    # ---------------------------------------------------------------------------
    # Check required logs
    # ---------------------------------------------------------------------------
    lines.append("## Required Raw Log Files\n")
    lines.append("| Task | File | Status |")
    lines.append("| :--- | :--- | :----: |")
    for task, path in sorted(REQUIRED_LOGS.items()):
        exists = os.path.exists(path)
        status = "✓ FOUND" if exists else "✗ MISSING"
        lines.append(f"| {task} | `{path}` | {status} |")
        if not exists:
            missing_logs.append((task, path))
    for task, path in sorted(OPTIONAL_LOGS.items()):
        exists = os.path.exists(path)
        status = "✓ FOUND" if exists else "⚠️ OPTIONAL/MISSING"
        lines.append(f"| {task} | `{path}` | {status} |")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # T2: Jetson Nano (128,128) re-verification
    # ---------------------------------------------------------------------------
    lines.append("## T2: Jetson Nano (128,128) Re-verification\n")
    t2_summary = _read_csv("summary/jetson_128x128_reverify_summary.csv")
    if t2_summary:
        lines.append("| Method | Paper (ms) | Measured (ms) | n | CI95 | p-value | Status | %diff | Raw Log |")
        lines.append("| :----- | ---------: | ------------: | -: | ---: | :------ | :----: | ----: | :------ |")
        for row in t2_summary:
            lines.append(f"| {row['method']} | {row['paper_claimed_ms']} | {row['mean_ms']} | "
                          f"{row['n']} | ±{row['ci95_ms']} | {row['p_value']} | "
                          f"{row['match']} | {row['pct_diff_from_paper']}% | `raw_logs/jetson_128x128_reverify.csv` |")
    else:
        lines.append("> **NOT RUN** — Execute `python benchmarks/jetson_128x128_reverify.py` on Jetson Nano")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # T3: TVM/ArmCL Statistical Rigor
    # ---------------------------------------------------------------------------
    lines.append("## T3: TVM/ArmCL Statistical Rigor\n")
    t3_summary = _read_csv("summary/tvm_armcl_significance.csv")
    if t3_summary:
        lines.append("| Config | Baseline | n | Mean (ms) | CI95 | p-value | CW Mean | %Improvement | Status |")
        lines.append("| :----- | :------- | -: | --------: | ---: | :------ | ------: | -----------: | :----: |")
        for row in t3_summary:
            lines.append(f"| {row['config']} | {row['baseline']} | {row['n']} | {row['mean_ms']} | "
                          f"±{row['ci95_ms']} | {row['p_value']} | {row['cw_mean_ms']} | "
                          f"{row['pct_improvement']} | {row['status']} |")
    else:
        lines.append("> **NOT RUN** — Execute `python benchmarks/tvm_armcl_rigor.py` on Jetson Nano")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # T4: RPi4 Adaptive-m Sweep
    # ---------------------------------------------------------------------------
    lines.append("## T4: RPi4 Adaptive-m Sweep\n")
    t4_summary = _read_csv("summary/rpi4_adaptive_m_verification.csv")
    if t4_summary:
        lines.append("| Config | Policy m | Empirical Best m | m=2 Mean | m=2 CI95 | m=6 Mean | m=6 CI95 | p(m2 vs m6) | Match |")
        lines.append("| :----- | -------: | ---------------: | -------: | -------: | -------: | -------: | :---------- | :---: |")
        for row in t4_summary:
            match = "✓" if row.get("policy_matches_empirical", "").lower() == "true" else "✗"
            lines.append(f"| {row['config']} | {row['policy_m']} | {row['empirical_best_m']} | "
                          f"{row['m2_mean_ms']}ms | ±{row.get('m2_ci95_ms','N/A')} | "
                          f"{row['m6_mean_ms']}ms | ±{row.get('m6_ci95_ms','N/A')} | "
                          f"{row.get('p_value_m2_vs_m6','N/A')} | {match} |")
    else:
        lines.append("> **NOT RUN** — Execute `python benchmarks/rpi4_adaptive_m_sweep.py` on Raspberry Pi 4")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # T5: End-to-End Comparison
    # ---------------------------------------------------------------------------
    lines.append("## T5: End-to-End Comparison\n")
    lines.append("> **METHODOLOGY NOTE (constraint #5):** CacheWinograd end-to-end latency is a "
                  "**layer-time aggregate** of isolated per-layer kernel measurements, NOT a single "
                  "fused end-to-end trace. Baseline end-to-end numbers (ORT, TVM, ArmCL) ARE true "
                  "single-process runs. This asymmetry is stated explicitly and not hidden.\n")
    t5_summary = _read_csv("summary/e2e_comparison_table.csv")
    if t5_summary:
        lines.append("| Model | Platform | Baseline | Mean (ms) | CI95 | p-value | %Improvement |")
        lines.append("| :---- | :------- | :------- | --------: | ---: | :------ | -----------: |")
        for row in t5_summary:
            lines.append(f"| {row['model']} | {row['platform']} | {row['baseline']} | "
                          f"{row['mean_ms']} | ±{row['CI95']} | {row['p_value']} | "
                          f"{row['pct_improvement']} |")
    else:
        lines.append("> **NOT RUN** — Execute `python benchmarks/e2e_benchmark.py` on target device")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # T6: Register Spill
    # ---------------------------------------------------------------------------
    lines.append("## T6: Register-Spill Instrumentation\n")
    t6_data = _read_csv("raw_logs/register_spill_128x128.csv")
    if t6_data:
        for row in t6_data:
            method = row.get("method", "unknown")
            lines.append(f"### Method: {method}\n")
            for k, v in sorted(row.items()):
                if k not in ("timestamp", "method"):
                    lines.append(f"- **{k}**: {v}")
            lines.append("")
    else:
        lines.append("> **NOT RUN** — Execute `python benchmarks/register_spill_instrument.py`")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # T7: Energy Per-Run
    # ---------------------------------------------------------------------------
    lines.append("## T7: Energy Per-Run Distribution (VGG16)\n")
    t7_summary = _read_csv("summary/energy_vgg16_summary.csv")
    if t7_summary:
        for row in t7_summary:
            lines.append(f"- **Model**: {row['model']}")
            lines.append(f"- **n**: {row['n']}")
            lines.append(f"- **Mean Energy**: {row['mean_energy_mj']} mJ")
            lines.append(f"- **CI95 Energy**: ±{row['ci95_energy_mj']} mJ")
            lines.append(f"- **Mean Latency**: {row['mean_latency_ms']} ms")
            lines.append(f"- **Mean Power**: {row['mean_power_mw']} mW")
            lines.append(f"- **On Jetson (INA3221)**: {row['is_jetson']}")
    else:
        lines.append("> **NOT RUN** — Execute `python benchmarks/energy_per_run.py` on Jetson Nano")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # T8: Combined Policy
    # ---------------------------------------------------------------------------
    lines.append("## T8: Combined Fusion+Parallelism Policy\n")
    t8_summary = _read_csv("summary/combined_policy_summary.csv")
    if t8_summary:
        lines.append("| Config | Regime | Policy Mode | Policy Mean (ms) | Best Mode | Best Mean (ms) | Optimal | p-value |")
        lines.append("| :----- | :----- | :---------- | ---------------: | :-------- | -------------: | :-----: | :------ |")
        for row in t8_summary:
            optimal = "✓" if row.get("policy_is_optimal", "").lower() == "true" else "✗"
            lines.append(f"| {row['config']} | {row['regime']} | {row['policy_mode']} | "
                          f"{row['policy_mean_ms']} | {row['best_mode']} | {row['best_mean_ms']} | "
                          f"{optimal} | {row['p_value']} |")
    else:
        lines.append("> **NOT RUN** — Execute `python benchmarks/combined_policy.py`")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # Global Verification Pass
    # ---------------------------------------------------------------------------
    lines.append("## Global Verification Pass\n")
    lines.append("| Metric | Paper Value | Measured Value | n | CI95 | Status | %Diff | Raw Log |")
    lines.append("| :----- | ----------: | -------------: | -: | ---: | :----: | ----: | :------ |")

    # Process existing microbenchmark data
    microbench = _read_csv("artifacts/processed/paper_table_microbench.csv")
    if microbench:
        for row in microbench:
            c_in, c_out = row.get("C_in", ""), row.get("C_out", "")
            fused = row.get("Fused", "")
            mc = row.get("MultiCore", "")
            imp = row.get("Improvement_vs_Baseline_pct", "N/A")
            if imp != "N/A":
                lines.append(f"| ({c_in},{c_out}) F={fused} T={'4' if mc=='True' else '1'} | "
                              f"(see paper) | {row.get('Mean_Latency_ms', 'N/A')}ms / {imp}% | "
                              f"{row.get('Runs', 'N/A')} | ±{row.get('CI95_ms', 'N/A')} | "
                              f"MEASURED | — | `artifacts/raw/microbenchmark_raw_latencies.csv` |")

    # Add paper metrics that need on-device verification
    for key, info in PAPER_METRICS.items():
        measured = "PENDING"
        n = "—"
        ci95 = "—"
        status = "PENDING"
        pct_diff = "—"
        raw_log = "—"

        # Try to match from existing data
        if "128x128" in key and t2_summary:
            for row in t2_summary:
                if "baseline" in key.lower() and row["method"] == "baseline_nonfused":
                    measured = f"{row['mean_ms']}ms"
                    n = row["n"]
                    ci95 = f"±{row['ci95_ms']}"
                    status = row["match"]
                    pct_diff = f"{row['pct_diff_from_paper']}%"
                    raw_log = "`raw_logs/jetson_128x128_reverify.csv`"
                elif "fused" in key.lower() and "baseline" not in key.lower() and row["method"] == "fused":
                    measured = f"{row['mean_ms']}ms"
                    n = row["n"]
                    ci95 = f"±{row['ci95_ms']}"
                    status = row["match"]
                    pct_diff = f"{row['pct_diff_from_paper']}%"
                    raw_log = "`raw_logs/jetson_128x128_reverify.csv`"

        lines.append(f"| {info['desc']} | {info['value']}{info['unit']} | {measured} | "
                      f"{n} | {ci95} | {status} | {pct_diff} | {raw_log} |")

    lines.append(f"\n---\n")

    # ---------------------------------------------------------------------------
    # Missing logs warning
    # ---------------------------------------------------------------------------
    if missing_logs:
        lines.append("## ⚠️ Missing Required Logs\n")
        lines.append("The following raw log files are required but missing:\n")
        for task, path in missing_logs:
            lines.append(f"- **{task}**: `{path}`")
        lines.append("\nRun `make verify-all` on the target devices to collect all data.")

    # Write report
    report_path = "reproducibility_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Report] Written to {report_path}")
    print(f"         {len(missing_logs)} required log(s) missing")

    if missing_logs:
        print("\nERROR: Missing required raw logs. Exit code 1.")
        return 1
    return 0


def main():
    sys.exit(generate_report())


if __name__ == "__main__":
    main()
