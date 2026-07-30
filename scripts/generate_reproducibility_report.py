#!/usr/bin/env python3
"""
Generate reproducibility_report.md

Compares (a) prior logged value from raw_logs_prior/summary_prior/,
(b) fresh measured value from raw_logs/summary/,
(c) MATCH/MISMATCH with %diff, for every metric/table in the paper.

Constraint #5: Do NOT edit paper's claimed numbers. Only report diffs.
Constraint #6: CacheWinograd E2E is layer-time aggregate — explicitly stated.
Constraint #7: sys.exit(1) if any REQUIRED log is missing when called by verify-all.
"""
import os, sys, csv, json, datetime, argparse

OUT_FILE = "reproducibility_report.md"

# Required logs for verify-all (non-zero exit if missing)
REQUIRED_LOGS = {
    "A_cache_probe":          "raw_logs/jetson_cacheprobe.csv",
    "B_ws_validation":        "raw_logs/jetson_ws_validation.csv",
    "C_jetson_microbench":    "raw_logs/jetson_main_microbench.csv",
    "D_tvm_armcl":            "raw_logs/tvm_armcl_full_traces.csv",
    "E_ablation":             "raw_logs/jetson_ablation.csv",
    "F_l2_macsj":             "raw_logs/jetson_l2_macsj.csv",
    "G_fallback_heldout":     "raw_logs/jetson_fallback_heldout.csv",
    "H_scheduling_overhead":  "raw_logs/jetson_scheduling_overhead.csv",
    "I_register_spill":       "raw_logs/register_spill_128x128.csv",
    "J_energy_per_run":       "raw_logs/energy_per_run_vgg16.csv",
    "K_e2e_jetson":           "summary/e2e_comparison_table.csv",
    "M_rpi4_microbench":      "raw_logs/rpi4_main_microbench.csv",
    "N_adaptive_m":           "raw_logs/rpi4_adaptive_m_sweep.csv",
    "O_e2e_rpi4":             "summary/e2e_comparison_table.csv",
}

OPTIONAL_LOGS = {
    "L_combined_policy":      "raw_logs/combined_policy.csv",
    "P_apple_silicon":        "raw_logs/e2e_alexnet_macos_apple_silicon.csv",
}


def _csv(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _prior(subpath):
    """Get prior version of a file (from raw_logs_prior/ or summary_prior/)."""
    p = subpath.replace("raw_logs/", "raw_logs_prior/").replace("summary/", "summary_prior/")
    return p


def _pct_diff(fresh, prior):
    """Return %diff string between two numeric strings."""
    try:
        a, b = float(fresh), float(prior)
        diff = (a - b) / abs(b) * 100.0
        return f"{diff:+.2f}%"
    except Exception:
        return "N/A"


def _match(pct_diff_str, threshold_pct=5.0):
    try:
        v = float(pct_diff_str.replace("%", ""))
        return "✓ MATCH" if abs(v) <= threshold_pct else "✗ MISMATCH"
    except Exception:
        return "—"


def section(lines, title):
    lines.append(f"\n---\n\n## {title}\n")


def subsection(lines, title):
    lines.append(f"\n### {title}\n")


def table_row(lines, cols):
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")


def table_header(lines, cols):
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(":---" for _ in cols) + " |")


def generate_report(verify_mode=False):
    ts = datetime.datetime.now().isoformat()
    lines = []
    missing_logs = []

    lines.append(f"# CacheWinograd Reproducibility Report")
    lines.append(f"\nGenerated: {ts}")
    lines.append("\n> **METHODOLOGY NOTE (Constraint #6):** CacheWinograd end-to-end latency "
                 "is a **layer-time aggregate** of isolated per-layer kernel measurements, "
                 "NOT a single fused end-to-end trace. Baseline end-to-end numbers "
                 "(ORT, TVM, ArmCL) ARE true single-process runs. This asymmetry is "
                 "stated explicitly and not hidden in any table below.\n")
    lines.append("\n> **Column definitions:**\n"
                 "> - **Prior**: last value from `raw_logs_prior/` / `summary_prior/`\n"
                 "> - **Fresh**: value measured today on-device\n"
                 "> - **%Diff**: (fresh − prior) / |prior| × 100\n"
                 "> - **Match**: ✓ if |%diff| ≤ 5%, ✗ otherwise\n")

    # =========================================================================
    # Section 0: Log file inventory
    # =========================================================================
    section(lines, "0. Log File Inventory")
    table_header(lines, ["Task", "File", "Status", "Prior File", "Prior Status"])
    for task, path in sorted(REQUIRED_LOGS.items()):
        fresh_ok = "✓ FOUND" if os.path.exists(path) else "✗ MISSING (REQUIRED)"
        if not os.path.exists(path):
            missing_logs.append((task, path))
        prior_path = _prior(path)
        prior_ok = "✓ FOUND" if os.path.exists(prior_path) else "⚠ MISSING"
        table_row(lines, [task, f"`{path}`", fresh_ok, f"`{prior_path}`", prior_ok])
    for task, path in sorted(OPTIONAL_LOGS.items()):
        fresh_ok = "✓ FOUND" if os.path.exists(path) else "⚠ OPTIONAL"
        prior_path = _prior(path)
        prior_ok = "✓ FOUND" if os.path.exists(prior_path) else "⚠ MISSING"
        table_row(lines, [task, f"`{path}`", fresh_ok, f"`{prior_path}`", prior_ok])

    # =========================================================================
    # Section A: Cache params
    # =========================================================================
    section(lines, "A. Cache Parameters")
    fresh_rows = _csv("raw_logs/jetson_cacheprobe.csv")
    prior_rows = _csv("raw_logs_prior/jetson_cacheprobe.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskA` on Jetson Nano\n")
    else:
        table_header(lines, ["Cache", "Paper (bytes)", "Prior (bytes)", "Fresh (bytes)",
                              "%Diff Prior→Fresh", "Paper Match", "Prior→Fresh Match"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows if r["cache_level"] == row["cache_level"]), None)
            prior_val = prior["measured_bytes"] if prior else "N/A"
            pd = _pct_diff(row["measured_bytes"], prior_val)
            table_row(lines, [
                row["cache_level"], row["paper_bytes"], prior_val,
                row["measured_bytes"], pd, row["status"], _match(pd),
            ])

    # =========================================================================
    # Section B: Working-set model validation
    # =========================================================================
    section(lines, "B. Working-Set Model Validation")
    fresh_rows = _csv("summary/ws_model_validation_summary.csv")
    prior_rows = _csv("summary_prior/ws_model_validation_summary.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskB` on Jetson Nano\n")
    else:
        table_header(lines, ["Config", "WS Model (B)", "Paper WS_base", "Paper WS_ext",
                              "Miss Rate % (fresh)", "Miss Rate % (prior)", "%Diff"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows if r["config"] == row["config"]), None)
            prior_miss = prior["cache_miss_rate_pct"] if prior else "N/A"
            pd = _pct_diff(row["cache_miss_rate_pct"], prior_miss)
            table_row(lines, [
                row["config"], row["ws_model_bytes"],
                row.get("paper_ws_base", "N/A"), row.get("paper_ws_ext", "N/A"),
                row["cache_miss_rate_pct"], prior_miss, pd,
            ])

    # =========================================================================
    # Section C: Jetson Nano main microbenchmark — ALL 7 configs
    # =========================================================================
    section(lines, "C. Jetson Nano Main Microbenchmark (All 7 Configs)")
    fresh_rows = _csv("summary/jetson_main_microbench_summary.csv")
    prior_rows = _csv("summary_prior/jetson_main_microbench_summary.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskC` on Jetson Nano\n")
    else:
        table_header(lines, [
            "Config", "Prior Baseline ms", "Fresh Baseline ms", "Prior Fused ms",
            "Fresh Fused ms", "Prior Imp%", "Fresh Imp%", "%Diff Imp",
            "Paper Imp%", "Paper Match", "p-value", "Sig",
        ])
        for row in fresh_rows:
            prior = next((r for r in prior_rows if r["config"] == row["config"]), None)
            p_bl  = prior["baseline_mean_ms"] if prior else "N/A"
            p_fu  = prior["fused_mean_ms"]    if prior else "N/A"
            p_imp = prior["improvement_pct"]  if prior else "N/A"
            pd = _pct_diff(row["improvement_pct"], p_imp)
            table_row(lines, [
                row["config"], p_bl, row["baseline_mean_ms"],
                p_fu, row["fused_mean_ms"],
                p_imp, row["improvement_pct"], pd,
                row.get("paper_improvement_pct", "N/A"),
                row.get("match", "—"),
                row.get("p_value", "N/A"),
                row.get("significance", "N/A"),
            ])

    # =========================================================================
    # Section D: TVM/AutoTVM + ArmCL statistical rigor
    # =========================================================================
    section(lines, "D. TVM / AutoTVM / ArmCL Statistical Rigor")
    fresh_rows = _csv("summary/tvm_armcl_significance.csv")
    prior_rows = _csv("summary_prior/tvm_armcl_significance.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskD` on Jetson Nano\n")
    else:
        table_header(lines, ["Config", "Backend", "Fresh Mean ms", "Fresh CI95",
                              "Prior Mean ms", "%Diff", "p-value", "Significant?"])
        for row in fresh_rows:
            key = (row.get("config", ""), row.get("backend", ""))
            prior = next((r for r in prior_rows
                          if r.get("config") == key[0] and r.get("backend") == key[1]), None)
            p_mean = prior["mean_ms"] if prior else "N/A"
            pd = _pct_diff(row.get("mean_ms", "N/A"), p_mean)
            table_row(lines, [
                row.get("config"), row.get("backend"),
                row.get("mean_ms"), row.get("ci95"),
                p_mean, pd,
                row.get("p_value", "N/A"), row.get("status", "N/A"),
            ])

    # =========================================================================
    # Section E: Ablation
    # =========================================================================
    section(lines, "E. Multi-Core / Fusion × Threading Ablation")
    fresh_rows = _csv("summary/jetson_ablation_summary.csv")
    prior_rows = _csv("summary_prior/jetson_ablation_summary.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskE` on Jetson Nano\n")
    else:
        table_header(lines, ["Config", "Mode", "Fresh Mean ms", "Fresh CI95",
                              "Prior Mean ms", "%Diff", "Paper Imp%", "Match"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows
                          if r.get("config") == row.get("config")
                          and r.get("mode") == row.get("mode")), None)
            p_mean = prior["mean_ms"] if prior else "N/A"
            pd = _pct_diff(row.get("mean_ms"), p_mean)
            table_row(lines, [
                row.get("config"), row.get("mode"),
                row.get("mean_ms"), row.get("ci95_ms"),
                p_mean, pd,
                row.get("paper_fused_4T_vs_unfused_4T_pct", "N/A"),
                row.get("match", "—"),
            ])

    # =========================================================================
    # Section F: L2 hit rate + MACs/Joule
    # =========================================================================
    section(lines, "F. L2 Hit Rate + MACs/Joule")
    fresh_rows = _csv("summary/jetson_l2_macsj_summary.csv")
    prior_rows = _csv("summary_prior/jetson_l2_macsj_summary.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskF` on Jetson Nano\n")
    else:
        table_header(lines, ["Config", "Method", "Fresh Mean ms", "MACs/J (fresh)",
                              "Prior MACs/J", "%Diff MACs/J"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows
                          if r.get("config") == row.get("config")
                          and r.get("method") == row.get("method")), None)
            p_mj = prior["macs_per_joule"] if prior else "N/A"
            pd = _pct_diff(row.get("macs_per_joule", "N/A"), p_mj)
            table_row(lines, [
                row.get("config"), row.get("method"),
                row.get("mean_ms"), row.get("macs_per_joule"), p_mj, pd,
            ])

    # =========================================================================
    # Section G: Fallback-guard verification (all 8 rows = 6 from C + 2 from G)
    # =========================================================================
    section(lines, "G. Fallback-Guard Verification Table (All 8 Rows)")
    subsection(lines, "G1. Standard configs (from Task C data)")
    lines.append("*(Rows for (16,32),(32,16),(32,32),(32,64),(64,32),(64,64) "
                 "come from Section C above — see there for details)*\n")
    subsection(lines, "G2. Held-out configs")
    fresh_rows = _csv("summary/jetson_fallback_heldout_summary.csv")
    prior_rows = _csv("summary_prior/jetson_fallback_heldout_summary.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskG` on Jetson Nano\n")
    else:
        table_header(lines, ["Config", "Fallback?", "Tile", "WS (B)", "L1 (B)",
                              "Fresh Imp%", "Prior Imp%", "%Diff", "p-value", "Sig"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows if r.get("config") == row.get("config")), None)
            p_imp = prior["improvement_pct"] if prior else "N/A"
            pd = _pct_diff(row.get("improvement_pct"), p_imp)
            table_row(lines, [
                row.get("config"), row.get("fallback_triggered"),
                row.get("tile"), row.get("working_set_bytes"), row.get("l1_capacity_bytes"),
                row.get("improvement_pct"), p_imp, pd,
                row.get("p_value"), row.get("significance"),
            ])

    # =========================================================================
    # Section H: Scheduling overhead
    # =========================================================================
    section(lines, "H. Scheduling Overhead — Measured vs Architectural Estimate")
    fresh_rows = _csv("summary/jetson_scheduling_overhead_summary.csv")
    prior_rows = _csv("summary_prior/jetson_scheduling_overhead_summary.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskH` on Jetson/any Linux\n")
    else:
        table_header(lines, ["Phase", "Config", "Paper Claim", "Prior (μs)",
                              "Fresh (μs)", "CI95 (μs)", "%Diff", "Match"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows if r.get("phase") == row.get("phase")
                          and r.get("config") == row.get("config")), None)
            p_us = prior["mean_us"] if prior else "N/A"
            pd = _pct_diff(row.get("mean_us"), p_us)
            paper = row.get("paper_claim_us", row.get("paper_claim_ns", "N/A"))
            table_row(lines, [
                row.get("phase"), row.get("config"), paper, p_us,
                row.get("mean_us"), row.get("ci95_us"), pd, row.get("match", "—"),
            ])

    # =========================================================================
    # Section I: Register-spill
    # =========================================================================
    section(lines, "I. Register-Spill Instrumentation (128,128)")
    fresh_rows = _csv("raw_logs/register_spill_128x128.csv")
    prior_rows = _csv("raw_logs_prior/register_spill_128x128.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskI` on Jetson Nano\n")
    else:
        row = fresh_rows[-1]  # most recent
        prior = prior_rows[-1] if prior_rows else None
        p_count = prior.get("spill_count", "N/A") if prior else "N/A"
        pd = _pct_diff(row.get("spill_count", "N/A"), p_count)
        table_header(lines, ["Method", "Spill Count (fresh)", "Spill Count (prior)", "%Diff",
                              "Over-sub Ratio", "Hypothesis"])
        table_row(lines, [
            row.get("method", "N/A"), row.get("spill_count", "N/A"), p_count, pd,
            row.get("over_sub_ratio", "N/A"), row.get("hypothesis", "N/A"),
        ])

    # =========================================================================
    # Section J: Energy per-run VGG16
    # =========================================================================
    section(lines, "J. Energy Per-Run Distribution — VGG16")
    fresh_rows = _csv("raw_logs/energy_per_run_vgg16.csv")
    prior_rows = _csv("raw_logs_prior/energy_per_run_vgg16.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskJ` on Jetson Nano\n")
    else:
        import statistics, math
        vals = [float(r["energy_mj"]) for r in fresh_rows
                if r.get("energy_mj") not in (None, "N/A", "")]
        if vals:
            n = len(vals)
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals)
            ci95 = 1.96 * sd / math.sqrt(n)
            pvals = [float(r["energy_mj"]) for r in prior_rows
                     if r.get("energy_mj") not in (None, "N/A", "")]
            prior_mean = statistics.mean(pvals) if pvals else None
            pd = _pct_diff(mean, prior_mean) if prior_mean else "N/A"
            lines.append(f"| Metric | Prior | Fresh | %Diff |\n|:---|---:|---:|---:|")
            lines.append(f"| Mean energy (mJ) | {round(prior_mean,4) if prior_mean else 'N/A'} "
                         f"| {round(mean,4)} | {pd} |")
            lines.append(f"| CI95 (mJ)        | N/A | ±{round(ci95,4)} | N/A |")
            lines.append(f"| n                | {len(pvals)} | {n} | — |")
            lines.append(f"\n> Paper claims 5.5% energy improvement. "
                         f"Compare fused vs baseline rows in the raw CSV.\n")
        else:
            lines.append("> Energy data present but no numeric energy_mj values found.\n")

    # =========================================================================
    # Section K/O: End-to-end inference (Jetson + RPi4)
    # =========================================================================
    section(lines, "K/O. End-to-End Inference (Jetson Nano + Raspberry Pi 4)")
    lines.append("\n> **METHODOLOGY (Constraint #6):** CacheWinograd E2E = layer-time aggregate. "
                 "ORT / TVM / ArmCL = true single-process runs.\n")
    fresh_rows = _csv("summary/e2e_comparison_table.csv")
    prior_rows = _csv("summary_prior/e2e_comparison_table.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskK` (Jetson) and `make taskO` (RPi4)\n")
    else:
        table_header(lines, ["Model", "Platform", "Baseline", "Prior Mean ms", "Fresh Mean ms",
                              "CI95", "%Diff", "Fresh Imp%", "p-value", "Sig", "Match"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows
                          if r.get("model") == row.get("model")
                          and r.get("platform") == row.get("platform")
                          and r.get("baseline") == row.get("baseline")), None)
            p_mean = prior["mean_ms"] if prior else "N/A"
            pd = _pct_diff(row.get("mean_ms"), p_mean)
            table_row(lines, [
                row.get("model"), row.get("platform"), row.get("baseline"),
                p_mean, row.get("mean_ms"), row.get("CI95"),
                pd, row.get("pct_improvement"), row.get("p_value"), row.get("significance"),
                _match(pd),
            ])

    # =========================================================================
    # Section M: RPi4 main microbenchmark
    # =========================================================================
    section(lines, "M. Raspberry Pi 4 Main Microbenchmark (All 7 Configs, m=4)")
    fresh_rows = _csv("summary/rpi4_main_microbench_summary.csv")
    prior_rows = _csv("summary_prior/rpi4_main_microbench_summary.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskM` on Raspberry Pi 4\n")
    else:
        table_header(lines, ["Config", "Prior Baseline ms", "Fresh Baseline ms",
                              "Prior Fused ms", "Fresh Fused ms", "Prior Imp%",
                              "Fresh Imp%", "%Diff Imp", "Paper Imp%", "Match", "p-value"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows if r.get("config") == row.get("config")), None)
            p_bl  = prior["baseline_mean_ms"] if prior else "N/A"
            p_fu  = prior["fused_mean_ms"]    if prior else "N/A"
            p_imp = prior["improvement_pct"]  if prior else "N/A"
            pd = _pct_diff(row.get("improvement_pct"), p_imp)
            table_row(lines, [
                row.get("config"), p_bl, row.get("baseline_mean_ms"),
                p_fu, row.get("fused_mean_ms"), p_imp, row.get("improvement_pct"), pd,
                row.get("paper_improvement_pct", "N/A"), row.get("match", "—"),
                row.get("p_value", "N/A"),
            ])

    # =========================================================================
    # Section N: Adaptive-m selection correctness
    # =========================================================================
    section(lines, "N. Adaptive-m Selection — RPi4 All 7 Configs at m∈{2,4,6}")
    fresh_rows = _csv("summary/rpi4_adaptive_m_verification.csv")
    prior_rows = _csv("summary_prior/rpi4_adaptive_m_verification.csv")
    if not fresh_rows:
        lines.append("> **NOT RUN** — Execute `make taskN` on Raspberry Pi 4\n")
    else:
        table_header(lines, ["Config", "Policy m", "Empirical Best m", "m=2 ms", "m=6 ms",
                              "p(m2 vs m6)", "Match?", "Prior Match?"])
        for row in fresh_rows:
            prior = next((r for r in prior_rows if r.get("config") == row.get("config")), None)
            p_match = prior.get("policy_matches_empirical", "N/A") if prior else "N/A"
            table_row(lines, [
                row.get("config"), row.get("policy_m"), row.get("empirical_best_m"),
                f"{row.get('m2_mean_ms')}±{row.get('m2_ci95_ms')}",
                f"{row.get('m6_mean_ms')}±{row.get('m6_ci95_ms')}",
                row.get("p_value_m2_vs_m6", "N/A"),
                row.get("policy_matches_empirical"), p_match,
            ])

    # =========================================================================
    # Derived tables
    # =========================================================================
    section(lines, "R. R_min Sensitivity Table (Recomputed)")
    fresh_rows = _csv("summary/rmin_sensitivity.csv")
    if not fresh_rows:
        lines.append("> **NOT COMPUTED** — Execute `make derived-tables`\n")
    else:
        table_header(lines, ["R_min", "Config", "Selected Tile", "Jetson Imp%", "RPi4 Imp%"])
        for row in fresh_rows:
            table_row(lines, [row.get("r_min"), row.get("c_in") + "," + row.get("c_out"),
                               row.get("selected_tile"),
                               row.get("jetson_improvement_pct", "N/A"),
                               row.get("rpi4_improvement_pct", "N/A")])

    section(lines, "S. Cross-Platform Comparison Table (Recomputed)")
    fresh_rows = _csv("summary/cross_platform_comparison.csv")
    if not fresh_rows:
        lines.append("> **NOT COMPUTED** — Execute `make derived-tables`\n")
    else:
        table_header(lines, ["Config", "Jetson Baseline ms", "Jetson Fused ms", "Jetson Imp%",
                              "RPi4 Baseline ms", "RPi4 Fused ms", "RPi4 Imp%"])
        for row in fresh_rows:
            table_row(lines, [
                f"({row['c_in']},{row['c_out']})",
                row["jetson_baseline_ms"], row["jetson_fused_ms"], row["jetson_improvement"],
                row["rpi4_baseline_ms"],   row["rpi4_fused_ms"],   row["rpi4_improvement"],
            ])

    section(lines, "T. Roofline Table (Recomputed)")
    fresh_rows = _csv("summary/roofline_table.csv")
    if not fresh_rows:
        lines.append("> **NOT COMPUTED** — Execute `make derived-tables`\n")
    else:
        table_header(lines, ["Config", "MACs", "AI (FLOP/B)", "Ridge (FLOP/B)",
                              "Bounded By", "Measured GFLOPS", "Note"])
        for row in fresh_rows:
            table_row(lines, [
                f"({row['c_in']},{row['c_out']})", row["macs"],
                row["arithmetic_intensity_flop_byte"], row["ridge_point_flop_byte"],
                row["bounded_by"], row["measured_gflops"], row.get("note", "")[:60],
            ])

    section(lines, "U. Fusion DRAM Savings Table (Recomputed)")
    fresh_rows = _csv("summary/fusion_dram_savings.csv")
    if not fresh_rows:
        lines.append("> **NOT COMPUTED** — Execute `make derived-tables`\n")
    else:
        table_header(lines, ["Config", "Tile", "Transform Bytes Eliminated",
                              "Total Non-fused DRAM", "Savings %"])
        for row in fresh_rows:
            table_row(lines, [
                f"({row['c_in']},{row['c_out']})", row["tile_name"],
                row["transform_bytes_eliminated"], row["total_nonfused_dram_bytes"],
                row["dram_savings_pct"],
            ])

    # =========================================================================
    # Missing log summary
    # =========================================================================
    section(lines, "Z. Summary of Missing Logs")
    if missing_logs:
        lines.append(f"> **{len(missing_logs)} REQUIRED log(s) missing:**\n")
        for task, path in missing_logs:
            lines.append(f"> - `{task}`: `{path}`")
        lines.append("")
    else:
        lines.append("> ✓ All required logs present.\n")

    # Write report
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to: {OUT_FILE}")

    # Constraint #7
    if verify_mode and missing_logs:
        print(f"\nERROR: {len(missing_logs)} required log(s) missing. "
              "Cannot certify full reproducibility.")
        for task, path in missing_logs:
            print(f"  MISSING: {path}  (task {task})")
        sys.exit(1)

    return missing_logs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="Exit non-zero if any required log is missing (for make verify-all)")
    args = p.parse_args()
    generate_report(verify_mode=args.verify)

if __name__ == "__main__":
    main()
