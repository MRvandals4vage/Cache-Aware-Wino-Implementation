#!/usr/bin/env python3
"""
Derived tables — pure recomputation from task C (Jetson) and task M (RPi4) CSV data.
No new hardware runs. Produces:
  summary/rmin_sensitivity.csv
  summary/cross_platform_comparison.csv
  summary/roofline_table.csv
  summary/fusion_dram_savings.csv
"""
import os, sys, csv, argparse, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler

# Jetson Nano hardware constants (paper Table 1)
JN_PEAK_GFLOPS = 45.8    # GFLOPS/s (FP32, NEON)
JN_BW_GBS      = 25.6    # GB/s peak DRAM bandwidth
JN_GFLOPS_INT8 = 236.0   # INT8 (not used for FP32 Winograd)

# R_min sensitivity grid
RMIN_VALS = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]

CONFIGS = [
    {"c_in": 16,  "c_out": 32},
    {"c_in": 32,  "c_out": 16},
    {"c_in": 32,  "c_out": 32},
    {"c_in": 32,  "c_out": 64},
    {"c_in": 64,  "c_out": 32},
    {"c_in": 64,  "c_out": 64},
    {"c_in": 128, "c_out": 128},
]

H, W = 56, 56  # spatial dims per paper protocol


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path}")


def compute_rmin_sensitivity(jetson_rows, rpi4_rows, out_dir):
    """
    For each R_min in {0.10,...,0.80} re-run the tile-selection logic to see which
    configs change their selected tile and what the resulting improvement would be.
    Uses the latency data already measured in tasks C/M.
    """
    tiler = CacheAdaptiveAutotiler()
    rows = []
    for rmin in RMIN_VALS:
        for cfg in CONFIGS:
            c_in, c_out = cfg["c_in"], cfg["c_out"]
            # Re-run selection with this alpha (R_min maps to alpha = 1 - rmin)
            from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
            t2 = CacheAdaptiveAutotiler()
            t2.alpha = 1.0 - rmin  # aggressive = lower rmin -> less conservative
            decision = t2.select_best_tile(c_in, c_out)
            tile = decision["selected_tile"]["name"]

            # Pull improvement from measured data if available
            j_row = next((r for r in jetson_rows if r["c_in"] == str(c_in)
                          and r["c_out"] == str(c_out)), None)
            jetson_imp = j_row["improvement_pct"] if j_row else "N/A"
            r_row = next((r for r in rpi4_rows if r["c_in"] == str(c_in)
                          and r["c_out"] == str(c_out)), None)
            rpi4_imp = r_row["improvement_pct"] if r_row else "N/A"

            rows.append({
                "r_min": rmin, "c_in": c_in, "c_out": c_out,
                "selected_tile": tile,
                "jetson_improvement_pct": jetson_imp,
                "rpi4_improvement_pct": rpi4_imp,
            })
    write_csv(os.path.join(out_dir, "rmin_sensitivity.csv"), rows)


def compute_cross_platform(jetson_rows, rpi4_rows, out_dir):
    rows = []
    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        j = next((r for r in jetson_rows if r["c_in"] == str(c_in)
                  and r["c_out"] == str(c_out)), None)
        r = next((r for r in rpi4_rows if r["c_in"] == str(c_in)
                  and r["c_out"] == str(c_out)), None)
        rows.append({
            "c_in": c_in, "c_out": c_out,
            "jetson_baseline_ms": j["baseline_mean_ms"] if j else "NOT_RUN",
            "jetson_fused_ms":    j["fused_mean_ms"]    if j else "NOT_RUN",
            "jetson_improvement": j["improvement_pct"]  if j else "NOT_RUN",
            "rpi4_baseline_ms":   r["baseline_mean_ms"] if r else "NOT_RUN",
            "rpi4_fused_ms":      r["fused_mean_ms"]    if r else "NOT_RUN",
            "rpi4_improvement":   r["improvement_pct"]  if r else "NOT_RUN",
        })
    write_csv(os.path.join(out_dir, "cross_platform_comparison.csv"), rows)


def compute_roofline(jetson_rows, out_dir):
    """
    Recompute arithmetic intensity from measured latencies.
    AI = MACs / bytes_transferred
    Compute-bound if AI > JN_PEAK_GFLOPS / JN_BW_GBS (ridge point).
    """
    ridge = JN_PEAK_GFLOPS / JN_BW_GBS  # FLOP/byte at ridge point
    rows = []

    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        macs = c_in * c_out * 3 * 3 * H * W  # one conv2d image
        # bytes transferred: input + kernel + output (float32)
        input_bytes  = c_in  * H * W * 4
        kernel_bytes = c_in  * c_out * 3 * 3 * 4
        output_bytes = c_out * H * W * 4
        total_bytes  = input_bytes + kernel_bytes + output_bytes
        ai = macs / total_bytes  # FLOP/byte

        j = next((r for r in jetson_rows if r["c_in"] == str(c_in)
                  and r["c_out"] == str(c_out)), None)
        if j:
            fused_ms = float(j["fused_mean_ms"])
            fused_gflops = (macs / 1e9) / (fused_ms / 1000.0)
            bounded = "COMPUTE" if ai > ridge else "MEMORY"
        else:
            fused_ms = "NOT_RUN"
            fused_gflops = "NOT_RUN"
            bounded = "UNKNOWN"

        rows.append({
            "c_in": c_in, "c_out": c_out,
            "macs": macs, "total_bytes": total_bytes,
            "arithmetic_intensity_flop_byte": round(ai, 4),
            "ridge_point_flop_byte": round(ridge, 4),
            "jn_peak_gflops": JN_PEAK_GFLOPS,
            "jn_peak_bw_gbs": JN_BW_GBS,
            "bounded_by": bounded,
            "fused_mean_ms": fused_ms,
            "measured_gflops": round(fused_gflops, 4) if isinstance(fused_gflops, float) else fused_gflops,
            "note": ("JN peak-compute and BW are board-level constants from paper Table 1. "
                     "Verify against tegrastats if board differs."),
        })
    write_csv(os.path.join(out_dir, "roofline_table.csv"), rows)


def compute_fusion_dram_savings(out_dir):
    """
    Pure formula: fusion eliminates intermediate DRAM stores.
    For F(m,r): non-fused stores (m+r-1)^2 * C_in tiles + reads them back.
    Fused never writes them.
    """
    rows = []
    for cfg in CONFIGS:
        c_in, c_out = cfg["c_in"], cfg["c_out"]
        tiler = CacheAdaptiveAutotiler()
        dec = tiler.select_best_tile(c_in, c_out)
        td = dec["selected_tile"]["tile"]
        m_val = td - 2  # r=3

        # Non-fused: store transformed input tiles (td^2 * c_in), then read back
        transform_bytes = td * td * c_in * 4  # float32
        saved_bytes_per_pass = transform_bytes * 2  # write + read eliminated by fusion
        # Normalised as % of total non-fused memory traffic
        total_nonfused_bytes = (c_in * H * W * 4 +          # read input
                                transform_bytes +             # write transform
                                transform_bytes +             # read transform back
                                c_out * H * W * 4)           # write output
        savings_pct = saved_bytes_per_pass / total_nonfused_bytes * 100.0

        rows.append({
            "c_in": c_in, "c_out": c_out,
            "tile_name": dec["selected_tile"]["name"],
            "tile_dim": td, "m": m_val,
            "transform_bytes_eliminated": saved_bytes_per_pass,
            "total_nonfused_dram_bytes": total_nonfused_bytes,
            "dram_savings_pct": round(savings_pct, 2),
        })
    write_csv(os.path.join(out_dir, "fusion_dram_savings.csv"), rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jetson-csv",  default="summary/jetson_main_microbench_summary.csv")
    p.add_argument("--rpi4-csv",    default="summary/rpi4_main_microbench_summary.csv")
    p.add_argument("--out-dir",     default="summary")
    args = p.parse_args()

    print("[Derived Tables] Loading measured data ...")
    jetson = read_csv(args.jetson_csv)
    rpi4   = read_csv(args.rpi4_csv)
    if not jetson:
        print(f"  WARNING: {args.jetson_csv} missing — Jetson tables will show NOT_RUN")
    if not rpi4:
        print(f"  WARNING: {args.rpi4_csv} missing — RPi4 tables will show NOT_RUN")

    print("[Derived Tables] R_min sensitivity ...")
    compute_rmin_sensitivity(jetson, rpi4, args.out_dir)

    print("[Derived Tables] Cross-platform comparison ...")
    compute_cross_platform(jetson, rpi4, args.out_dir)

    print("[Derived Tables] Roofline table ...")
    compute_roofline(jetson, args.out_dir)

    print("[Derived Tables] Fusion DRAM savings ...")
    compute_fusion_dram_savings(args.out_dir)

    print("[Derived Tables] Done.")

if __name__ == "__main__":
    main()
