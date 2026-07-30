#!/usr/bin/env python3
"""
Task B — Working-set model validation.

Uses perf cache-miss counting (n=100 runs) to measure actual working-set
behaviour vs the paper's WS_base / WS_ext model predictions for configs:
  (16,32), (32,32), (64,64), (128,128)

If perf is unavailable, prints install steps and exits non-zero.

Writes: raw_logs/jetson_ws_validation.csv
        summary/ws_model_validation_summary.csv

MUST be run on the Jetson Nano (Linux with perf_event support).
"""
import os, sys, re, subprocess, csv, datetime, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler

CONFIGS = [
    {"c_in": 16,  "c_out": 32},
    {"c_in": 32,  "c_out": 32},
    {"c_in": 64,  "c_out": 64},
    {"c_in": 128, "c_out": 128},
]

# Paper's claimed values for each config (WS_base, WS_ext, Err_base, Err_ext)
# These are the model's predicted working sets (bytes), not latencies.
PAPER_WS = {
    (16, 32):   {"ws_base": None, "ws_ext": None},  # fill from paper
    (32, 32):   {"ws_base": None, "ws_ext": None},
    (64, 64):   {"ws_base": None, "ws_ext": None},
    (128, 128): {"ws_base": None, "ws_ext": None},
}


def check_perf():
    try:
        r = subprocess.run(["perf", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if r.returncode == 0:
            return True
    except (FileNotFoundError, OSError):
        pass
    print("\nBLOCKER [Task B]: `perf` not found.")
    print("Install steps:")
    print("  sudo apt-get install -y linux-tools-$(uname -r) linux-tools-generic")
    print("  # If kernel version mismatch:")
    print("  sudo apt-get install -y linux-tools-4.9-tegra  # for JetPack 4.x")
    print("  echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid")
    sys.exit(1)


def run_perf_cache_miss(binary, events="cache-misses,cache-references,instructions,cycles", n=100):
    """Run binary under perf stat n times, return list of per-run counter dicts."""
    results = []
    for i in range(n):
        cmd = ["perf", "stat", "-e", events, "--", binary]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
            perf_out = (r.stderr or "") + (r.stdout or "")
            row = {}
            for event in events.split(","):
                m = re.search(r'([\d,]+)\s+' + re.escape(event), perf_out)
                if m:
                    row[event] = int(m.group(1).replace(",", ""))
            if row:
                results.append(row)
        except Exception as e:
            print(f"  perf run {i} failed: {e}")
    return results


def build_probe_binary(c_in, c_out, tile_dim, tmpdir):
    """Compile a small C driver that exercises the fused kernel to provoke cache behaviour."""
    import tempfile
    kernel_src = os.path.join(os.path.dirname(__file__), "..", "fused_winograd.c")
    driver = os.path.join(tmpdir, f"ws_probe_{c_in}_{c_out}.c")
    binary = os.path.join(tmpdir, f"ws_probe_{c_in}_{c_out}")
    code = f"""
#include <stdlib.h>
#include <string.h>
extern void fused_winograd_f23(const float*, const float*, float*, int, int);
int main() {{
    int cin={c_in}, cout={c_out}, td={tile_dim};
    float *inp = (float*)calloc(cin*td*td, sizeof(float));
    float *U   = (float*)calloc(cout*cin*td*td, sizeof(float));
    float *out = (float*)calloc(cout*td*td, sizeof(float));
    for(int i=0;i<50;i++) fused_winograd_f23(inp, U, out, cin, cout);
    free(inp); free(U); free(out);
    return 0;
}}
"""
    with open(driver, "w") as f:
        f.write(code)
    import platform as plat
    flags = ["-O2", "-g"]
    mach = plat.machine().lower()
    if mach in ("arm64", "aarch64"):
        flags += ["-march=armv8-a"]
    elif mach.startswith("armv7"):
        flags += ["-mfpu=neon"]
    if not os.path.exists(kernel_src):
        print(f"  WARNING: {kernel_src} not found — using stub driver.")
        code2 = code.replace(
            'extern void fused_winograd_f23(const float*, const float*, float*, int, int);',
            'void fused_winograd_f23(const float*a,const float*b,float*c,int d,int e){}'
        )
        with open(driver, "w") as f:
            f.write(code2)
        cmd = ["gcc"] + flags + [driver, "-o", binary, "-lm"]
    else:
        cmd = ["gcc"] + flags + [driver, kernel_src, "-o", binary, "-lm"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        cmd[0] = "clang"
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        print(f"  Compilation failed: {r.stderr[:200]}")
        return None
    return binary


def run_ws_validation(n=100, out_path="raw_logs/jetson_ws_validation.csv",
                      summary_path="summary/ws_model_validation_summary.csv"):
    check_perf()
    import tempfile
    ts = datetime.datetime.now().isoformat()
    tiler = CacheAdaptiveAutotiler()
    raw_rows, summary_rows = [], []

    print("[B] Working-set model validation")
    with tempfile.TemporaryDirectory() as tmpdir:
        for cfg in CONFIGS:
            c_in, c_out = cfg["c_in"], cfg["c_out"]
            decision = tiler.select_best_tile(c_in, c_out)
            td = decision["selected_tile"]["tile"]
            ws_model = tiler.compute_working_set(td, c_in, c_out)
            print(f"\n  Config ({c_in},{c_out}): tile={td}, WS_model={ws_model}B")

            binary = build_probe_binary(c_in, c_out, td, tmpdir)
            if binary is None:
                print(f"  SKIPPED ({c_in},{c_out}): compilation failed")
                continue

            perf_data = run_perf_cache_miss(binary, n=n)
            if not perf_data:
                print(f"  SKIPPED ({c_in},{c_out}): perf returned no data")
                continue

            for run_id, row in enumerate(perf_data):
                raw_rows.append({
                    "timestamp": ts, "c_in": c_in, "c_out": c_out,
                    "tile_dim": td, "ws_model_bytes": ws_model,
                    "run_id": run_id,
                    **row,
                })

            misses = [r.get("cache-misses", 0) for r in perf_data]
            refs   = [r.get("cache-references", 0) for r in perf_data]
            mean_misses = float(np.mean(misses)) if misses else 0
            mean_refs   = float(np.mean(refs)) if refs else 0
            miss_rate   = mean_misses / max(mean_refs, 1) * 100

            paper_ws = PAPER_WS.get((c_in, c_out), {})
            summary_rows.append({
                "config": f"({c_in},{c_out})", "c_in": c_in, "c_out": c_out,
                "tile_dim": td,
                "ws_model_bytes": ws_model,
                "n": len(perf_data),
                "mean_cache_misses": round(mean_misses, 1),
                "mean_cache_refs": round(mean_refs, 1),
                "cache_miss_rate_pct": round(miss_rate, 3),
                "paper_ws_base": paper_ws.get("ws_base", "N/A"),
                "paper_ws_ext":  paper_ws.get("ws_ext", "N/A"),
            })
            print(f"  -> miss_rate={miss_rate:.2f}%, mean_misses={mean_misses:.0f}")

    # Write raw
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if raw_rows:
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()), extrasaction="ignore")
            if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
                w.writeheader()
            w.writerows(raw_rows)
        print(f"\n[B] Raw written: {out_path} ({len(raw_rows)} rows)")

    # Write summary
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[B] Summary written: {summary_path}")

    if not raw_rows:
        print("[B] ERROR: No data collected. Check perf installation.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out", default="raw_logs/jetson_ws_validation.csv")
    parser.add_argument("--summary", default="summary/ws_model_validation_summary.csv")
    args = parser.parse_args()
    run_ws_validation(n=args.runs, out_path=args.out, summary_path=args.summary)

if __name__ == "__main__":
    main()
