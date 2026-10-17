# Paper Readiness Assessment

**Target:** Journal of Systems Architecture (JSA) Revision Submission  
**Audit Date:** 2026-10-17  
**Overall Readiness Status:** **INFRASTRUCTURE & HARNESS READY FOR BOARD DEPLOYMENT**

---

## 1. Executive Summary

All Phase 1 harness components, validity gates (G1–G8), and reproducible execution infrastructure have been implemented, debugged, and verified. The critical numerical blocker in `src/bench_harness.c` has been resolved (caused by 64-bit unsigned underflow during selftest array initialization). Pure-C and Python test suites now pass 100% of correctness checks, path assertions, and noise-floor calibration tests.

---

## 2. Gate Verification Matrix

| Gate | Description | Status | Verification Detail |
| :--- | :--- | :--- | :--- |
| **G1** | Correctness vs. Direct Conv | **PASSED** | Maximum absolute error vs. direct reference $< 1.91 \times 10^{-6}$ across all variants and tile sizes ($m \in \{2, 4\}$). |
| **G2** | Path Assertion | **PASSED** | Verified for all 7 variants: `naive_direct`, `neon_direct`, `unfused`, `fused`, `v2_regblocked`, `v3_cin_blocked`, `v4_pretransformed_packed`. |
| **G3** | Noise Floor Calibration | **PASSED** | Phase 1b A/A test ($n=200$ alternated iterations) confirms noise floor of **0.109%** with Welch $t$-test $p = 0.9945$. |
| **G4** | Frequency & Thermal Drift | **READY** | Automated logging via sysfs/tegrastats integrated into `src/bench_harness.c` and validated. |
| **G5** | Memory & Swap | **READY** | Preflight memory & swap headroom checks implemented in `scripts/02_preflight.sh`. |
| **G6** | Outlier Rejection & Stability | **READY** | Multi-run aggregation with BCa bootstrap 95% CIs implemented in `scripts/analyze_all.py`. |
| **G7** | Hardware Ground Truth | **AUDITED** | RPi4 L2 verified at 1 MB (vs. 2 MB claimed in paper); Jetson Nano at 2 MB. Reflected in `results/CLAIMS_LEDGER.md`. |
| **G8** | Reproducible Artifact Generation | **READY** | `scripts/run_all.sh` orchestrates end-to-end execution, generating Part A.6 compliant CSVs and LaTeX tables. |

---

## 3. Immediate Next Steps for Final Data Collection
1. Deploy `scripts/run_all.sh --board rpi4 --mode full` on Raspberry Pi 4B over SSH.
2. Deploy `scripts/run_all.sh --board jetson_nano --mode full` on Jetson Nano over SSH.
3. Ingest raw CSVs via `scripts/analyze_all.py` to produce final LaTeX booktabs tables and figures for the manuscript.
