# Session Checkpoints

This file tracks the operational progress across sessions and tasks according to the project specifications.

---

## Completed Tasks

### Setup & Preregistration
- [x] Pre-registration document locked: `results/PREREGISTRATION.md`
- [x] Claims ledger initialized: `results/CLAIMS_LEDGER.md`
- [x] Formula and provenance audits complete: `results/FORMULA_AUDIT.md`, `results/PROVENANCE.md`
- [x] Board configuration templates verified: `configs/rpi4.yaml` (1 MB L2 corrected), `configs/jetson_nano.yaml` (2 MB L2)
- [x] Blockers and operational scope tracked: `results/BLOCKERS.md`

### Harness & Validity Gates (G1–G8)
- [x] Pure-C high-precision benchmark harness implemented: `src/bench_harness.c`
  - Variants: `naive_direct`, `neon_direct`, `unfused`, `fused` (v1), `v2_regblocked`, `v3_cin_blocked`, `v4_pretransformed_packed`
  - G1 (Correctness): Verified against reference direct convolution across $m \in \{2, 4\}$ ($< 1.91 \times 10^{-6}$ max abs error).
  - G2 (Path assertion): Strict assertion that executed path matches requested variant.
  - G3 (Noise floor): $n=200$ alternated A/A null test measured noise floor of 0.1094% ($p = 0.9945$).
- [x] Automated Phase 1 selftest suite passing: `scripts/03_selftest.sh`
- [x] Single-cell runner with atomic append and `.done` marker: `scripts/run_cell.sh`
- [x] Master benchmark orchestrator with `--board`, `--mode`, `--tiers`: `scripts/run_all.sh`
- [x] Statistical analysis and LaTeX booktabs table generator: `scripts/analyze_all.py`, `scripts/analyze_all.sh`

---

## Active Status & Next Execution Step
- Local harness validation complete on macOS workstation.
- Awaiting physical board access credentials (SSH host/alias or local native terminal on RPi4/Jetson Nano) to execute `scripts/run_all.sh` directly on physical aarch64 hardware.
