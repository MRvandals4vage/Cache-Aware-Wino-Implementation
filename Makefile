# ============================================================================
# CacheWinograd — Top-level Makefile
# Full experiment suite: tasks A-O (Jetson Nano + Raspberry Pi 4) + derived.
# Run `make verify-all` to execute every task and produce reproducibility report.
# Constraint: NEVER fabricate or estimate — run only on the named device.
# ============================================================================

PYTHON   ?= python3
ROOT_DIR := $(shell pwd)
export PYTHONPATH := $(ROOT_DIR):$(PYTHONPATH)

RAW_LOGS  := raw_logs
SUMMARY   := summary
ARTIFACTS := artifacts
PLOTS     := $(ARTIFACTS)/plots

.PHONY: help dirs doctor preflight preflight-backup clean-safe reset smoke-test quick-test \
        taskA taskB taskC taskD taskE taskF taskG taskH taskI taskJ taskK taskL \
        taskM taskN taskO taskP \
        derived-tables plots report verify-all \
        verify-jetson verify-rpi4 e2e clean-logs \
        t1 t2 t3 t4 t5 t6 t7 t8

help:
	@echo "CacheWinograd Full Experiment Suite"
	@echo ""
	@echo "PRE-FLIGHT & ENVIRONMENT:"
	@echo "  make doctor            — Health check environment, dependencies & tools (Task 14)"
	@echo "  make preflight-backup  — Copy raw_logs/ -> raw_logs_prior/, summary/ -> summary_prior/"
	@echo "  make clean-safe        — Clean build artifacts, keeping raw_logs and summary (Task 5)"
	@echo "  make reset             — Reset build & compile C extension from scratch (Task 5)"
	@echo "  make smoke-test        — n=5 sanity check (Jetson Nano)"
	@echo ""
	@echo "JETSON NANO TASKS:"
	@echo "  make taskA  — Cache-probe sanity (A)"
	@echo "  make taskB  — Working-set model validation (B)"
	@echo "  make taskC  — Main microbenchmark suite, all 7 configs (C)"
	@echo "  make taskD  — TVM/AutoTVM + ArmCL statistical rigor (D)"
	@echo "  make taskE  — Multi-core/fusion ablation (E)"
	@echo "  make taskF  — L2 hit rate + MACs/Joule (F)"
	@echo "  make taskG  — Fallback-guard held-out configs (G)"
	@echo "  make taskH  — Scheduling overhead direct timing (H)"
	@echo "  make taskI  — Register-spill instrumentation (I)"
	@echo "  make taskJ  — Energy per-run distribution VGG16 (J)"
	@echo "  make taskK  — End-to-end inference Jetson Nano (K)"
	@echo "  make taskL  — Combined fusion+parallelism policy (L, optional)"
	@echo ""
	@echo "RASPBERRY PI 4 TASKS:"
	@echo "  make taskM  — Main microbenchmark suite, all 7 configs (M)"
	@echo "  make taskN  — Adaptive-m sweep m in {2,4,6} (N)"
	@echo "  make taskO  — End-to-end inference RPi4 (O)"
	@echo ""
	@echo "DERIVED + CLOSE-OUT (run on any machine after A-O CSVs collected):"
	@echo "  make derived-tables  — R_min sensitivity, cross-platform, roofline"
	@echo "  make plots           — Regenerate paper figures from fresh CSV"
	@echo "  make report          — Generate reproducibility_report.md"
	@echo "  make verify-all      — Run everything + report + check (non-zero if logs missing)"
	@echo ""
	@echo "JETSON SUITE:  make verify-jetson"
	@echo "RPI4 SUITE:    make verify-rpi4"

dirs:
	@mkdir -p $(RAW_LOGS) $(SUMMARY) build tmp results $(ARTIFACTS)/raw $(ARTIFACTS)/processed $(PLOTS) \
	          $(ARTIFACTS)/logs logs raw_logs_prior summary_prior

doctor: dirs
	$(PYTHON) scripts/doctor.py

clean-safe:
	rm -rf build tmp results *.so __pycache__ benchmarks/__pycache__ src/__pycache__ scripts/__pycache__ tools/__pycache__
	@echo "Cleaned build artifacts, temporary files, and compiled extension. Preserved raw_logs and summary."

reset: clean-safe dirs
	@echo "Resetting environment & rebuilding C extension..."
	$(PYTHON) -c "from src.benchmark_utils import ensure_c_extension; ensure_c_extension()"
	$(PYTHON) scripts/doctor.py --fix
	@echo "Reset complete."

# ============================================================================
# Pre-flight
# ============================================================================
preflight-backup: dirs
	bash scripts/preflight_backup.sh

smoke-test: dirs
	@echo "=== Smoke Test (n=5, sanity only — do NOT use these numbers) ==="
	$(PYTHON) benchmarks/cache_probe_sanity.py
	$(PYTHON) benchmarks/jetson_main_microbench.py --runs 5 --warmup 2 --height 14 --width 14
	$(PYTHON) benchmarks/scheduling_overhead.py --runs 5
	$(PYTHON) scripts/generate_paper_figures.py \
		--microbench-csv $(ARTIFACTS)/processed/paper_table_microbench.csv \
		--e2e-csv $(SUMMARY)/e2e_comparison_table.csv \
		--plot-dir $(PLOTS) || true
	$(PYTHON) scripts/generate_reproducibility_report.py || true
	@echo "=== Smoke Test PASSED (errors above are expected on empty data) ==="

# ============================================================================
# quick-test: fast local dev sanity — reduced params, completes in ~2 min
# Use this for iterative development; taskC/taskE are intended for Jetson Nano.
# ============================================================================
quick-test: dirs
	@echo "=== Quick Test (n=50, dev only — do NOT use these numbers) ==="
	$(PYTHON) benchmarks/jetson_main_microbench.py \
		--runs 50 --warmup 5 --height 14 --width 14 \
		--raw $(RAW_LOGS)/quicktest_main_microbench.csv \
		--summary $(SUMMARY)/quicktest_main_microbench_summary.csv
	$(PYTHON) benchmarks/jetson_ablation.py \
		--runs 50 --warmup 5 --height 14 --width 14
	@echo "=== Quick Test DONE ==="

# ============================================================================
# Task A: Cache-probe sanity
# ============================================================================
taskA: dirs
	$(PYTHON) benchmarks/cache_probe_sanity.py \
		--out $(RAW_LOGS)/jetson_cacheprobe.csv

# ============================================================================
# Task B: Working-set model validation
# ============================================================================
taskB: dirs
	$(PYTHON) benchmarks/ws_model_validation.py \
		--runs 100 \
		--out $(RAW_LOGS)/jetson_ws_validation.csv \
		--summary $(SUMMARY)/ws_model_validation_summary.csv

# ============================================================================
# Task C: Main microbenchmark suite — ALL 7 configs, Jetson Nano
# ============================================================================
taskC: dirs
	$(PYTHON) benchmarks/jetson_main_microbench.py \
		--runs 1000 --warmup 20 --height 56 --width 56 \
		--raw $(RAW_LOGS)/jetson_main_microbench.csv \
		--summary $(SUMMARY)/jetson_main_microbench_summary.csv

# ============================================================================
# Task D: TVM/AutoTVM + ArmCL statistical rigor
# ============================================================================
taskD: dirs
	$(PYTHON) benchmarks/tvm_armcl_rigor.py --runs 1000 --warmup 20

# ============================================================================
# Task E: Multi-core / fusion x threading ablation
# ============================================================================
taskE: dirs
	$(PYTHON) benchmarks/jetson_ablation.py \
		--runs 1000 --warmup 20 --height 56 --width 56

# ============================================================================
# Task F: L2 hit rate + MACs/Joule
# ============================================================================
taskF: dirs
	$(PYTHON) benchmarks/jetson_l2_macsj.py \
		--runs 100 --warmup 20 --height 56 --width 56

# ============================================================================
# Task G: Fallback-guard held-out configs
# ============================================================================
taskG: dirs
	$(PYTHON) benchmarks/jetson_fallback_heldout.py \
		--runs 1000 --warmup 20 --height 56 --width 56

# ============================================================================
# Task H: Scheduling-overhead direct timing
# ============================================================================
taskH: dirs
	$(PYTHON) benchmarks/scheduling_overhead.py --runs 1000

# ============================================================================
# Task I: Register-spill instrumentation
# ============================================================================
taskI: dirs
	$(PYTHON) benchmarks/register_spill_instrument.py --c-in 128 --c-out 128

# ============================================================================
# Task J: Energy per-run distribution (VGG16, Jetson Nano)
# ============================================================================
taskJ: dirs
	$(PYTHON) benchmarks/energy_per_run.py --runs 50 --warmup 10 --model vgg16

# ============================================================================
# Task K: End-to-end inference — Jetson Nano
# ============================================================================
taskK: dirs
	$(PYTHON) benchmarks/e2e_benchmark.py \
		--models all --runs 50 --warmup 10 --platform jetson_nano

# ============================================================================
# Task L: Combined fusion+parallelism policy (optional)
# ============================================================================
taskL: dirs
	$(PYTHON) benchmarks/combined_policy.py \
		--runs 50 --warmup 10 --height 56 --width 56

# ============================================================================
# Task M: RPi4 main microbenchmark suite — ALL 7 configs, m=4
# ============================================================================
taskM: dirs
	$(PYTHON) benchmarks/rpi4_main_microbench.py \
		--runs 30 --warmup 10 \
		--raw $(RAW_LOGS)/rpi4_main_microbench.csv \
		--summary $(SUMMARY)/rpi4_main_microbench_summary.csv

# ============================================================================
# Task N: RPi4 adaptive-m sweep (m=2 and m=6; m=4 data from task M)
# ============================================================================
taskN: dirs
	$(PYTHON) benchmarks/rpi4_adaptive_m_sweep.py --runs 30 --warmup 10

# ============================================================================
# Task O: End-to-end inference — Raspberry Pi 4
# ============================================================================
taskO: dirs
	$(PYTHON) benchmarks/e2e_benchmark.py \
		--models all --runs 50 --warmup 10 --platform rpi4

# ============================================================================
# Task P: Apple Silicon re-check (optional)
# ============================================================================
taskP: dirs
	$(PYTHON) benchmarks/e2e_benchmark.py \
		--models all --runs 50 --warmup 10 --platform macos_apple_silicon

# ============================================================================
# Derived tables (pure recomputation, no new hardware runs)
# ============================================================================
derived-tables: dirs
	$(PYTHON) scripts/compute_derived_tables.py \
		--jetson-csv $(SUMMARY)/jetson_main_microbench_summary.csv \
		--rpi4-csv   $(SUMMARY)/rpi4_main_microbench_summary.csv \
		--out-dir    $(SUMMARY)

# ============================================================================
# Plots (T1 — regenerate paper figures)
# ============================================================================
t1 plots: dirs
	$(PYTHON) scripts/generate_paper_figures.py \
		--microbench-csv $(ARTIFACTS)/processed/paper_table_microbench.csv \
		--e2e-csv $(SUMMARY)/e2e_comparison_table.csv \
		--plot-dir $(PLOTS)

# ============================================================================
# Report
# ============================================================================
report: dirs
	$(PYTHON) scripts/generate_reproducibility_report.py

report-verify: dirs
	$(PYTHON) scripts/generate_reproducibility_report.py --verify

# ============================================================================
# Composite suite targets
# ============================================================================
verify-jetson: dirs taskA taskB taskC taskD taskE taskF taskG taskH taskI taskJ taskK
	@echo "Jetson Nano verification complete."

verify-rpi4: dirs taskM taskN taskO
	@echo "Raspberry Pi 4 verification complete."

e2e: dirs taskK taskO

# ============================================================================
# verify-all: full run + derived + report + check.
# Non-zero exit if any REQUIRED raw log is missing.
# ============================================================================
verify-all: dirs \
    taskA taskB taskC taskD taskE taskF taskG taskH taskI taskJ taskK \
    taskM taskN taskO \
    derived-tables plots report-verify
	@echo ""
	@echo "============================================"
	@echo "  verify-all COMPLETE"
	@echo "  Report: reproducibility_report.md"
	@echo "============================================"

# ============================================================================
# Legacy aliases (old t2-t8 targets)
# ============================================================================
t2: taskC
t3: taskD
t4: taskN
t5 e2e_legacy: taskK taskO
t6: taskI
t7: taskJ
t8: taskL

# ============================================================================
# Cleanup
# ============================================================================
clean-logs:
	rm -rf $(RAW_LOGS) $(SUMMARY)
	@echo "Cleaned raw_logs/ and summary/"
