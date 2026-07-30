# ============================================================================
# CacheWinograd — Top-level Makefile
# Run `make verify-all` to execute every benchmark task and produce
# the reproducibility report.
# ============================================================================

PYTHON   ?= python3
ROOT_DIR := $(shell pwd)
export PYTHONPATH := $(ROOT_DIR):$(PYTHONPATH)

# Directories
RAW_LOGS  := raw_logs
SUMMARY   := summary
ARTIFACTS := artifacts
PLOTS     := $(ARTIFACTS)/plots

.PHONY: help dirs plots smoke-test report verify-all \
        t1 t2 t3 t4 t5 t6 t7 t8 \
        verify-jetson verify-rpi4 e2e clean-logs

help:
	@echo "CacheWinograd Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make verify-all     — Run ALL tasks and produce reproducibility_report.md"
	@echo "  make smoke-test     — Quick local sanity check (n=5)"
	@echo "  make plots          — Regenerate figures from CSV (T1)"
	@echo "  make report         — Generate reproducibility_report.md from existing data"
	@echo "  make verify-jetson  — Run T2, T3, T6, T7 on Jetson Nano"
	@echo "  make verify-rpi4    — Run T4 on Raspberry Pi 4"
	@echo "  make e2e            — Run T5 end-to-end on current platform"
	@echo "  make t1..t8         — Run individual tasks"
	@echo "  make clean-logs     — Remove generated raw_logs/ and summary/"
	@echo ""

dirs:
	@mkdir -p $(RAW_LOGS) $(SUMMARY) $(ARTIFACTS)/raw $(ARTIFACTS)/processed $(PLOTS)

# ============================================================================
# T1: Plotting scripts
# ============================================================================
t1 plots: dirs
	$(PYTHON) scripts/generate_paper_figures.py \
		--microbench-csv $(ARTIFACTS)/processed/paper_table_microbench.csv \
		--e2e-csv $(SUMMARY)/e2e_comparison_table.csv \
		--plot-dir $(PLOTS)

# ============================================================================
# T2: Jetson Nano (128,128) re-verification
# ============================================================================
t2: dirs
	$(PYTHON) benchmarks/jetson_128x128_reverify.py --runs 1000 --warmup 20

# ============================================================================
# T3: TVM/ArmCL statistical rigor
# ============================================================================
t3: dirs
	$(PYTHON) benchmarks/tvm_armcl_rigor.py --runs 1000 --warmup 20

# ============================================================================
# T4: RPi4 adaptive-m sweep
# ============================================================================
t4: dirs
	$(PYTHON) benchmarks/rpi4_adaptive_m_sweep.py --runs 30 --warmup 10

# ============================================================================
# T5: End-to-end benchmarking
# ============================================================================
t5 e2e: dirs
	$(PYTHON) benchmarks/e2e_benchmark.py --models all --runs 50 --warmup 10

# ============================================================================
# T6: Register-spill instrumentation
# ============================================================================
t6: dirs
	$(PYTHON) benchmarks/register_spill_instrument.py --c-in 128 --c-out 128

# ============================================================================
# T7: Energy per-run distribution
# ============================================================================
t7: dirs
	$(PYTHON) benchmarks/energy_per_run.py --runs 50 --warmup 10

# ============================================================================
# T8: Combined fusion+parallelism policy
# ============================================================================
t8: dirs
	$(PYTHON) benchmarks/combined_policy.py --runs 50 --warmup 10 --height 56 --width 56

# ============================================================================
# Reproducibility Report
# ============================================================================
report: dirs
	$(PYTHON) scripts/generate_reproducibility_report.py

# ============================================================================
# Composite Targets
# ============================================================================
verify-jetson: dirs t2 t3 t6 t7
	@echo "Jetson Nano verification complete."

verify-rpi4: dirs t4
	@echo "Raspberry Pi 4 verification complete."

# verify-all: run everything, then generate report.
# Fail loudly (non-zero exit) if any required raw log is missing.
verify-all: dirs t1 t2 t3 t4 t5 t6 t7 t8 report
	@echo ""
	@echo "============================================"
	@echo "  verify-all COMPLETE"
	@echo "  Report: reproducibility_report.md"
	@echo "============================================"

# ============================================================================
# Smoke Test (quick local run, n=5, for sanity checking)
# ============================================================================
smoke-test: dirs
	@echo "=== Smoke Test: T1 (plots) ==="
	$(PYTHON) scripts/generate_paper_figures.py \
		--microbench-csv $(ARTIFACTS)/processed/paper_table_microbench.csv \
		--e2e-csv $(SUMMARY)/e2e_comparison_table.csv \
		--plot-dir $(PLOTS)
	@echo ""
	@echo "=== Smoke Test: T2 (reverify, n=5) ==="
	$(PYTHON) benchmarks/jetson_128x128_reverify.py --runs 5 --warmup 2 --height 14 --width 14
	@echo ""
	@echo "=== Smoke Test: T5 (e2e, n=3) ==="
	$(PYTHON) benchmarks/e2e_benchmark.py --models resnet18 --runs 3 --warmup 2
	@echo ""
	@echo "=== Smoke Test: T6 (register spill) ==="
	$(PYTHON) benchmarks/register_spill_instrument.py
	@echo ""
	@echo "=== Smoke Test: T8 (combined policy, n=5) ==="
	$(PYTHON) benchmarks/combined_policy.py --runs 5 --warmup 2
	@echo ""
	@echo "=== Smoke Test: Report ==="
	$(PYTHON) scripts/generate_reproducibility_report.py || true
	@echo ""
	@echo "=== Smoke Test PASSED ==="

# ============================================================================
# Cleanup
# ============================================================================
clean-logs:
	rm -rf $(RAW_LOGS) $(SUMMARY)
	@echo "Cleaned raw_logs/ and summary/"
