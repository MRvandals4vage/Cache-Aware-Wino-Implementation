# CacheWinograd Reproducibility Report

Generated: 2026-10-15T23:20:20.444404

> **METHODOLOGY NOTE (Constraint #6):** CacheWinograd end-to-end latency is a **layer-time aggregate** of isolated per-layer kernel measurements, NOT a single fused end-to-end trace. Baseline end-to-end numbers (ORT, TVM, ArmCL) ARE true single-process runs. This asymmetry is stated explicitly and not hidden in any table below.


> **Column definitions:**
> - **Prior**: last value from `raw_logs_prior/` / `summary_prior/`
> - **Fresh**: value measured today on-device
> - **%Diff**: (fresh − prior) / |prior| × 100
> - **Match**: ✓ if |%diff| ≤ 5%, ✗ otherwise


---

## 0. Log File Inventory

| Task | File | Status | Prior File | Prior Status |
| :--- | :--- | :--- | :--- | :--- |
| A_cache_probe | `raw_logs/jetson_cacheprobe.csv` | ✓ FOUND | `raw_logs_prior/jetson_cacheprobe.csv` | ⚠ MISSING |
| B_ws_validation | `raw_logs/jetson_ws_validation.csv` | ✗ MISSING (REQUIRED) | `raw_logs_prior/jetson_ws_validation.csv` | ⚠ MISSING |
| C_jetson_microbench | `raw_logs/jetson_main_microbench.csv` | ✓ FOUND | `raw_logs_prior/jetson_main_microbench.csv` | ⚠ MISSING |
| D_tvm_armcl | `raw_logs/tvm_armcl_full_traces.csv` | ✓ FOUND | `raw_logs_prior/tvm_armcl_full_traces.csv` | ⚠ MISSING |
| E_ablation | `raw_logs/jetson_ablation.csv` | ✓ FOUND | `raw_logs_prior/jetson_ablation.csv` | ⚠ MISSING |
| F_l2_macsj | `raw_logs/jetson_l2_macsj.csv` | ✗ MISSING (REQUIRED) | `raw_logs_prior/jetson_l2_macsj.csv` | ⚠ MISSING |
| G_fallback_heldout | `raw_logs/jetson_fallback_heldout.csv` | ✓ FOUND | `raw_logs_prior/jetson_fallback_heldout.csv` | ⚠ MISSING |
| H_scheduling_overhead | `raw_logs/jetson_scheduling_overhead.csv` | ✗ MISSING (REQUIRED) | `raw_logs_prior/jetson_scheduling_overhead.csv` | ⚠ MISSING |
| I_register_spill | `raw_logs/register_spill_128x128.csv` | ✓ FOUND | `raw_logs_prior/register_spill_128x128.csv` | ⚠ MISSING |
| J_energy_per_run | `raw_logs/energy_per_run_vgg16.csv` | ✗ MISSING (REQUIRED) | `raw_logs_prior/energy_per_run_vgg16.csv` | ⚠ MISSING |
| K_e2e_jetson | `summary/e2e_comparison_table.csv` | ✓ FOUND | `summary_prior/e2e_comparison_table.csv` | ⚠ MISSING |
| M_rpi4_microbench | `raw_logs/rpi4_main_microbench.csv` | ✓ FOUND | `raw_logs_prior/rpi4_main_microbench.csv` | ⚠ MISSING |
| N_adaptive_m | `raw_logs/rpi4_adaptive_m_sweep.csv` | ✓ FOUND | `raw_logs_prior/rpi4_adaptive_m_sweep.csv` | ⚠ MISSING |
| O_e2e_rpi4 | `summary/e2e_comparison_table.csv` | ✓ FOUND | `summary_prior/e2e_comparison_table.csv` | ⚠ MISSING |
| L_combined_policy | `raw_logs/combined_policy.csv` | ✓ FOUND | `raw_logs_prior/combined_policy.csv` | ⚠ MISSING |
| P_apple_silicon | `raw_logs/e2e_alexnet_macos_apple_silicon.csv` | ⚠ OPTIONAL | `raw_logs_prior/e2e_alexnet_macos_apple_silicon.csv` | ⚠ MISSING |

---

## A. Cache Parameters

| Cache | Paper (bytes) | Prior (bytes) | Fresh (bytes) | %Diff Prior→Fresh | Paper Match | Prior→Fresh Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| L1D | 32768 | N/A | 32768 | N/A | MATCH | — |
| L2 | 2097152 | N/A | 2097152 | N/A | MATCH | — |
| L1D | 32768 | N/A | 32768 | N/A | MATCH | — |
| L2 | 2097152 | N/A | 2097152 | N/A | MATCH | — |
| L1D | 32768 | N/A | 32768 | N/A | MATCH | — |
| L2 | 2097152 | N/A | 2097152 | N/A | MATCH | — |
| L1D | 32768 | N/A | 32768 | N/A | MATCH | — |
| L2 | 2097152 | N/A | 2097152 | N/A | MATCH | — |
| L1D | 32768 | N/A | 32768 | N/A | MATCH | — |
| L2 | 2097152 | N/A | 2097152 | N/A | MATCH | — |
| L1D | 32768 | N/A | 32768 | N/A | MATCH | — |
| L2 | 2097152 | N/A | 2097152 | N/A | MATCH | — |
| L1D | 32768 | N/A | 32768 | N/A | MATCH | — |
| L2 | 2097152 | N/A | 2097152 | N/A | MATCH | — |

---

## B. Working-Set Model Validation

> **NOT RUN** — Execute `make taskB` on Jetson Nano


---

## C. Jetson Nano Main Microbenchmark (All 7 Configs)

| Config | Prior Baseline ms | Fresh Baseline ms | Prior Fused ms | Fresh Fused ms | Prior Imp% | Fresh Imp% | %Diff Imp | Paper Imp% | Paper Match | p-value | Sig |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (16,32) | N/A | 2.3064 | N/A | 0.9881 | N/A | 57.1572 | N/A | N/A | NO_PAPER_CLAIM | 2.798286e-01 | NOT_SIGNIFICANT |
| (32,16) | N/A | 1.0978 | N/A | 1.1056 | N/A | -0.7105 | N/A | N/A | NO_PAPER_CLAIM | 6.230036e-01 | NOT_SIGNIFICANT |
| (32,32) | N/A | 2.7597 | N/A | 2.7208 | N/A | 1.4069 | N/A | N/A | NO_PAPER_CLAIM | 1.277267e-01 | NOT_SIGNIFICANT |
| (32,64) | N/A | 8.191 | N/A | 8.1507 | N/A | 0.4917 | N/A | N/A | NO_PAPER_CLAIM | 6.580246e-02 | NOT_SIGNIFICANT |
| (64,32) | N/A | 4.5615 | N/A | 4.545 | N/A | 0.3612 | N/A | N/A | NO_PAPER_CLAIM | 2.764236e-01 | NOT_SIGNIFICANT |
| (64,64) | N/A | 14.7619 | N/A | 14.7539 | N/A | 0.0548 | N/A | N/A | NO_PAPER_CLAIM | 5.779144e-01 | NOT_SIGNIFICANT |
| (128,128) | N/A | 44281.4987 | N/A | 43695.8825 | N/A | 1.3225 | N/A | -6.52 | MISMATCH | 1.089041e-01 | NOT_SIGNIFICANT |

---

## D. TVM / AutoTVM / ArmCL Statistical Rigor

| Config | Backend | Fresh Mean ms | Fresh CI95 | Prior Mean ms | %Diff | p-value | Significant? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (32,64) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (32,64) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (32,64) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (64,32) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (64,32) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (64,32) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (64,64) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (64,64) | None | N/A | None | N/A | N/A | N/A | SKIPPED |
| (64,64) | None | N/A | None | N/A | N/A | N/A | SKIPPED |

---

## E. Multi-Core / Fusion × Threading Ablation

| Config | Mode | Fresh Mean ms | Fresh CI95 | Prior Mean ms | %Diff | Paper Imp% | Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (64,64) | unfused_1T | 14.7835 | 0.0368 | N/A | N/A |  |  |
| (64,64) | unfused_4T | 14.8315 | 0.0132 | N/A | N/A |  |  |
| (64,64) | fused_1T | 14.7183 | 0.0238 | N/A | N/A |  |  |
| (64,64) | fused_4T | 14.7211 | 0.0153 | N/A | N/A | 33.11 | MISMATCH |
| (128,128) | unfused_1T | 100.71 | 0.1376 | N/A | N/A |  |  |
| (128,128) | unfused_4T | 101.1222 | 0.1633 | N/A | N/A |  |  |
| (128,128) | fused_1T | 101.1078 | 0.1296 | N/A | N/A |  |  |
| (128,128) | fused_4T | 100.9899 | 0.119 | N/A | N/A |  |  |
| (32,64) | fused_1T | 8.1822 | 0.0182 | N/A | N/A |  |  |
| (32,64) | fused_4T | 8.1635 | 0.0094 | N/A | N/A |  |  |

---

## F. L2 Hit Rate + MACs/Joule

> **NOT RUN** — Execute `make taskF` on Jetson Nano


---

## G. Fallback-Guard Verification Table (All 8 Rows)


### G1. Standard configs (from Task C data)

*(Rows for (16,32),(32,16),(32,32),(32,64),(64,32),(64,64) come from Section C above — see there for details)*


### G2. Held-out configs

| Config | Fallback? | Tile | WS (B) | L1 (B) | Fresh Imp% | Prior Imp% | %Diff | p-value | Sig |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (48,96) | True | F(2,3) | 53376 | 32768 | 0.3045 | N/A | N/A | < 1e-10 | SIGNIFICANT |
| (96,48) | True | F(2,3) | 105600 | 32768 | 0.2122 | N/A | N/A | < 1e-10 | SIGNIFICANT |

---

## H. Scheduling Overhead — Measured vs Architectural Estimate

> **NOT RUN** — Execute `make taskH` on Jetson/any Linux


---

## I. Register-Spill Instrumentation (128,128)

| Method | Spill Count (fresh) | Spill Count (prior) | %Diff | Over-sub Ratio | Hypothesis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| static_objdump | N/A | N/A | N/A | 2.0 | N/A |

---

## J. Energy Per-Run Distribution — VGG16

> **NOT RUN** — Execute `make taskJ` on Jetson Nano


---

## K/O. End-to-End Inference (Jetson Nano + Raspberry Pi 4)


> **METHODOLOGY (Constraint #6):** CacheWinograd E2E = layer-time aggregate. ORT / TVM / ArmCL = true single-process runs.

| Model | Platform | Baseline | Prior Mean ms | Fresh Mean ms | CI95 | %Diff | Fresh Imp% | p-value | Sig | Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VGG16 | rpi4 | cachewinograd | N/A | 1846777.8981 | 17767.1346 | N/A | N/A | N/A | None | — |
| VGG16 | rpi4 | naive_winograd | N/A | 1939926.3183 | 542.063 | N/A | 4.8 | 2.822119e-06 | None | — |
| VGG16 | rpi4 | nonfused_winograd | N/A | 1939916.3387 | 120.1175 | N/A | 4.8 | 2.852992e-06 | None | — |

---

## M. Raspberry Pi 4 Main Microbenchmark (All 7 Configs, m=4)

| Config | Prior Baseline ms | Fresh Baseline ms | Prior Fused ms | Fresh Fused ms | Prior Imp% | Fresh Imp% | %Diff Imp | Paper Imp% | Match | p-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (16,32) | N/A | 8.9637 | N/A | 8.9436 | N/A | 0.2237 | N/A | N/A | NO_PAPER_CLAIM | 2.134591e-03 |
| (32,16) | N/A | 4.3871 | N/A | 4.3759 | N/A | 0.255 | N/A | N/A | NO_PAPER_CLAIM | 3.616083e-02 |
| (32,32) | N/A | 13.3931 | N/A | 13.372 | N/A | 0.1579 | N/A | N/A | NO_PAPER_CLAIM | 2.972568e-01 |
| (32,64) | N/A | 44.0 | N/A | 43.8069 | N/A | 0.4388 | N/A | N/A | NO_PAPER_CLAIM | 1.595468e-01 |
| (64,32) | N/A | 21.9022 | N/A | 21.7475 | N/A | 0.7064 | N/A | N/A | NO_PAPER_CLAIM | 1.757165e-01 |
| (64,64) | N/A | 85.33 | N/A | 85.5026 | N/A | -0.2023 | N/A | N/A | NO_PAPER_CLAIM | 4.635008e-01 |
| (128,128) | N/A | 2322.1473 | N/A | 2322.3976 | N/A | -0.0108 | N/A | 63.91 | MISMATCH | 6.020136e-01 |

---

## N. Adaptive-m Selection — RPi4 All 7 Configs at m∈{2,4,6}

| Config | Policy m | Empirical Best m | m=2 ms | m=6 ms | p(m2 vs m6) | Match? | Prior Match? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (16,32) | 2 | 6 | 19.208±0.0471 | 6.7565±0.005 | < 1e-10 | False | N/A |
| (32,16) | 2 | 6 | 9.6936±0.0088 | 3.8688±0.0036 | < 1e-10 | False | N/A |
| (32,32) | 2 | 6 | 24.9978±0.1512 | 12.525±0.0181 | < 1e-10 | False | N/A |
| (32,64) | 2 | 6 | 68.4253±0.0902 | 52.3418±0.1018 | < 1e-10 | False | N/A |
| (64,32) | 2 | 6 | 34.0639±0.0403 | 21.5727±0.1517 | < 1e-10 | False | N/A |
| (64,64) | 2 | 2 | 99.1615±0.1461 | 154.3409±0.293 | < 1e-10 | True | N/A |
| (128,128) | 2 | 2 | 1229.2463±2.0973 | 1244.8327±0.542 | < 1e-10 | True | N/A |

---

## R. R_min Sensitivity Table (Recomputed)

| R_min | Config | Selected Tile | Jetson Imp% | RPi4 Imp% |
| :--- | :--- | :--- | :--- | :--- |
| 0.1 | 16,32 | F(4,3) | 57.1572 | 0.2237 |
| 0.1 | 32,16 | F(2,3) | -0.7105 | 0.255 |
| 0.1 | 32,32 | F(2,3) | 1.4069 | 0.1579 |
| 0.1 | 32,64 | F(2,3) | 0.4917 | 0.4388 |
| 0.1 | 64,32 | F(2,3) | 0.3612 | 0.7064 |
| 0.1 | 64,64 | F(2,3) | 0.0548 | -0.2023 |
| 0.1 | 128,128 | F(2,3) | 1.3225 | -0.0108 |
| 0.2 | 16,32 | F(4,3) | 57.1572 | 0.2237 |
| 0.2 | 32,16 | F(2,3) | -0.7105 | 0.255 |
| 0.2 | 32,32 | F(2,3) | 1.4069 | 0.1579 |
| 0.2 | 32,64 | F(2,3) | 0.4917 | 0.4388 |
| 0.2 | 64,32 | F(2,3) | 0.3612 | 0.7064 |
| 0.2 | 64,64 | F(2,3) | 0.0548 | -0.2023 |
| 0.2 | 128,128 | F(2,3) | 1.3225 | -0.0108 |
| 0.35 | 16,32 | F(4,3) | 57.1572 | 0.2237 |
| 0.35 | 32,16 | F(2,3) | -0.7105 | 0.255 |
| 0.35 | 32,32 | F(2,3) | 1.4069 | 0.1579 |
| 0.35 | 32,64 | F(2,3) | 0.4917 | 0.4388 |
| 0.35 | 64,32 | F(2,3) | 0.3612 | 0.7064 |
| 0.35 | 64,64 | F(2,3) | 0.0548 | -0.2023 |
| 0.35 | 128,128 | F(2,3) | 1.3225 | -0.0108 |
| 0.5 | 16,32 | F(2,3) | 57.1572 | 0.2237 |
| 0.5 | 32,16 | F(2,3) | -0.7105 | 0.255 |
| 0.5 | 32,32 | F(2,3) | 1.4069 | 0.1579 |
| 0.5 | 32,64 | F(2,3) | 0.4917 | 0.4388 |
| 0.5 | 64,32 | F(2,3) | 0.3612 | 0.7064 |
| 0.5 | 64,64 | F(2,3) | 0.0548 | -0.2023 |
| 0.5 | 128,128 | F(2,3) | 1.3225 | -0.0108 |
| 0.65 | 16,32 | F(2,3) | 57.1572 | 0.2237 |
| 0.65 | 32,16 | F(2,3) | -0.7105 | 0.255 |
| 0.65 | 32,32 | F(2,3) | 1.4069 | 0.1579 |
| 0.65 | 32,64 | F(2,3) | 0.4917 | 0.4388 |
| 0.65 | 64,32 | F(2,3) | 0.3612 | 0.7064 |
| 0.65 | 64,64 | F(2,3) | 0.0548 | -0.2023 |
| 0.65 | 128,128 | F(2,3) | 1.3225 | -0.0108 |
| 0.8 | 16,32 | F(2,3) | 57.1572 | 0.2237 |
| 0.8 | 32,16 | F(2,3) | -0.7105 | 0.255 |
| 0.8 | 32,32 | F(2,3) | 1.4069 | 0.1579 |
| 0.8 | 32,64 | F(2,3) | 0.4917 | 0.4388 |
| 0.8 | 64,32 | F(2,3) | 0.3612 | 0.7064 |
| 0.8 | 64,64 | F(2,3) | 0.0548 | -0.2023 |
| 0.8 | 128,128 | F(2,3) | 1.3225 | -0.0108 |

---

## S. Cross-Platform Comparison Table (Recomputed)

| Config | Jetson Baseline ms | Jetson Fused ms | Jetson Imp% | RPi4 Baseline ms | RPi4 Fused ms | RPi4 Imp% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (16,32) | 2.3064 | 0.9881 | 57.1572 | 8.9637 | 8.9436 | 0.2237 |
| (32,16) | 1.0978 | 1.1056 | -0.7105 | 4.3871 | 4.3759 | 0.255 |
| (32,32) | 2.7597 | 2.7208 | 1.4069 | 13.3931 | 13.372 | 0.1579 |
| (32,64) | 8.191 | 8.1507 | 0.4917 | 44.0 | 43.8069 | 0.4388 |
| (64,32) | 4.5615 | 4.545 | 0.3612 | 21.9022 | 21.7475 | 0.7064 |
| (64,64) | 14.7619 | 14.7539 | 0.0548 | 85.33 | 85.5026 | -0.2023 |
| (128,128) | 44281.4987 | 43695.8825 | 1.3225 | 2322.1473 | 2322.3976 | -0.0108 |

---

## T. Roofline Table (Recomputed)

| Config | MACs | AI (FLOP/B) | Ridge (FLOP/B) | Bounded By | Measured GFLOPS | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (16,32) | 14450688 | 23.2871 | 1.7891 | COMPUTE | 14.6247 | JN peak-compute and BW are board-level constants from paper  |
| (32,16) | 14450688 | 23.2871 | 1.7891 | COMPUTE | 13.0704 | JN peak-compute and BW are board-level constants from paper  |
| (32,32) | 28901376 | 34.4195 | 1.7891 | COMPUTE | 10.6224 | JN peak-compute and BW are board-level constants from paper  |
| (32,64) | 57802752 | 45.2308 | 1.7891 | COMPUTE | 7.0918 | JN peak-compute and BW are board-level constants from paper  |
| (64,32) | 57802752 | 45.2308 | 1.7891 | COMPUTE | 12.7179 | JN peak-compute and BW are board-level constants from paper  |
| (64,64) | 115605504 | 65.9439 | 1.7891 | COMPUTE | 7.8356 | JN peak-compute and BW are board-level constants from paper  |
| (128,128) | 462422016 | 121.6552 | 1.7891 | COMPUTE | 0.0106 | JN peak-compute and BW are board-level constants from paper  |

---

## U. Fusion DRAM Savings Table (Recomputed)

| Config | Tile | Transform Bytes Eliminated | Total Non-fused DRAM | Savings % |
| :--- | :--- | :--- | :--- | :--- |
| (16,32) | F(4,3) | 4608 | 606720 | 0.76 |
| (32,16) | F(2,3) | 4096 | 606208 | 0.68 |
| (32,32) | F(2,3) | 4096 | 806912 | 0.51 |
| (32,64) | F(2,3) | 4096 | 1208320 | 0.34 |
| (64,32) | F(2,3) | 8192 | 1212416 | 0.68 |
| (64,64) | F(2,3) | 8192 | 1613824 | 0.51 |
| (128,128) | F(2,3) | 16384 | 3227648 | 0.51 |

---

## Z. Summary of Missing Logs

> **4 REQUIRED log(s) missing:**

> - `B_ws_validation`: `raw_logs/jetson_ws_validation.csv`
> - `F_l2_macsj`: `raw_logs/jetson_l2_macsj.csv`
> - `H_scheduling_overhead`: `raw_logs/jetson_scheduling_overhead.csv`
> - `J_energy_per_run`: `raw_logs/energy_per_run_vgg16.csv`
