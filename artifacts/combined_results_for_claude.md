# Cache-Aware Winograd Implementation: Combined Experimental Results

This document contains all experimental results, statistical analysis, and system metadata in a clean Markdown format optimized for LLMs (like Claude) to parse.

---

## 1. Hardware & Platform Profile
- **OS:** Darwin
- **Architecture:** arm64
- **CPU Model:** arm
- **Python Version:** 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)]
- **Logical / Physical Cores:** 12 / 12
- **L1 Data Cache Size:** 65536 bytes (64 KB)
- **L2 Cache Size:** 4194304 bytes (4096 KB)
- **Cache Line Size:** 128 bytes
- **Git Commit:** `4c42e69cb319d97ca164a7ea167acc20c586b717`
- **Timestamp:** 2026-03-22T17:58:51.955551

---

## 2. End-to-End Layer Run Status
| mode | status | reason |
| --- | --- | --- |
| end-to-end | unsupported | missing_dependencies |

---

## 3. Autotiler Decisions & Model Configuration
| Workload | Target Cache | Selected Tile | Estimated Working Set (B) | Reuse Score |
| --- | --- | --- | --- | --- |
| C_in=16, C_out=32 | 32 KB | F(2,3) | 18,560 | 0.4414 |
| C_in=32, C_out=16 | 32 KB | F(2,3) | 35,968 | 0.1139 |
| C_in=32, C_out=32 | 32 KB | F(2,3) | 35,968 | 0.2278 |
| C_in=32, C_out=64 | 32 KB | F(2,3) | 35,968 | 0.4555 |
| C_in=64, C_out=32 | 32 KB | F(2,3) | 70,784 | 0.1157 |
| C_in=64, C_out=64 | 32 KB | F(2,3) | 70,784 | 0.2315 |
| C_in=128, C_out=128 | 32 KB | F(2,3) | 140,416 | 0.2334 |
| C_in=16, C_out=32 | 64 KB | F(4,3) | 41,760 | 1.5657 |
| C_in=32, C_out=16 | 64 KB | F(2,3) | 35,968 | 0.2278 |
| C_in=32, C_out=32 | 64 KB | F(2,3) | 35,968 | 0.4555 |
| C_in=32, C_out=64 | 64 KB | F(2,3) | 35,968 | 0.911 |
| C_in=64, C_out=32 | 64 KB | F(2,3) | 70,784 | 0.2315 |
| C_in=64, C_out=64 | 64 KB | F(2,3) | 70,784 | 0.4629 |
| C_in=128, C_out=128 | 64 KB | F(2,3) | 140,416 | 0.4667 |

---

## 4. Microbenchmark Performance & Statistical Verification
| C_in | C_out | Fused | Threads | Mean Latency (ms) | Improvement % | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 32 | No | 1 | 0.5587 | - | N/A |
| 16 | 32 | Yes | 1 | 0.6300 | -12.77% | < 1e-10 |
| 16 | 32 | No | 4 | 0.5541 | +0.83% | 0.00012708484143657386 |
| 16 | 32 | Yes | 4 | 0.6298 | -12.73% | < 1e-10 |
| 32 | 16 | No | 1 | 0.8343 | - | N/A |
| 32 | 16 | Yes | 1 | 0.9956 | -19.33% | < 1e-10 |
| 32 | 16 | No | 4 | 0.8320 | +0.29% | 0.06921017059631815 |
| 32 | 16 | Yes | 4 | 0.9900 | -18.66% | < 1e-10 |
| 32 | 32 | No | 1 | 2.5315 | - | N/A |
| 32 | 32 | Yes | 1 | 2.3396 | +7.58% | < 1e-10 |
| 32 | 32 | No | 4 | 2.5383 | -0.27% | 0.4636329671442905 |
| 32 | 32 | Yes | 4 | 2.3783 | +6.05% | < 1e-10 |
| 32 | 64 | No | 1 | 7.6729 | - | N/A |
| 32 | 64 | Yes | 1 | 6.1602 | +19.71% | < 1e-10 |
| 32 | 64 | No | 4 | 7.6527 | +0.26% | 0.10577731874760402 |
| 32 | 64 | Yes | 4 | 6.1468 | +19.89% | < 1e-10 |
| 64 | 32 | No | 1 | 3.7595 | - | N/A |
| 64 | 32 | Yes | 1 | 3.1474 | +16.28% | < 1e-10 |
| 64 | 32 | No | 4 | 3.7665 | -0.19% | 0.23109333337777843 |
| 64 | 32 | Yes | 4 | 3.1487 | +16.25% | < 1e-10 |
| 64 | 64 | No | 1 | 12.3424 | - | N/A |
| 64 | 64 | Yes | 1 | 8.7733 | +28.92% | < 1e-10 |
| 64 | 64 | No | 4 | 12.3103 | +0.26% | 0.03914590939858788 |
| 64 | 64 | Yes | 4 | 8.8056 | +28.66% | < 1e-10 |
| 128 | 128 | No | 1 | 79.0075 | - | N/A |
| 128 | 128 | Yes | 1 | 46.1133 | +41.63% | < 1e-10 |
| 128 | 128 | No | 4 | 77.9719 | +1.31% | < 1e-10 |
| 128 | 128 | Yes | 4 | 46.0179 | +41.76% | < 1e-10 |

---

## 5. System-Level Backend Comparison (vs ONNX Runtime)
| Backend | Regime (C_in → C_out) | Avg Latency (ms) | Std Dev (ms) | Notes |
| --- | --- | --- | --- | --- |
| project | 64 → 64 | 0.1807 | 0.0150 | Fused Winograd F(2,3) tile microbenchmark. Overhead-dominated regime; Winograd transform cost dominates on 4x4 input. |
| onnxruntime | 64 → 64 | 0.0164 | 0.0010 | Single Conv ONNX Runtime CPU benchmark. Direct convolution is optimal on micro-inputs. |
| project | 128 → 128 | 0.2736 | 0.0644 | Fused Winograd F(2,3) tile microbenchmark. Arithmetic-dominated regime; fused path dominates on large tiles. |
| onnxruntime | 128 → 128 | 5.9723 | 0.3014 | Single Conv ONNX Runtime CPU benchmark. |

---

## 6. Summary Statistics of Raw Trials
Below are the aggregated statistics across all 840 trials grouped by configuration:
| Configuration | Count | Mean (ms) | Min (ms) | Max (ms) | Std Dev (ms) |
| --- | --- | --- | --- | --- | --- |
| C_in=128, C_out=128, Fused=False, Threads=1 | 30 | 79.00751 | 77.79083 | 79.75462 | 0.56135 |
| C_in=128, C_out=128, Fused=False, Threads=4 | 30 | 77.97192 | 77.51904 | 78.34633 | 0.22318 |
| C_in=128, C_out=128, Fused=True, Threads=1 | 30 | 46.11326 | 45.81154 | 46.39921 | 0.13779 |
| C_in=128, C_out=128, Fused=True, Threads=4 | 30 | 46.01791 | 45.65742 | 46.36863 | 0.21280 |
| C_in=16, C_out=32, Fused=False, Threads=1 | 30 | 0.55868 | 0.54875 | 0.56921 | 0.00509 |
| C_in=16, C_out=32, Fused=False, Threads=4 | 30 | 0.55405 | 0.54850 | 0.56321 | 0.00335 |
| C_in=16, C_out=32, Fused=True, Threads=1 | 30 | 0.63003 | 0.62533 | 0.63846 | 0.00356 |
| C_in=16, C_out=32, Fused=True, Threads=4 | 30 | 0.62982 | 0.62404 | 0.63846 | 0.00392 |
| C_in=32, C_out=16, Fused=False, Threads=1 | 30 | 0.83435 | 0.82604 | 0.85379 | 0.00571 |
| C_in=32, C_out=16, Fused=False, Threads=4 | 30 | 0.83197 | 0.82600 | 0.84242 | 0.00411 |
| C_in=32, C_out=16, Fused=True, Threads=1 | 30 | 0.99561 | 0.98950 | 1.00271 | 0.00351 |
| C_in=32, C_out=16, Fused=True, Threads=4 | 30 | 0.99001 | 0.98458 | 0.99921 | 0.00335 |
| C_in=32, C_out=32, Fused=False, Threads=1 | 30 | 2.53149 | 2.49817 | 2.67100 | 0.03998 |
| C_in=32, C_out=32, Fused=False, Threads=4 | 30 | 2.53833 | 2.51496 | 2.69304 | 0.03131 |
| C_in=32, C_out=32, Fused=True, Threads=1 | 30 | 2.33959 | 2.32992 | 2.36271 | 0.00747 |
| C_in=32, C_out=32, Fused=True, Threads=4 | 30 | 2.37830 | 2.32858 | 2.64437 | 0.07580 |
| C_in=32, C_out=64, Fused=False, Threads=1 | 30 | 7.67289 | 7.59213 | 7.77775 | 0.04035 |
| C_in=32, C_out=64, Fused=False, Threads=4 | 30 | 7.65273 | 7.58200 | 7.83354 | 0.05366 |
| C_in=32, C_out=64, Fused=True, Threads=1 | 30 | 6.16021 | 6.12846 | 6.42825 | 0.05284 |
| C_in=32, C_out=64, Fused=True, Threads=4 | 30 | 6.14683 | 6.12504 | 6.23829 | 0.02255 |
| C_in=64, C_out=32, Fused=False, Threads=1 | 30 | 3.75946 | 3.72113 | 3.80058 | 0.02477 |
| C_in=64, C_out=32, Fused=False, Threads=4 | 30 | 3.76651 | 3.73533 | 3.79854 | 0.02003 |
| C_in=64, C_out=32, Fused=True, Threads=1 | 30 | 3.14745 | 3.13254 | 3.17575 | 0.00868 |
| C_in=64, C_out=32, Fused=True, Threads=4 | 30 | 3.14871 | 3.11333 | 3.46754 | 0.06096 |
| C_in=64, C_out=64, Fused=False, Threads=1 | 30 | 12.34244 | 12.27167 | 12.39600 | 0.03046 |
| C_in=64, C_out=64, Fused=False, Threads=4 | 30 | 12.31028 | 12.23808 | 12.62229 | 0.07660 |
| C_in=64, C_out=64, Fused=True, Threads=1 | 30 | 8.77330 | 8.72321 | 9.06729 | 0.06939 |
| C_in=64, C_out=64, Fused=True, Threads=4 | 30 | 8.80565 | 8.76987 | 8.94921 | 0.03565 |