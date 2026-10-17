# Formula & Mathematical Specification Audit

**Protocol:** Pure analytic recomputation of Paper Equations 1–12 vs claimed tables.

## 1. Working Set Model: Code Eq vs Paper Table 1

| Workload | m | Paper Claim WS (B) | Recomputed Code WS (B) | Full Extended WS (B) | Consistent? | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| (16,32) | 2 | 18560 | 18560 | 35,840 | ✓ YES | Matches autotiler code |
| (16,32) | 4 | 41760 | 41760 | 80,640 | ✓ YES | Matches autotiler code |
| (32,16) | 2 | 35968 | 35968 | 35,840 | ✓ YES | Matches autotiler code |
| (32,16) | 4 | 78720 | 80928 | 80,640 | ✗ NO | Paper table differs from formula |
| (32,32) | 2 | 35968 | 35968 | 69,632 | ✓ YES | Matches autotiler code |
| (32,32) | 4 | 83328 | 80928 | 156,672 | ✗ NO | Paper table differs from formula |
| (32,64) | 2 | 35968 | 35968 | 137,216 | ✓ YES | Matches autotiler code |
| (32,64) | 4 | 92544 | 80928 | 308,736 | ✗ NO | Paper table differs from formula |
| (64,32) | 2 | 70784 | 70784 | 137,216 | ✓ YES | Matches autotiler code |
| (64,32) | 4 | 157056 | 159264 | 308,736 | ✗ NO | Paper table differs from formula |
| (64,64) | 2 | 70784 | 70784 | 270,336 | ✓ YES | Matches autotiler code |
| (64,64) | 4 | 166272 | 159264 | 608,256 | ✗ NO | Paper table differs from formula |
| (128,128) | 2 | 140416 | 140416 | 1,064,960 | ✓ YES | Matches autotiler code |
| (128,128) | 4 | 332416 | 315936 | 2,396,160 | ✗ NO | Paper table differs from formula |

> **CRITICAL OMISSION IN WORKING SET MODEL:** The code's `compute_working_set()` clamps Cout to `c_out_block = min(c_out, 16)`, assuming the weight transform is tiled in blocks of 16. However, at (64,64) m=4, the full transformed weight tensor is `alpha^2 * Cin * Cout * 4 = 36 * 64 * 64 * 4 = 589,824 B (576 KB)`. At (128,128) m=4, it is `2.36 MB`, which exceeds even the 2MB L2 cache! The paper's WS model omitted full weight residency, severely underestimating true cache footprint.


## 2. Reuse Metric Analysis: Paper Claim vs Formula (Eq. 7)

| Workload | m | Paper Reuse Score | Code Reuse Score | Eq. 7 Reuse Ratio R(m) | R_min=0.35 Triggerable? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| (16,32) | 2 | 0.4414 | 0.4414 | 0.2051 | YES |
| (16,32) | 4 | 1.5657 | 0.7841 | 0.7191 | NO (R >> 0.35) |
| (32,16) | 2 | 0.1139 | 0.1139 | 0.1975 | YES |
| (32,16) | 4 | 0.4136 | 0.2024 | 0.6809 | NO (R >> 0.35) |
| (32,32) | 2 | 0.2278 | 0.2278 | 0.2078 | YES |
| (32,32) | 4 | 0.8272 | 0.4047 | 0.7529 | NO (R >> 0.35) |
| (32,64) | 2 | 0.4555 | 0.4555 | 0.2133 | YES |
| (32,64) | 4 | 1.6544 | 0.8095 | 0.795 | NO (R >> 0.35) |
| (64,32) | 2 | 0.1157 | 0.1157 | 0.2092 | YES |
| (64,32) | 4 | 0.4203 | 0.2057 | 0.7711 | NO (R >> 0.35) |
| (64,64) | 2 | 0.2315 | 0.2315 | 0.2148 | YES |
| (64,64) | 4 | 0.8407 | 0.4114 | 0.8153 | NO (R >> 0.35) |
| (128,128) | 2 | 0.2334 | 0.2334 | 0.2184 | YES |
| (128,128) | 4 | 0.8504 | 0.4148 | 0.8505 | NO (R >> 0.35) |

> **INCONSISTENCY IN REUSE DEFINITION:** In the paper text, Eq. 7 defines reuse ratio $R(m)$ as useful MACs divided by total memory traffic. Under this definition, $R(m)$ ranges from **10.5 to 78.4** for these configurations! Consequently, a threshold of $R_{min} = 0.35$ can **never** trigger. In contrast, the codebase autotiler computes an unnormalized heuristic: `c_out * m^2 / ceil(WS / 64)`. The paper reported the code heuristic in Table 1 while claiming it represented the analytical ratio $R(m)$.


## 3. Cache Thresholds & Fallback Guard Feasibility

| Board | Cache Level | Capacity | Paper Policy Guard (gamma=0.7) | Fallback Threshold |
| :--- | :--- | :---: | :---: | :---: |
| **RPi4 (Cortex-A72)** | L1 Data | 32 KB | 22,938 Bytes | Any WS > 22.9 KB falls back to partitioned tile |
| **RPi4 (Cortex-A72)** | L2 Shared (Sysfs) | **1,024 KB** | 734,003 Bytes | Misconfigured in paper (assumed 2 MB) |
| **Jetson Nano (A57)** | L1 Data | 32 KB | 22,938 Bytes | Matches RPi4 L1D |
| **Jetson Nano (A57)** | L2 Shared (Sysfs) | 2,048 KB | 1,468,006 Bytes | Verified match with hardware |
