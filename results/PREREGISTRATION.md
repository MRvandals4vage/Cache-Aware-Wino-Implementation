# Preregistration of Experimental Protocol & Analysis Plan

**Document Status:** LOCKED PRIOR TO EXPERIMENTAL EXECUTION  
**Target Submission:** Journal of Systems Architecture (JSA) Revision  
**Target Hardware:** Raspberry Pi 4 Model B (Quad Cortex-A72, 1 MB L2) and NVIDIA Jetson Nano (Quad Cortex-A57, 2 MB L2)  
**Evaluated Systems:** CacheWinograd-v1 (paper baseline), Additive Variants (v2+), Baselines (Naive direct, NEON direct, Naive Winograd, Non-fused Winograd, ArmCL, TVM/AutoTVM, ONNX Runtime/In-house Graph Runner).

---

## 1. Experimental Protocol & Primary Metrics

### 1.1 Fixed Execution Parameters
- **Data Type:** Single-precision floating point (FP32).
- **Batch Size:** 1 (Edge inference scenario).
- **Spatial Dimensions:** $H = W = 56$ (Primary standard), with Tier B sweeps at $H = W \in \{7, 14, 28, 56, 112\}$.
- **Primary Baselines:** Every performance delta MUST be reported against BOTH:
  1. `naive_direct`: Unvectorized 3-loop spatial direct convolution.
  2. `neon_direct`: Vectorized over $C_{out}$ with register-tiled NEON FMA inner loops.
- **Reference Policy ($P_0$):** Exactly as specified in paper: largest $m$ satisfying $WS_{ext} \le \gamma \cdot C_{L2}$ ($\gamma = 0.70$), subject to two-condition guard.

### 1.2 Primary Metrics & Statistical Suite
- **Primary Metric:** Mean wall-clock execution latency (ms) over $n$ post-warmup iterations.
- **Secondary Metrics:** Implied throughput (GFLOPS), L1D/L2 cache refills per MAC, Energy per inference (mJ, Jetson only).
- **Statistical Significance Suite:**
  - Parametric: Two-sided Welch's $t$-test (unequal variance).
  - Non-parametric: Mann-Whitney $U$ test.
  - Effect Size: Cohen's $d$.
  - Uncertainty: Bias-corrected and accelerated (BCa) bootstrap 95% Confidence Intervals (10,000 resamples) on mean latency and percent improvement.
  - Family-Wise Error Rate: Holm-Bonferroni step-down correction applied across the 7 canonical workloads.
  - Empirical Noise Floor: Defined by Phase 1b null-test false-positive rate. Any delta falling below the noise floor is strictly classified as *"Not distinguishable from noise"*.

---

## 2. Train / Held-Out Partitioning

To rigorously evaluate the Working Set Model (Phase 5) and Tile/Thread Policies (Phase 6) without overfitting:

### 2.1 Working Set Model Split (Phase 5)
- **Training Set (Fit coefficients):**  
  `{(16, 32), (32, 16), (32, 32), (64, 64)}` across $m \in \{2, 4, 6\}$.
- **Held-Out Test Set (Evaluate generalization):**  
  `{(32, 64), (64, 32), (128, 128), (48, 96), (96, 48)}` across $m \in \{2, 4, 6\}$.

### 2.2 Policy Selection Split (Phase 6)
- **Training Grid:** Canonical microbenchmarks $C_{in} = C_{out} \in \{16, 32, 64\}$.
- **Held-Out Grid:** Cross-channel workloads $(32, 64), (64, 32), (48, 96), (96, 48)$ and full network layer shapes (Tier D).

---

## 3. Pre-Registered Outcome-Conditional Reporting Rules

These rules dictate reporting language and classifications objectively based on data, preventing confirmation bias:

| Condition / Observation | Pre-Registered Classification | Required Paper Reporting Action |
| :--- | :--- | :--- |
| $\text{Improvement} \le \text{Noise Floor}$ (Phase 1b) | **NULL RESULT** | Report as "No statistically significant difference observed". Do not claim speedup. |
| Fused is slower than NEON-direct baseline ($p < 0.05$) | **REGRESSION** | Report exact regression percentage and diagnose root cause (e.g. transform overhead). |
| $P_0$ Regret vs Oracle $P_2 > 5.0\%$ on Held-Out | **POLICY INVERSION / FAILURE** | Report $P_0$ as failing; document where static cache heuristic makes sub-optimal choice. |
| $(128,128)$ Disassembly has $>0$ spills & ratio $>2.0$ | **REGISTER SPILL BOTTLENECK** | Confirm register pressure hypothesis as the definitive limiter on 32-register NEON. |
| 4-Thread Fused vs 4-Thread Unfused $< 5.0\%$ speedup | **MULTI-CORE SCALING COLLAPSE** | Report lack of multi-threaded scaling due to barrier and cache-line false sharing. |
| Measured Sysfs L2 $\ne$ Paper Assumption | **HARDWARE MISMATCH** | Document physical hardware discrepancy (e.g. RPi4 1MB vs 2MB) in Claims Ledger. |
| E2E inference measured as sum of isolated layers | **LAYER-TIME AGGREGATE** | Explicitly prefix every metric with "Layer-time aggregate, not monolithic graph trace". |

---

## 4. Gate Enforcement Criteria (Part E)

Every measurement row must satisfy Gates G1 through G8 before inclusion in final paper tables. Any violating run is immediately quarantined to `results/<board>/quarantine/` and documented in `results/QUARANTINE.md`.
