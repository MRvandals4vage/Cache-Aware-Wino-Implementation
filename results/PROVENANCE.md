# Provenance Audit & Inconsistency Tracking

Auditing historical code paths, spatial dimensions, baseline definitions, and discrepancies between earlier internal audit runs and paper claims.

## 1. Discrepancy Reconciliation Ledger

| Issue / Investigation | Earlier Audit Protocol | Paper Protocol | Physical Reality | Root Cause & Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **(128,128) Spatial Resolution** | H=W=14 on RPi; H=W=128 on Jetson | H=W=56 claimed | H=W=128 took 43.7s on Jetson; H=W=14 took 2.3s on RPi | Different spatial dimensions were run in different legacy scripts. Pinned to H=W=56 for primary protocol. |
| **Fused vs Baseline ~0% delta** | Fused and baseline within 0.2% | Fused claimed +63.9% | Baseline and fused paths executed identical C code if fallback triggered | Guard fallback silently dropped into non-fused kernel without logging executed path. Asserted in fresh harness. |
| **Multi-core 4T Scaling == 1T** | 4T showed 0% scaling even on unfused | 4T claimed 33.1% speedup | Thread affinity / OpenMP thread pool was idle or unengaged | Harness failed to bind threads with OMP_PROC_BIND=close. Phase 1e asserts thread engagement. |
| **RPi4 L2 Cache Size** | Assumed 2 MB in paper text | Assumed 2 MB | Sysfs confirms exactly 1,024 KB (1 MB) | Broadcom BCM2711 integrates 1MB L2. Paper claim was incorrect. Documented in Claims Ledger. |
| **E2E Inference Latencies** | AlexNet ~52s, VGG16 ~1847s | Single model latency claimed | Harness summed isolated single-layer benchmark runs | Layer-time aggregate carries overhead of kernel startup and array allocations. Explicitly labeled per Constraint #6. |
