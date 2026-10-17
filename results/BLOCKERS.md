# Blockers and Operational Constraints

This document records all unmet requirements, environment constraints, and degraded evaluation scopes per Part A Rule 5 and Part B.

---

## Active Operational Constraints & Status

| Category | Constraint / Item | Impact / Scope Degradation | Fallback / Mitigation Applied | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Execution Mode** | Local Workstation Dev Mode | Physical board execution cannot run directly in this shell session without SSH target IP/aliases. | Scripts, harness, offline analysis, and test suites are developed, verified with self-test on local environment; orchestrator prepared for SSH deployment. | **MANAGED** |
| **Privileges** | Restricted Root / No Passwordless Sudo | `cpufreq` governor write and `swapoff` may fail if executed without elevated privileges. | Harness checks for root: if unavailable, it queries and logs current active clock frequency per run into `env.json` and CSV rather than aborting. Frequency stability gate (G4) flags drift >2%. | **MANAGED** |
| **RPi Power** | No USB Power Meter Attached | Cannot physically measure energy consumption (Joules/run) on Raspberry Pi 4. | Per Part B3 & Phase 10: RPi energy claims are strictly omitted from all paper tables. Jetson Nano energy is measured via onboard `tegrastats` (VDD_IN). | **SCOPE DEGRADED** |
| **Network** | Internet Available | Dependencies can be fetched online. | Packages downloaded to `third_party/cache/` to ensure idempotent `--offline` capability for future runs. | **RESOLVED** |
| **Workstation** | Workstation Available | TVM auto-tuning host, ONNX model export, and offline analysis can be executed off-board. | Cross-compilation and RPC tuning architecture enabled; lightweight C++ runner on target boards. | **RESOLVED** |

---

## Log of Degraded Claims

1. **Raspberry Pi 4 Energy & MACs/Joule**: Omitted. No external hardware power meter is attached.
2. **Frequency Lock Hard Enforcement**: Degraded to passive frequency logging per run if running without root.
