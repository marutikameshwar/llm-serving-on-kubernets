# Engineering Standards for LLM Serving Infrastructure

These engineering principles govern all code, manifests, configurations, and operations in this project:

### 1. No Secrets in Version Control
- Never hardcode API keys, Hugging Face tokens, or credentials in manifests or scripts.
- Use Kubernetes `Secret` objects and environment variable substitution. Provide `.template.yaml` files with dummy values for repository tracking.

### 2. Strict Kubernetes Workload Guardrails
- **Resource Limits & Requests**: Every Pod, Job, and Deployment must declare explicit CPU and Memory `requests` and `limits`, plus `nvidia.com/gpu: 1`. Unconstrained containers can cause node-wide kernel panics or random OOM termination.
- **Shared Memory (`/dev/shm`)**: vLLM and PyTorch multiprocessing require a sized POSIX shared memory mount. Mount an `emptyDir` with `medium: Memory` to `/dev/shm` (recommended >= 4GiB).
- **Health Probes**: Every serving container must declare:
  - `startupProbe`: Allows multi-minute model weight download and CUDA kernel compilation without being killed prematurely.
  - `livenessProbe`: Detects deadlocks or crashed worker processes.
  - `readinessProbe`: Removes traffic from the Service endpoint if the engine is busy initializing or unready.

### 3. Idempotent Infrastructure Scripts
- All automation scripts must be idempotent (safe to run multiple times without duplicating state or failing).
- Use `set -euo pipefail` in all bash scripts to fail immediately upon errors or unset variables.

### 4. Continuous Observability & Metric Integrity
- Services exposing metrics must provide explicit annotations or `ServiceMonitor` definitions.
- Always monitor both **Hardware Telemetry** (DCGM: GPU utilization, VRAM, temperature, throttling) and **Application Telemetry** (vLLM: TTFT, ITL, queue depth, KV cache capacity). Raw GPU compute utilization alone is misleading for LLMs.

### 5. Systematic Failure Documentation
- Whenever an issue or crash occurs (either simulated or accidental), document it following standard SRE Blameless Postmortem format:
  1. Incident Summary & Timeline
  2. Telemetry Evidence (Logs & Grafana snapshots)
  3. Root Cause Analysis (Kernel / Hardware / Application level)
  4. Remediation & Future Prevention
