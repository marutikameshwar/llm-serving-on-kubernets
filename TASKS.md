# TASKS.md — llm-serving-k8s Roadmap & Learning Plan

**Project**: Production LLM Serving on GPU Kubernetes (`llm-serving-k8s`)  
**Target Roles**: AI Infrastructure / Forward Deployed Engineer (Baseten, Fireworks AI, etc.)  
**Pair-Programming Model**: AI plans, explains, and drafts manifests/code; User reviews, learns, executes GPU VM commands, and tests explain-back understanding.

---

## Target Repository Structure

```text
llm-serving-k8s/
├── .agent/
│   ├── rules/
│   │   └── engineering-standards.md
│   └── workflows/
│       └── deployment-flow.md
├── k8s/
│   ├── base/
│   │   ├── namespace.yaml
│   │   ├── hf-secret.template.yaml
│   │   ├── model-cache-pvc.yaml
│   │   └── cuda-vector-add.yaml
│   ├── vllm/
│   │   ├── deployment-fp16.yaml
│   │   ├── deployment-awq.yaml
│   │   ├── deployment-gptq.yaml
│   │   ├── deployment-fp8.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   ├── keda/
│   │   └── scaledobject.yaml
│   └── jobs/
│       ├── guidellm-benchmark.yaml
│       └── lmeval-benchmark.yaml
├── monitoring/
│   ├── prometheus/
│   │   ├── values.yaml
│   │   └── vllm-servicemonitor.yaml
│   ├── grafana/
│   │   └── dashboards/
│   │       └── vllm-dcgm-overview.json
│   └── alerts/
│       └── vllm-alerts.yaml
├── benchmarks/
│   ├── guidellm/
│   │   ├── config-fp16.yaml
│   │   ├── config-awq.yaml
│   │   ├── config-gptq.yaml
│   │   ├── config-fp8.yaml
│   │   └── parse_results.py
│   └── lm-eval/
│       ├── run-eval.sh
│       └── parse_eval.py
├── baseten/
│   └── truss-trtllm/
│       ├── config.yaml
│       └── model/
├── scripts/
│   ├── setup-gpu-vm.sh
│   ├── test-client.py
│   └── analyze_cost.py
├── docs/
│   ├── notes.md
│   ├── versions.md
│   ├── blog-post.md
│   └── postmortems/
│       ├── 01-kv-cache-exhaustion.md
│       ├── 02-gpu-oom-killed.md
│       ├── 03-traffic-spike-recovery.md
│       └── 04-pod-crashloop-debug.md
├── README.md
└── TASKS.md
```

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **GPU Operator on k3s conflicts** | Containerd runtime fails to inject CDI / NVIDIA runtime; driver reinstall collisions. | Set `driver.enabled=false` in GPU Operator Helm values (use VM's preinstalled NVIDIA drivers), configure k3s to use the NVIDIA container runtime via `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl`. |
| **vLLM OOM on 24 GB GPU (L4/A10)** | Qwen2.5-7B weights take ~15 GB in FP16; unconstrained context/batch causes instant CUDA OOM. | Explicitly pin `--gpu-memory-utilization 0.90` and limit context to `--max-model-len 4096`. Benchmark INT4 AWQ to reduce weight footprint to ~5 GB. |
| **Tool version & CLI flag drift** | GuideLLM v0.7+ syntax changes (`guidellm run` with `--target`), vLLM arg deprecations. | Pin exact versions in `docs/versions.md`; consult official documentation before drafting any manifest or command. |
| **GPU VM rental cost runaway** | Forgetting a running cloud GPU VM leads to unexpected billing. | Enforce strict separation: write all manifests/scripts on laptop first; rent VM only for batched execution sessions; stop VM immediately after verification. |

---

## Result Tracking Tables

### 1. Load Test Results (GuideLLM)
*Target: Qwen/Qwen2.5-7B-Instruct on single 24GB GPU (L4/A10)*

| Configuration | Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | ITL Mean (ms) | Output Throughput (tok/s) | GPU Memory (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16** | 1 | — | — | — | — | — |
| **FP16** | 4 | — | — | — | — | — |
| **FP16** | 8 | — | — | — | — | — |
| **FP16** | 16 | — | — | — | — | — |
| **INT4 AWQ** | 1 | — | — | — | — | — |
| **INT4 AWQ** | 4 | — | — | — | — | — |
| **INT4 AWQ** | 8 | — | — | — | — | — |
| **INT4 AWQ** | 16 | — | — | — | — | — |
| **INT4 GPTQ** | 1 | — | — | — | — | — |
| **INT4 GPTQ** | 4 | — | — | — | — | — |
| **INT4 GPTQ** | 8 | — | — | — | — | — |
| **INT4 GPTQ** | 16 | — | — | — | — | — |
| **FP8 (H100)** | 1 | — | — | — | — | — |
| **FP8 (H100)** | 8 | — | — | — | — | — |
| **FP8 (H100)** | 16 | — | — | — | — | — |

### 2. Quantization & Accuracy Comparison
*Evaluating speed, memory footprint, and quality retention across formats*

| Format | Weight Bits | Activation Bits | KV Cache Bits | Peak GPU Mem (GB) | Output tok/s (c=8) | TTFT p95 (ms) | GSM8K Acc (%) | MMLU Subset Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16 Baseline** | 16 | 16 | 16 | — | — | — | — | — |
| **INT4 AWQ** | 4 | 16 | 16 | — | — | — | — | — |
| **INT4 GPTQ** | 4 | 16 | 16 | — | — | — | — | — |
| **FP8 (Weights+Act)** | 8 | 8 | 16 | — | — | — | — | — |
| **FP8 + FP8 KV Cache** | 8 | 8 | 8 | — | — | — | — | — |

### 3. Self-Hosted (k3s + vLLM) vs. Managed (Baseten + TRT-LLM)

| Dimension | Self-Hosted (k3s + vLLM on L4/A10) | Managed Platform (Baseten + TRT-LLM) |
| :--- | :--- | :--- |
| **Inference Engine** | vLLM (PagedAttention) | TensorRT-LLM (In-flight batching) |
| **TTFT p95 (c=8)** | — ms | — ms |
| **Peak Throughput** | — tok/sec | — tok/sec |
| **Cold Start Duration** | — s (image pull + weight load from PVC) | — s (scale from zero) |
| **Cost per 1M Output Tokens** | $— / 1M tokens | $— / 1M tokens |
| **Operational Overhead** | High (K8s cluster, drivers, PVCs, upgrades) | Low (Truss deploy, managed autoscaling) |
| **Customizability** | Complete (kernel flags, custom schedulers) | Platform-bounded |

---

# Milestone 1: v1 Core (Resume-Ready Baseline)

> **Milestone Goal**: Establish a functional, observable, load-tested, and quantized LLM serving deployment on GPU Kubernetes that can immediately anchor an AI Infrastructure resume.
>
> **Execution Strategy (GitOps Batched Run)**: All manifests, scripts, benchmark configs, and dashboards are authored and validated on the laptop first (**Phase 1A**). Once the full codebase is committed and pushed, a single cloud GPU VM is rented to execute all live deployments, load tests, and benchmarks in one focused, cost-controlled run (**Phase 1B**), after which the VM is immediately terminated.

### Resume Checkpoint (v1 Core)
> *"Engineered production LLM serving infrastructure on Kubernetes (k3s) for Qwen2.5-7B using vLLM and NVIDIA GPU Operator; optimized throughput to **[X]** tokens/sec across concurrency levels 1–16 and reduced VRAM by **[Y]%** via 4-bit quantization (comparing INT4 AWQ vs INT4 GPTQ) while monitoring TTFT, ITL, and KV cache saturation in Prometheus/Grafana."*

---

### [Phase 1A: Offline Manifest & Tooling Authoring (Laptop — $0 Cost)]

#### Part 1.1: Local Repository Scaffolding & Version Registry
- [x] **Part 1.1**
  - **Goal**: Establish the repository directory structure, documentation files, and pin all software versions so every component is reproducible and documented.
  - **Learn**:
    - Infrastructure as code dependency pinning: [Twelve-Factor App: Dependencies](https://12factor.net/dependencies)
    - Production Kubernetes folder organization patterns: [Kubernetes Standard Directory Layout](https://kubernetes.io/docs/concepts/overview/working-with-objects/)
  - **Build**:
    - `docs/versions.md` (pinned versions for k3s, GPU Operator, Helm, vLLM, GuideLLM, Prometheus)
    - `docs/notes.md` (template for phase notes and rationale)
    - `.agent/rules/engineering-standards.md`
    - `docs/interview-study-guide.md`
  - **Where**: Laptop
  - **Run**: Git commit and push from laptop. No GPU VM required.
  - **Verify**: Files exist locally, directory tree matches target structure, all versions are recorded with links to release notes.
  - **Explain-back**:
    1. Why is version pinning even more critical in GPU Kubernetes workloads than in standard web microservices?
    2. What specific components in a GPU cluster must maintain strict ABI/driver compatibility?

---

#### Part 1.2: Cluster Bootstrapping Script & GPU Operator Configuration
- [x] **Part 1.2**
  - **Goal**: Write an idempotent bash setup script to install k3s, Helm, and configure the NVIDIA GPU Operator without conflicting with host GPU drivers.
  - **Learn**:
    - NVIDIA GPU Operator Architecture & Driver options: [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
    - Configuring containerd for NVIDIA runtime on k3s: [k3s Advanced Configuration / Containerd](https://docs.k3s.io/advanced#configuring-containerd)
  - **Build**:
    - `scripts/setup-gpu-vm.sh`
    - `k8s/base/namespace.yaml`
    - `k8s/base/cuda-vector-add.yaml` (CUDA test job)
  - **Where**: Laptop
  - **Run**: Review script locally. No GPU VM execution yet.
  - **Verify**: Shellcheck validation passes, k3s containerd template includes nvidia-container-runtime, GPU Operator values specify `driver.enabled=false`.
  - **Explain-back**:
    1. Why do cloud GPU VMs with preinstalled drivers require `driver.enabled=false` in the GPU Operator Helm values?
    2. What is the role of the NVIDIA Container Toolkit and Container Device Interface (CDI) in Kubernetes pod scheduling?

---

#### Part 1.3: Model Cache Storage & Secrets Manifests
- [ ] **Part 1.3**
  - **Goal**: Define a PersistentVolumeClaim and Hugging Face Secret template so model weights are cached on disk across pod restarts and deployments.
  - **Learn**:
    - Kubernetes Storage & k3s `local-path` provisioner: [k3s Storage](https://docs.k3s.io/storage)
    - Hugging Face cache management & `HF_HOME`: [HF Hub Cache System](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
  - **Build**:
    - `k8s/base/hf-secret.template.yaml`
    - `k8s/base/model-cache-pvc.yaml` (50Gi local-path storage)
  - **Where**: Laptop
  - **Run**: Git commit and push to remote.
  - **Verify**: Manifests specify proper access modes (`ReadWriteOnce`), storage class (`local-path`), and secret key references.
  - **Explain-back**:
    1. What failure modes occur if model weights are downloaded to ephemeral container storage instead of a dedicated volume?
    2. How does `HF_HOME` share cached model snapshots between different pods or restart cycles?

---

#### Part 1.4: Serving Manifests — vLLM FP16 Deployment, Service & Test Client
- [ ] **Part 1.4**
  - **Goal**: Create production-grade Kubernetes Deployment and Service manifests for Qwen2.5-7B-Instruct using vLLM with health probes, resource limits, and shared memory, plus a streaming Python test client.
  - **Learn**:
    - vLLM Engine Arguments & Server Configuration: [vLLM Serving Arguments](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
    - PyTorch Shared Memory (`/dev/shm`) in Docker/K8s: [PyTorch IPC Multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)
    - Liveness and Readiness probes for LLM inference engines: [Kubernetes Configure Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
  - **Build**:
    - `k8s/vllm/deployment-fp16.yaml`
    - `k8s/vllm/service.yaml`
    - `scripts/test-client.py` (Python test script with streaming output)
  - **Where**: Laptop
  - **Run**: Code review and git commit.
  - **Verify**: Manifest includes `emptyDir` with `medium: Memory` mounted to `/dev/shm`, startup/readiness probes point to `/health`, limits request 1 GPU and explicit memory/CPU.
  - **Explain-back**:
    1. Why does PyTorch / vLLM require a large `/dev/shm` mount, and what error occurs if it is left at the default 64MB?
    2. Why must we use a `startupProbe` with a generous failure threshold before the `readinessProbe` kicks in for LLM serving?

---

#### Part 1.5: Observability Manifests — Prometheus Stack & Grafana Dashboard
- [ ] **Part 1.5**
  - **Goal**: Prepare Prometheus Helm values, vLLM ServiceMonitor, and a customized Grafana dashboard JSON to capture both LLM engine metrics and DCGM GPU hardware telemetry.
  - **Learn**:
    - vLLM Prometheus metrics specification: [vLLM Metrics Documentation](https://docs.vllm.ai/en/latest/serving/metrics.html)
    - NVIDIA DCGM Exporter metrics: [NVIDIA DCGM Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)
    - Prometheus Operator CRDs (`ServiceMonitor`): [Prometheus Operator Getting Started](https://prometheus-operator.dev/docs/user-guides/getting-started/)
  - **Build**:
    - `monitoring/prometheus/values.yaml` (kube-prometheus-stack Helm values)
    - `monitoring/prometheus/vllm-servicemonitor.yaml`
    - `monitoring/grafana/dashboards/vllm-dcgm-overview.json`
  - **Where**: Laptop
  - **Run**: Validate YAML and JSON syntax, commit and push.
  - **Verify**: ServiceMonitor matches vLLM service labels and port; Grafana dashboard contains panels for TTFT, ITL, KV Cache Saturation, and DCGM VRAM usage.
  - **Explain-back**:
    1. What is the difference between Time to First Token (TTFT) and Inter-Token Latency (ITL), and which Prometheus metrics represent them in vLLM?
    2. Why is raw GPU compute utilization often misleading for LLM workloads compared to memory bandwidth and KV cache usage?

---

#### Part 1.6: Load Testing Manifests — GuideLLM Job, Configs & Results Parser
- [ ] **Part 1.6**
  - **Goal**: Author the GuideLLM Kubernetes Job manifest, benchmark configuration files, and a Python results parser to evaluate throughput and latency across concurrencies 1, 4, 8, and 16.
  - **Learn**:
    - GuideLLM architecture and benchmark methodology: [Neural Magic GuideLLM Docs](https://github.com/vllm-project/guidellm)
    - Poisson arrival processes vs constant concurrency in generative AI benchmarking: [GuideLLM Benchmark Guide](https://github.com/vllm-project/guidellm)
  - **Build**:
    - `benchmarks/guidellm/config-fp16.yaml`
    - `k8s/jobs/guidellm-benchmark.yaml`
    - `benchmarks/guidellm/parse_results.py`
  - **Where**: Laptop
  - **Run**: Verify syntax, commit and push.
  - **Verify**: GuideLLM Job mounts results volume, uses correct `guidellm run --target` CLI syntax, and parser outputs markdown table format.
  - **Explain-back**:
    1. How does increasing concurrency affect TTFT versus ITL under continuous batching?
    2. What causes the "knee in the curve" where throughput saturates and latency spikes exponentially?

---

#### Part 1.7: 4-Bit Quantization Manifests — INT4 AWQ & INT4 GPTQ
- [ ] **Part 1.7**
  - **Goal**: Author Kubernetes Deployment manifests and GuideLLM benchmark configurations for both `Qwen/Qwen2.5-7B-Instruct-AWQ` and `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`.
  - **Learn**:
    - Activation-aware Weight Quantization (AWQ) concepts: [AWQ Paper](https://arxiv.org/abs/2306.00978)
    - Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ): [GPTQ Paper](https://arxiv.org/abs/2210.17323)
    - vLLM Quantization backends & kernel execution (Marlin, AWQ, and GPTQ kernels): [vLLM Quantization Docs](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html)
  - **Build**:
    - `k8s/vllm/deployment-awq.yaml`
    - `k8s/vllm/deployment-gptq.yaml`
    - `benchmarks/guidellm/config-awq.yaml`
    - `benchmarks/guidellm/config-gptq.yaml`
  - **Where**: Laptop
  - **Run**: Git commit and push.
  - **Verify**: Deployments target proper Hugging Face model IDs, configure quantization flags (`--quantization awq` / `--quantization gptq`), and share the same model cache PVC.
  - **Explain-back**:
    1. How do AWQ and GPTQ differ in their mathematical approach to selecting which weights to quantize?
    2. What are Marlin optimized kernels in vLLM, and how do they accelerate 4-bit matrix multiplication?

---

### [Phase 1B: Batched GPU Execution & Live Benchmarking (GPU VM Session)]

#### Part 1.8: GPU VM Genesis & CUDA Smoke Test
- [ ] **Part 1.8**
  - **Goal**: Rent an Ubuntu 22.04 GPU VM (L4/A10), execute `scripts/setup-gpu-vm.sh`, and verify the Kubernetes scheduler can allocate `nvidia.com/gpu: 1` via the CUDA vector-add test.
  - **Where**: GPU VM (Session 1 Start)
  - **Run**: Clone repo, run `bash scripts/setup-gpu-vm.sh`, apply `k8s/base/namespace.yaml` and `k8s/base/cuda-vector-add.yaml`.
  - **Verify**: `kubectl get nodes` reports 1 GPU allocatable; CUDA vector-add pod logs `Test PASSED`.
  - **Explain-back**:
    1. How does the kubelet discover and expose GPU capacity to the Kubernetes API server?

---

#### Part 1.9: Live vLLM FP16 Deployment & Streaming Test
- [ ] **Part 1.9**
  - **Goal**: Apply the model cache PVC, Hugging Face Secret, and vLLM FP16 deployment; verify weight caching and stream first inference tokens via `test-client.py`.
  - **Where**: GPU VM
  - **Run**: Apply base manifests and vLLM FP16 deployment; run `python3 scripts/test-client.py`.
  - **Verify**: Pod reaches `1/1 Running`; test client streams completed response from Qwen2.5-7B.
  - **Explain-back**:
    1. How does PagedAttention eliminate KV cache fragmentation compared to standard Hugging Face Transformers?
    2. What does the vLLM log line "Available KV cache memory" indicate about GPU memory allocation?

---

#### Part 1.10: Live Observability Stack Deployment & Metric Telemetry
- [ ] **Part 1.10**
  - **Goal**: Deploy `kube-prometheus-stack` via Helm, apply the vLLM ServiceMonitor, port-forward Grafana, and confirm live metrics from vLLM and DCGM.
  - **Where**: GPU VM
  - **Run**: Install Prometheus Helm chart with `values.yaml`, apply ServiceMonitor, port-forward Grafana (`3000:80`).
  - **Verify**: Prometheus targets `vllm` and `dcgm-exporter` show `UP (1/1)`; Grafana graphs show live GPU VRAM usage and vLLM request counts.

---

#### Part 1.11: Live GuideLLM Load Testing on FP16 Baseline
- [ ] **Part 1.11**
  - **Goal**: Run GuideLLM Kubernetes Job against vLLM FP16 across concurrencies 1, 4, 8, and 16; parse output and record baseline metrics.
  - **Where**: GPU VM
  - **Run**: Apply `k8s/jobs/guidellm-benchmark.yaml`, parse JSON logs with `parse_results.py`.
  - **Verify**: Table 1 in `TASKS.md` populated with real FP16 TTFT, ITL, and tok/s data.

---

#### Part 1.12: Live 4-Bit Quantization Benchmarking (AWQ & GPTQ)
- [ ] **Part 1.12**
  - **Goal**: Deploy `Qwen2.5-7B-Instruct-AWQ` and `Qwen2.5-7B-Instruct-GPTQ-Int4`, run identical GuideLLM load tests, record VRAM reduction and throughput deltas, then terminate the GPU VM.
  - **Where**: GPU VM (Session 1 Finish)
  - **Run**: Switch deployments, execute GuideLLM jobs for AWQ and GPTQ, record all numbers in Table 1 & Table 2, commit results, and stop/delete cloud VM.
  - **Verify**: All FP16, AWQ, and GPTQ rows in Result Tables populated; GPU VM stopped to eliminate all further rental billing.

---

### [Phase 1C: Synthesis & Portfolio]

#### Part 1.13: v1 Portfolio Artifacts, README & Architecture Diagrams
- [ ] **Part 1.13**
  - **Goal**: Author the v1 README with a professional Mermaid architecture diagram, populated benchmark charts, trade-off analysis, and update phase notes.
  - **Where**: Laptop
  - **Run**: Git commit and push to GitHub.
  - **Verify**: GitHub renders README cleanly with working diagrams, tables, and links; resume bullet is ready for applications.
  - **Explain-back**:
    1. Walk through the life of an inference request from HTTP ingress to GPU tensor execution in this cluster.
    2. How would you justify your choice between FP16, INT4 AWQ, and INT4 GPTQ to an engineering manager based on your latency and throughput data?

---

# Milestone 2: v2 Production Engineering (Reliability & Scaling)

> **Milestone Goal**: Fortify the serving infrastructure with automated accuracy validation, production alert rules, incident postmortems, event-driven autoscaling with KEDA, and cutting-edge FP8 evaluation.

### Resume Checkpoint (v2 Production)
> *"Hardened Kubernetes LLM serving infrastructure with KEDA autoscaling on vLLM queue length, Prometheus alerting for p99 TTFT and KV cache saturation, and authored 4 incident postmortems across simulated GPU failure drills; validated zero accuracy loss on GSM8K/MMLU using lm-evaluation-harness while reducing latency by **[Z]%** via FP8 kernels on NVIDIA Hopper."*

---

### Part 2.1: Accuracy Evaluation with lm-evaluation-harness
- [ ] **Part 2.1**
  - **Goal**: Run `lm-evaluation-harness` against the vLLM OpenAI-compatible endpoint to quantify accuracy retention of INT4 AWQ and INT4 GPTQ versus FP16 on GSM8K and an MMLU subset.
  - **Learn**:
    - Evaluation methodologies for generative models: [lm-evaluation-harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness)
    - Perplexity vs task-specific accuracy metrics in quantized models: [Quantization & Quality Tradeoffs](https://huggingface.co/docs/optimum/concept_guides/quantization)
  - **Build**:
    - `k8s/jobs/lmeval-benchmark.yaml`
    - `benchmarks/lm-eval/run-eval.sh`
    - `benchmarks/lm-eval/parse_eval.py`
  - **Where**: Laptop (draft) -> GPU VM (run)
  - **Run**: Execute lm-eval Job against FP16, AWQ, and GPTQ endpoints, log accuracy scores.
  - **Verify**: Quantization table accuracy columns (GSM8K and MMLU) populated with real evaluation scores for FP16, AWQ, and GPTQ.
  - **Explain-back**:
    1. Why is human eval or synthetic load testing insufficient to approve a quantized model for production deployment?
    2. How did AWQ compare to GPTQ in preserving reasoning capability on GSM8K? Which method had less quality degradation?

---

### Part 2.2: Production Alerting with PrometheusRule
- [ ] **Part 2.2**
  - **Goal**: Implement actionable Prometheus alert rules detecting p99 TTFT spikes (>1.5s), KV cache saturation (>90%), and pod crash loops, complete with runbook annotations.
  - **Learn**:
    - Alerting best practices in Kubernetes: [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
    - PrometheusRule CRD and PromQL histogram quantiles: [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
  - **Build**:
    - `monitoring/alerts/vllm-alerts.yaml`
  - **Where**: Laptop (draft) -> GPU VM (apply)
  - **Run**: Apply `PrometheusRule` manifest to cluster, verify registration in Prometheus Alerts UI.
  - **Verify**: Alerts appear in Prometheus UI in `Inactive` state; syntax passes `promtool check rules`.
  - **Explain-back**:
    1. Why is an alert on KV cache utilization (>90%) more actionable than an alert on raw GPU compute utilization?
    2. How do you construct a PromQL query to calculate the 99th percentile Time To First Token over a 5-minute sliding window?

---

### Part 2.3: Failure Drills & Incident Postmortems
- [ ] **Part 2.3**
  - **Goal**: Deliberately induce real production failures (CUDA OOM, KV cache exhaustion under extreme load, sudden traffic spike, and pod crash loop), debug them via `kubectl` and Grafana, and write professional postmortems.
  - **Learn**:
    - Blameless Postmortem culture and root cause analysis: [Google SRE Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)
    - Kubernetes cgroup OOM Killer vs NVIDIA CUDA Out of Memory: [Debugging Kubernetes Out of Memory](https://kubernetes.io/docs/tasks/administer-cluster/out-of-memory/)
  - **Build**:
    - `docs/postmortems/01-kv-cache-exhaustion.md`
    - `docs/postmortems/02-gpu-oom-killed.md`
    - `docs/postmortems/03-traffic-spike-recovery.md`
    - `docs/postmortems/04-pod-crashloop-debug.md`
  - **Where**: GPU VM (Session 2)
  - **Run**: Trigger each failure scenario systematically, capture error logs and metric snapshots, apply remediation, document findings.
  - **Verify**: All 4 postmortem documents completed with exact reproduction steps, telemetry evidence, root cause, and architectural mitigations.
  - **Explain-back**:
    1. What is the difference between an exit code 137 (OOMKilled by Linux kernel) and a PyTorch `torch.cuda.OutOfMemoryError`?
    2. What happens to incoming inference requests when the vLLM KV cache is 100% full? Does vLLM reject, queue, or preempt them?

---

### Part 2.4: Autoscaling with KEDA on vLLM Queue Metrics
- [ ] **Part 2.4**
  - **Goal**: Deploy KEDA and configure a `ScaledObject` that scales vLLM pod replicas dynamically based on the number of waiting requests in the vLLM scheduling queue.
  - **Learn**:
    - Kubernetes Event-driven Autoscaling (KEDA) architecture: [KEDA Documentation](https://keda.sh/docs/latest/concepts/)
    - Why standard CPU/memory Horizontal Pod Autoscaler (HPA) fails for LLM workloads: [Scaling LLMs on Kubernetes](https://keda.sh/docs/latest/scalers/prometheus/)
  - **Build**:
    - `k8s/keda/scaledobject.yaml` (triggers on `vllm:num_requests_waiting > 5`)
  - **Where**: Laptop (draft) -> GPU VM (deploy)
  - **Run**: Install KEDA via Helm, apply `ScaledObject`, push traffic spike with GuideLLM, observe replica scaling.
  - **Verify**: `kubectl get hpa` and `kubectl describe scaledobject` show active metric tracking; replica count scales up during load burst and scales down during idle.
  - **Explain-back**:
    1. Why is queue depth (`num_requests_waiting`) the single best metric for scaling LLM inference workers compared to CPU or GPU usage?
    2. What is scale-down stabilization (cooldown), and why is it critical when serving multi-gigabyte LLM models?

---

### Part 2.5: FP8 Inference & FP8 KV Cache Evaluation
- [ ] **Part 2.5**
  - **Goal**: Rent an NVIDIA Hopper GPU VM (H100), deploy Qwen2.5-7B with native FP8 (weights + activations) and FP8 KV cache, and benchmark latency, throughput, and accuracy.
  - **Learn**:
    - FP8 formats (E4M3 vs E5M2) and Ada/Hopper Tensor Core architecture: [NVIDIA FP8 Formats for Deep Learning](https://developer.nvidia.com/blog/nvidia-arm-and-intel-unveil-common-fp8-format/)
    - vLLM FP8 KV Cache implementation and memory savings: [vLLM FP8 Documentation](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
  - **Build**:
    - `k8s/vllm/deployment-fp8.yaml`
    - `benchmarks/guidellm/config-fp8.yaml`
  - **Where**: Laptop (draft) -> H100 GPU VM (run)
  - **Run**: Boot H100 VM, run setup script, deploy FP8 manifests, execute GuideLLM benchmark.
  - **Verify**: Quantization table filled with FP8 rows; verify throughput and KV cache capacity gains on Hopper architecture.
  - **Explain-back**:
    1. Why does FP8 require Hopper (or Ada Lovelace) GPU hardware to achieve hardware acceleration compared to FP16 or INT4?
    2. What are the pros and cons of quantizing the KV cache to FP8 versus quantizing model weights?

---

### Part 2.6: v2 Wrap-up & Resume Checkpoint
- [ ] **Part 2.6**
  - **Goal**: Update repository documentation, commit results, ensure all postmortems are linked in README, and update personal notes.
  - **Learn**: Synthesizing complex infrastructure findings into crisp portfolio summaries.
  - **Build**: Update `README.md`, `docs/notes.md`.
  - **Where**: Laptop
  - **Run**: Git commit and push.
  - **Verify**: Git repository reflects all v2 additions; resume checkpoint bullet 2 ready.
  - **Explain-back**:
    1. Summarize the end-to-end reliability story of this cluster: how does it detect, survive, and adapt to production traffic spikes?

---

# Milestone 3: v3 Complete (Managed Cloud Comparison & Final Polish)

> **Milestone Goal**: Deploy the identical model on Baseten using Truss + TensorRT-LLM, run a comprehensive head-to-head comparison against our self-hosted cluster (latency, cost, throughput, operational effort), and publish a polished engineering case study.

### Resume Checkpoint (v3 Complete)
> *"Conducted comparative evaluation of self-hosted vLLM on Kubernetes vs managed Baseten (Truss/TensorRT-LLM); benchmarked cost-per-million-tokens and p95 TTFT, demonstrating a **[A]%** infrastructure cost savings self-hosted at steady high utilization vs **[B]%** lower operational TCO on Baseten for bursty workloads."*

---

### Part 3.1: Package & Deploy on Baseten with Truss + TensorRT-LLM
- [ ] **Part 3.1**
  - **Goal**: Package Qwen2.5-7B using Baseten’s open-source Truss framework with a TensorRT-LLM backend, deploy to Baseten cloud, and verify inference responses.
  - **Learn**:
    - Baseten Truss model serving framework: [Baseten Truss Documentation](https://truss.baseten.co/)
    - TensorRT-LLM compilation and optimization pipeline: [NVIDIA TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
  - **Build**:
    - `baseten/truss-trtllm/config.yaml`
    - `baseten/deploy.py`
  - **Where**: Laptop (Baseten account CLI)
  - **Run**: Deploy model via Baseten CLI / API, monitor build logs, verify active status.
  - **Verify**: Model deployment reaches `Active` on Baseten; streaming query returns valid responses via Baseten REST API.
  - **Explain-back**:
    1. How does Truss encapsulate model dependencies and hardware requirements compared to a raw Kubernetes Deployment manifest?
    2. What optimizations does TensorRT-LLM perform during engine compilation (e.g., kernel fusion, in-flight batching)?

---

### Part 3.2: Comparative Benchmarking: Self-Hosted vs. Baseten
- [ ] **Part 3.2**
  - **Goal**: Execute identical GuideLLM load profiles against both self-hosted k3s vLLM and Baseten endpoints, comparing TTFT, throughput, cost per 1M tokens, and cold-start characteristics.
  - **Learn**:
    - Cloud cost modeling for LLM inference (amortized GPU instance pricing vs per-second/token serverless pricing): [AI Inference Cost Modeling](https://baseten.co/pricing)
    - Benchmarking across public network boundaries vs internal cluster ingress.
  - **Build**:
    - `scripts/analyze_cost.py` (calculates $/1M tokens based on VM hourly rate vs Baseten invocation pricing)
  - **Where**: Laptop
  - **Run**: Run GuideLLM against both endpoints with identical prompt distributions, execute cost calculation script.
  - **Verify**: Table 3 (Self-Hosted vs Managed) fully populated with verified empirical data.
  - **Explain-back**:
    1. At what request volume or utilization threshold does self-hosted GPU Kubernetes become cheaper than a managed platform like Baseten?
    2. How does ingress network overhead impact TTFT measurements when calling cloud-hosted endpoints?

---

### Part 3.3: Production Portfolio Polish & Technical Case Study
- [ ] **Part 3.3**
  - **Goal**: Author an in-depth technical case study / blog post and finalize the repository to be an undeniable portfolio showcase for Forward Deployed Engineer roles.
  - **Learn**:
    - Writing high-impact technical postmortems and system case studies for engineering leadership.
  - **Build**:
    - `docs/blog-post.md` (complete engineering breakdown: problem, architecture, benchmarks, failure drills, cost analysis, recommendation)
    - Final polish of `README.md` (badges, interactive table of contents, clean diagrams, reproducible reproduction steps)
  - **Where**: Laptop
  - **Run**: Git commit and push all final assets to GitHub.
  - **Verify**: Clean repository with zero uncommitted files, complete documentation, reproducible manifests, and verified data tables.
  - **Explain-back**:
    1. Pitch this entire project in 60 seconds as if presenting to a hiring manager at Baseten or Fireworks AI.
