# Project Engineering Notes & Decision Log

This document tracks technical decisions, architecture rationales, and milestone notes across each stage of building `llm-serving-k8s`.

---

## Part 1.1: Local Repository Scaffolding & Version Registry
- **Date**: 2026-10-30
- **Summary**: Established the initial directory layout, version registry (`docs/versions.md`), engineering standards (`.agent/rules/engineering-standards.md`), and the master interview study guide (`docs/interview-study-guide.md`).
- **Key Decision**:
  - Pinned k3s to `v1.30.4+k3s1` and vLLM to `v0.6.3.post1` with CUDA 12.4. In GPU environments, unpinned versions lead to CUDA Driver API vs Runtime API mismatches that silently crash tensor kernels.
  - Selected `Qwen/Qwen2.5-7B-Instruct` as the primary evaluation target because its 7B parameter size fits squarely on a cost-effective 24GB GPU (L4 or A10) while providing top-tier coding and reasoning benchmarks.

---

## Part 1.2: Cluster Bootstrapping Script & GPU Operator Configuration
- **Date**: 2026-10-30
- **Summary**: Created the automated VM bootstrap script (`scripts/setup-gpu-vm.sh`), base namespace manifest (`k8s/base/namespace.yaml`), and CUDA verification Job (`k8s/base/cuda-vector-add.yaml`).
- **Key Decisions**:
  - Configured k3s containerd using a template file (`/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl`) to inject `nvidia-container-runtime`. k3s regenerates its containerd configuration on startup, so raw edits to `config.toml` get overwritten; using the `.tmpl` file guarantees persistence.
  - Set `driver.enabled=false` and `toolkit.enabled=false` in the GPU Operator Helm values. Cloud VMs come with vendor-tuned NVIDIA drivers pre-baked into the kernel. Enabling the operator driver component would attempt to overwrite host kernel modules, causing kernel panics or module load conflicts.
  - Enabled `dcgmExporter.enabled=true` in the GPU Operator to establish hardware telemetry for Prometheus right from cluster genesis.

---

## Architectural & Workflow Decision: Transition to Pure GitOps / Batched GPU Execution
- **Date**: 2026-10-30
- **Summary**: Restructured Milestone 1 into two decoupled phases:
  - **Phase 1A (Offline Authoring on Laptop)**: Writing, validating, and committing all manifests, Helm values, Python test clients, and benchmark jobs locally without spinning up a GPU VM ($0 cloud spend).
  - **Phase 1B (Batched GPU Execution Session)**: Provisioning a single cloud GPU VM (L4/A10) to run through cluster bootstrapping, model caching, FP16 inference, Prometheus/Grafana observability, and GuideLLM load testing for FP16, INT4 AWQ, and INT4 GPTQ in one focused, sequential session.
- **Rationale**:
  1. **Cost Discipline**: Cloud GPU instances incur per-minute billing. Pausing to write YAML, inspect schemas, or debug Python test scripts on a running cloud GPU VM wastes significant money.
  2. **Production GitOps Emulation**: Real engineering teams do not author manifests interactively on live production clusters; they author, review, lint, and push all configuration declaratively to version control first.
  3. **Deterministic Runbooks**: Developing all manifests upfront allows the GPU execution session to follow a strictly ordered, error-checked runbook. The VM is booted, executes the workloads, populates the empirical benchmark tables, and is immediately terminated.

