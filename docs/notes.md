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

