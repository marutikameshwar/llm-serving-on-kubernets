# Master Interview Study Guide: Production LLM Serving on GPU Kubernetes

> **How to use this file**: This document is your comprehensive, pre-interview study guide. It is updated at every single step of building this project. Before going into an AI Infrastructure or Forward Deployed Engineer (FDE) interview, review the core concepts, mental models, and explain-back questions in this guide.

---

# Architecture & Project Vision at a Glance

### The 30-Second Interview Pitch
> *"I designed and built an end-to-end, production-grade LLM serving platform on GPU Kubernetes (k3s) hosting Qwen2.5-7B with vLLM. It features full observability using Prometheus, Grafana, and NVIDIA DCGM exporter, event-driven autoscaling via KEDA on vLLM queue depth, and rigorous benchmarking across precision formats—comparing FP16 baseline against INT4 AWQ, INT4 GPTQ, and FP8 on Hopper. In addition to measuring latency and throughput with GuideLLM, I validated task accuracy retention using `lm-evaluation-harness` on GSM8K and MMLU, conducted deliberate failure drills with postmortems, and performed an economic cost analysis comparing self-hosted Kubernetes vs. managed Baseten (Truss/TensorRT-LLM)."*

---

# Milestone 1: v1 Core (The Serving Foundation)

## Part 1.1: Local Repository Scaffolding & Version Registry

### 1. What Did We Build & Why?
We established the repository architecture, pinned exact software dependencies in `docs/versions.md`, defined strict cloud-native engineering rules in `.agent/rules/engineering-standards.md`, and created this living study guide.

**The Engineering Problem Solved**:
In cloud-native web microservices, version mismatches usually cause minor feature discrepancies. In **GPU AI Infrastructure**, version drift causes total, hard-to-debug system failure:
- A mismatched PyTorch and CUDA runtime version fails to invoke Tensor Cores.
- A GPU Operator version incompatible with the host NVIDIA kernel driver fails to expose `nvidia.com/gpu` to the kubelet.
- GuideLLM or vLLM CLI flag changes between minor versions break automated CI/CD jobs.

By pinning every version upfront in an IaC registry, we make our infrastructure deterministic and reproducible.

---

### 2. Core Concepts Explained Simply

#### Concept A: The GPU Software Stack (The "Three-Layer Cake")
When you run an LLM on Kubernetes, there are three distinct layers that must cooperate:
1. **The Host Layer (Bare-Metal / VM Kernel)**:
   - Contains the **NVIDIA Kernel Driver** (`nvidia.ko`, `nvidia-uvm.ko`) and the CUDA Driver library (`libcuda.so.1`).
   - This speaks directly to the PCIe bus and the physical GPU silicon.
2. **The Container Engine & Orchestrator Layer (k3s & Containerd)**:
   - Containerd must know how to pass GPU devices (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) into the container.
   - It uses the **NVIDIA Container Toolkit** and the **Container Device Interface (CDI)** to hook into the container lifecycle.
3. **The Application / Container Layer (vLLM & PyTorch)**:
   - Inside the container, PyTorch and vLLM link against the **CUDA Runtime** (`libcudart.so`) and math libraries (`cuBLAS`, `cuDNN`).
   
> **Interview Intuition**: The CUDA Driver API is backwards-compatible, but the CUDA Runtime API is not. If your container's CUDA Runtime is newer than your host VM's NVIDIA Driver, the container crashes instantly with `CUDA driver version is insufficient for CUDA runtime version`.

#### Concept B: Why k3s for Single-Node GPU Serving?
- Full Kubernetes (kubeadm / EKS / GKE) has extensive control plane overhead (etcd, multiple control plane processes) requiring several gigabytes of host RAM.
- **k3s** compiles the Kubernetes control plane into a single binary, replaces heavy etcd with SQLite for single-node deployments, and has a footprint under 512MB RAM.
- It is 100% compliant Kubernetes: every Pod, Deployment, Service, PVC, and CRD we write on k3s runs identically on enterprise AWS EKS or GCP GKE.

#### Concept C: Why Qwen2.5-7B-Instruct?
- Parameter size: **7.61 Billion parameters**.
- In **FP16** (16 bits = 2 bytes per parameter), the raw weights alone consume:
  $$\text{Weight VRAM} = 7.61 \times 10^9 \times 2 \text{ bytes} \approx 15.22\text{ GB}$$
- On an affordable cloud GPU with **24 GB VRAM** (such as NVIDIA L4 or A10):
  - 15.2 GB is taken by model weights.
  - ~1.5 GB is taken by PyTorch runtime and CUDA context overhead.
  - Leaves **~7.3 GB for the KV cache** (where active conversational memory and concurrent requests live).
- This makes 7B the sweet spot for demonstrating both baseline performance and the dramatic impact of quantization (which will shrink those weights to ~5.2 GB).

---

### 3. Interview Questions & Model Answers

#### Q1: "Why is version pinning more critical in GPU Kubernetes workloads than in standard CPU microservices?"
> **Model Answer**:
> *"In CPU microservices, applications run inside standard user-space runtimes (like Node.js or Python) that are completely isolated by Linux namespaces and cgroups. In GPU workloads, the container must punch through the container boundary to talk to host hardware. There is a hard ABI (Application Binary Interface) boundary between the host's NVIDIA kernel module and user-space CUDA libraries. If the container's CUDA runtime version exceeds what the host driver supports, or if the GPU Operator installs a conflicting driver on a cloud VM with pre-existing drivers, the GPU device plugin will fail and pods will remain stuck in `Pending` with `0/1 nodes available: Insufficient nvidia.com/gpu`."*

#### Q2: "What is the difference between the CUDA Driver API and the CUDA Runtime API?"
> **Model Answer**:
> *"The CUDA Driver API (provided by `libcuda.so` on the host system) is the low-level interface installed with the NVIDIA hardware driver. The CUDA Runtime API (provided by `libcudart.so` typically packaged inside the application container) is a higher-level API that compiles against PyTorch or vLLM. The key rule of GPU infrastructure is: **Driver API version $\ge$ Runtime API version**. You can run an older CUDA runtime inside a container on a newer host driver, but you can never run a newer CUDA runtime on an older host driver."*

---

*(Next up: Part 1.2 — Cluster Bootstrapping Script & GPU Operator Configuration)*
