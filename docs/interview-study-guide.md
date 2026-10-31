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

### 2. Foundational Principles & Official References (Why They Matter in Interviews)

#### Principle A: The Twelve-Factor App — Rule II: Explicit Dependencies
* **Official Reference**: [12factor.net/dependencies](https://12factor.net/dependencies)
* **What it means**: An application or infrastructure stack must explicitly declare and pin all dependencies; it must never rely on implicit system-wide packages or `:latest` floating tags.
* **Why it matters in AI Infrastructure**:
  * In web engineering, pulling `:latest` might break an API response. In AI infrastructure, pulling `vllm/vllm-openai:latest` can silently download an image built against CUDA 12.6 when your host driver only supports CUDA 12.2. The container will crash at runtime with `CUDA initialization error: no kernel image is available for execution on the device`.
  * By maintaining an explicit version registry (`docs/versions.md`), our cluster deployment is 100% deterministic, immutable, and reproducible.

#### Principle B: Kubernetes Directory Architecture & Object Separation
* **Official Reference**: [Kubernetes Concepts: Working with Objects](https://kubernetes.io/docs/concepts/overview/working-with-objects/)
* **What it means**: Structuring manifests hierarchically so that static foundational layers (namespaces, storage classes, secrets) are decoupled from dynamic workloads (deployments, jobs) and telemetry.
* **Why it matters in AI Infrastructure**:
  * LLM serving involves heavy persistent storage (multi-gigabyte weights) and dedicated device plugins. Dumping everything into a single YAML or random files causes race conditions where a pod starts before its PersistentVolumeClaim or driver daemon is ready.
  * Our repo layout (`k8s/base/`, `k8s/vllm/`, `monitoring/`, `benchmarks/`) mirrors production enterprise GitOps repositories.

---

### 3. What Each Project File Does & Why It Exists

| File Path | Role in the Project | Why an Interviewer Cares |
| :--- | :--- | :--- |
| **`docs/versions.md`** | The Single Source of Truth for pinned software versions (Ubuntu, k3s, vLLM, CUDA, GPU Operator, GuideLLM). | Demonstrates dependency discipline and deep awareness of CUDA driver/runtime compatibility constraints. |
| **`.agent/rules/engineering-standards.md`** | Production rulebook enforcing no secrets in git, `/dev/shm` sizing, health probes, and resource requests/limits. | Proves you know the common failure modes of containerized deep learning (e.g., PyTorch shared memory bus errors, unprobed pod traffic). |
| **`docs/notes.md`** | Chronological engineering decision log capturing trade-offs at each step. | Shows structured engineering thinking: why we picked specific architectures, models, and tools rather than just copying tutorials. |
| **`docs/interview-study-guide.md`** | Living pre-interview study guide with deep architectural breakdowns, hardware math, and interview Q&As. | Serves as your personal revision sheet so you can explain any system component with high clarity and authority. |

---

### 4. Core Concepts Explained Simply

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

### 5. Interview Questions & Model Answers

#### Q1: "Why is version pinning more critical in GPU Kubernetes workloads than in standard CPU microservices?"
> **Model Answer**:
> *"In CPU microservices, applications run inside standard user-space runtimes (like Node.js or Python) that are completely isolated by Linux namespaces and cgroups. In GPU workloads, the container must punch through the container boundary to talk to host hardware. There is a hard ABI (Application Binary Interface) boundary between the host's NVIDIA kernel module and user-space CUDA libraries. If the container's CUDA runtime version exceeds what the host driver supports, or if the GPU Operator installs a conflicting driver on a cloud VM with pre-existing drivers, the GPU device plugin will fail and pods will remain stuck in `Pending` with `0/1 nodes available: Insufficient nvidia.com/gpu`."*

#### Q2: "What is the difference between the CUDA Driver API and the CUDA Runtime API?"
> **Model Answer**:
> *"The CUDA Driver API (provided by `libcuda.so` on the host system) is the low-level interface installed with the NVIDIA hardware driver. The CUDA Runtime API (provided by `libcudart.so` typically packaged inside the application container) is a higher-level API that compiles against PyTorch or vLLM. The key rule of GPU infrastructure is: **Driver API version $\ge$ Runtime API version**. You can run an older CUDA runtime inside a container on a newer host driver, but you can never run a newer CUDA runtime on an older host driver."*

---

## Part 1.2: Cluster Bootstrapping Script & GPU Operator Configuration

### 1. What Did We Build & Why?
We created three foundational artifacts on our laptop before spinning up the paid GPU VM:
1. **`scripts/setup-gpu-vm.sh`**: An automated, idempotent bash script that installs k3s, Helm, the NVIDIA Container Toolkit, and the NVIDIA GPU Operator.
2. **`k8s/base/namespace.yaml`**: The isolated `llm-serving` Kubernetes namespace for our workloads.
3. **`k8s/base/cuda-vector-add.yaml`**: A GPU smoke-test Kubernetes Job that runs tensor vector addition to prove hardware scheduling works before we touch vLLM.

**The Engineering Problem Solved**:
Manually running `apt install` and `helm install` commands directly on a cloud VM is error-prone, untracked, and non-reproducible. More importantly, cloud GPU instances (like RunPod, Lambda Labs, GCP, AWS) come with proprietary NVIDIA drivers pre-installed in the Linux kernel. A naive installation of the NVIDIA GPU Operator attempts to compile and overwrite host drivers, causing kernel panics or device plugin failure. Our automated script solves this by configuring k3s containerd to use the host drivers cleanly.

---

### 2. How Did We Build It? (Step-by-Step Anatomy of `setup-gpu-vm.sh`)

Here is what happens under the hood when this script runs:

```
[ Step 1: nvidia-smi Check ]
       │  Verifies host driver is active and GPU is detected on PCIe bus
       ▼
[ Step 2: NVIDIA Container Toolkit ]
       │  Installs nvidia-ctk and generates CDI (Container Device Interface) spec
       ▼
[ Step 3: k3s Containerd Template ]
       │  Writes /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl
       │  Injects [plugins.cri.containerd.runtimes.nvidia] runtime binary
       ▼
[ Step 4: Install k3s Lightweight K8s ]
       │  Disables default Traefik/ServiceLB to save CPU/RAM for inference
       ▼
[ Step 5: Install Helm 3 ]
       │  Standard package manager for Kubernetes operators
       ▼
[ Step 6: Deploy NVIDIA GPU Operator via Helm ]
       │  --set driver.enabled=false    (Do NOT overwrite host driver)
       │  --set toolkit.enabled=false   (Use host containerd template)
       │  --set devicePlugin.enabled=true (Exposes nvidia.com/gpu to Kubelet)
       │  --set dcgmExporter.enabled=true (Exposes GPU metrics to Prometheus)
       ▼
[ Result: Node Allocatable nvidia.com/gpu: 1 ]
```

---

### 3. Deep Dive Concepts (Interview Essentials)

#### Concept A: Why `config.toml.tmpl` instead of editing `config.toml` directly?
In standard Kubernetes (kubeadm), containerd is managed via `/etc/containerd/config.toml`.  
In **k3s**, containerd is embedded and self-managed. Every time k3s boots or restarts, it **completely regenerates** `/var/lib/rancher/k3s/agent/etc/containerd/config.toml` from internal Go templates.  
If you edit `config.toml` directly, k3s will overwrite your changes on the next restart, wiping out the NVIDIA runtime!  
*The Solution*: k3s looks for a template file named `config.toml.tmpl`. If present, k3s executes this template. By defining `nvidia-container-runtime` in `config.toml.tmpl`, our GPU runtime survives reboots and upgrades permanently.

#### Concept B: Why `driver.enabled=false` in the GPU Operator?
* The NVIDIA GPU Operator was originally designed for bare-metal enterprise clusters where the host OS has NO drivers installed. In that mode, the operator deploys a DaemonSet that compiles kernel modules (`nvidia.ko`) against the host Linux kernel.
* On **cloud GPU VMs** (AWS, GCP, RunPod, Lambda), the cloud hypervisor and base image already have driver modules loaded into the running kernel.
* If you run the GPU Operator with `driver.enabled=true`, the operator will attempt to load its own kernel modules, causing an immediate kernel crash or module busy lock. Setting `driver.enabled=false` instructs the operator to skip driver installation and hook directly into the existing host driver.

#### Concept C: How Kubernetes Discovers GPUs (The Device Plugin API)
Standard Kubernetes only understands two hardware resources out of the box: **CPU** (`cpu`) and **Memory** (`memory`). Kubernetes does not natively understand GPUs!  
To support GPUs, Kubernetes uses **Extended Resources** via the **Device Plugin Framework**:
1. The GPU Operator runs the `nvidia-device-plugin` DaemonSet on the node.
2. The device plugin calls NVML (NVIDIA Management Library) to discover physical GPUs on the machine.
3. The device plugin registers with the local node's `kubelet` over a Unix domain socket (`/var/lib/kubelet/device-plugins/kubelet.sock`).
4. It announces: *"I have 1 healthy `nvidia.com/gpu` available."*
5. The kubelet updates its node status (`status.allocatable['nvidia.com/gpu'] = 1`), sending this capacity to the Kubernetes API server.
6. Now, any pod with `resources.limits: { nvidia.com/gpu: 1 }` can be scheduled to that node!

#### Concept D: Why Can't the GPU Driver Live Inside Docker? (Kernel vs. User Space)
In web development, the classic Docker promise is: *"Everything is containerized—the host OS doesn't matter!"*  
In GPU AI infrastructure, this is **physically impossible**.
* **Containers are NOT Virtual Machines**: Containers do not run their own OS kernel; they are isolated processes sharing the host Linux kernel.
* **Kernel Space (Ring 0) vs. User Space (Ring 3)**:
  * Talking to physical GPU hardware across the PCIe bus requires kernel-level direct memory access (DMA) and hardware interrupts. This is handled exclusively by **NVIDIA Kernel Modules** (`nvidia.ko`, `nvidia-uvm.ko`) running in the host kernel.
  * Containers live in user space. A container cannot load or compile kernel modules into the host kernel without destroying container security boundaries.
* **The Division of Labor**:
  ```text
  ┌──────────────────────────────────────────────────────────────┐
  │  INSIDE THE DOCKER CONTAINER (User Space)                    │
  │  • Python (vLLM, PyTorch, Transformers)                      │
  │  • CUDA Runtime (libcudart.so) & Math Libs (cuBLAS, cuDNN)   │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ Talks via mounted /dev/nvidia0
  ═══════════════════════════════╪════════════════════════════════ Container Boundary
                                 │
  ┌──────────────────────────────┴───────────────────────────────┐
  │  ON THE HOST LINUX VM (Kernel Space)                         │
  │  • Host Linux Kernel                                         │
  │  • NVIDIA Kernel Modules (nvidia.ko, nvidia-uvm.ko)          │
  │  • CUDA Driver Library (libcuda.so.1)                        │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ Physical PCIe Bus
  ┌──────────────────────────────┴───────────────────────────────┐
  │  PHYSICAL HARDWARE: NVIDIA L4 / A10 / H100 Silicon & VRAM   │
  └──────────────────────────────────────────────────────────────┘
  ```
* **The Bridge**: The **NVIDIA Container Toolkit** acts as the bridge. At container startup, it hooks into containerd, mounts the character devices (`/dev/nvidia*`), and links `libcuda.so` directly into the container.

#### Concept E: Why Bootstrap with an `.sh` Script instead of Manual Terminal Commands?
1. **Cost Efficiency & Rapid Recreation**: Cloud GPU instances are rented by the hour. When we shut down a VM to save money, spinning up a new one tomorrow takes **1 command** (`bash scripts/setup-gpu-vm.sh`) in 3 minutes, rather than 45 minutes of manual error-prone copy-pasting.
2. **Fail-Fast Safety (`set -euo pipefail`)**: If an `apt` package or network curl fails in a manual terminal session, subsequent commands run blindly on a broken machine. With `set -e`, execution halts immediately at the first failure.
3. **Executable Documentation for Interviews**: Interviewers looking at your repository see clean, version-controlled Infrastructure-as-Code proving you know how to automate containerd and GPU drivers.

---

### 4. Interview Questions & Model Answers

#### Q1: "How does Kubernetes schedule and allocate a physical GPU to a specific container?"
> **Model Answer**:
> *"Kubernetes uses the Device Plugin framework for extended resources. The NVIDIA Device Plugin runs as a DaemonSet on GPU nodes, scans the PCIe bus via NVML, and reports allocatable `nvidia.com/gpu` counts to the kubelet. When a pod requests `nvidia.com/gpu: 1`, the Kubernetes kube-scheduler matches the pod to a node with available GPU capacity and decrements the allocatable count. The kubelet then invokes the device plugin's `Allocate()` gRPC method, which returns the environment variables and host device paths (e.g. `/dev/nvidia0`). The container runtime (containerd with NVIDIA Container Toolkit) uses these device paths to mount the physical GPU into the container's Linux cgroup and namespace."*

#### Q2: "What is the Container Device Interface (CDI) and why is the industry moving to it from legacy hooks?"
> **Model Answer**:
> *"Legacy GPU container injection relied on pre-start hooks in Docker/containerd that inspected container environment variables (like `NVIDIA_VISIBLE_DEVICES`) and mutated container specs on the fly. This was brittle and tightly coupled to specific runtime implementations. CDI (Container Device Interface) is an open specification from the Container Orchestrated Device (COD) working group that standardizes how third-party devices (GPUs, FPGAs, TPUs) are described via static YAML files under `/etc/cdi/`. Runtimes like containerd and CRI-O can read CDI specs directly, eliminating proprietary hacks and enabling cleaner, more secure multi-device scheduling."*

#### Q3: "Why must the NVIDIA GPU driver be installed on the host operating system rather than containerized inside Docker?"
> **Model Answer**:
> *"Containers share the host operating system kernel and run in unprivileged user space. Interfacing with physical GPU silicon across the PCIe bus requires memory-mapped I/O and interrupt handling, which can only be performed in kernel space (Ring 0) by the NVIDIA kernel modules (`nvidia.ko` and `nvidia-uvm.ko`). Because containers cannot load kernel modules into the host kernel, the driver must reside on the host. The container only packages the high-level CUDA runtime and application frameworks, while the NVIDIA Container Toolkit dynamically mounts the host's device nodes (`/dev/nvidia*`) and driver library (`libcuda.so`) into the container at start time."*

#### Q4: "How did you manage host bootstrapping and node repeatability for your GPU cluster?"
> **Model Answer**:
> *"I automated the host configuration using an idempotent bash bootstrapping script pinned with `set -euo pipefail`. Because cloud GPU VMs come with proprietary kernel drivers preinstalled, the script automatically generates a k3s containerd template for the `nvidia-container-runtime`, installs k3s without redundant ingress components, and uses Helm to deploy the NVIDIA GPU Operator with `driver.enabled=false`. This allowed me to recreate the entire GPU cluster from scratch in under 3 minutes whenever a new VM was provisioned."*

---

*(Next up: Part 1.3 — GPU Cluster Bootstrapping & Validation on Cloud VM)*


