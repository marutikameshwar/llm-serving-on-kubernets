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

### 5. Kubernetes Core Foundations (The "Why & How")

#### A. Why Kubernetes (k8s) Instead of Raw Docker?
* **Origin**: "k8s" stands for Kubernetes (8 letters between 'k' and 's'), created by Google based on 15+ years of running internal container orchestration (*Borg*).
* **The Production AI Serving Reality**:
  * If you run `docker run vllm...` on a VM, you have a single point of failure. If the Python process crashes due to a CUDA memory spike, Docker leaves it dead.
  * In Kubernetes:
    1. **Self-Healing (Auto-Restart)**: The Kubelet continuously polls container health via probes and immediately restarts crashed workers.
    2. **Zero-Downtime Rolling Deployments**: When swapping models or quantizations, K8s holds user traffic until the 15GB model is fully loaded in VRAM, eliminating downtime.
    3. **Hardware-Aware Scheduling**: In a multi-GPU cluster, Kubernetes scans all nodes, checks available VRAM and allocatable `nvidia.com/gpu` capacity, and places the pod on the optimal node.
    4. **Event-Driven Autoscaling (KEDA)**: Scales inference replicas up or down based on incoming request queue length.

#### B. What is `kind: Namespace` and Why is it in a Separate YAML?
* **What it is**: A Namespace is a **virtual cluster / logical boundary** inside Kubernetes (like a folder on a disk). By default, Kubernetes has `kube-system` (cluster internal components) and `default`. We create `llm-serving` so all our inference pods, storage volumes, secrets, and services are logically isolated.
* **Why Separate YAML? (The Lifecycle Decoupling Rule)**:
  * A Namespace is created **once** when the cluster is provisioned; it lives permanently.
  * Workloads (like test Jobs or Deployments) are created, edited, and deleted **dozens of times**.
  * Mixing a Namespace into a workload YAML violates separation of concerns: deleting the Job could accidentally tear down the entire namespace and kill unrelated serving pods!

#### C. How Does Kubernetes Communicate with the Node? (Brain vs. Muscle)
* **Crucial Rule**: A Namespace **never** communicates with a node! A Namespace is purely a database record in the API server.
* **The Control Plane (The Brain)**: Runs `kube-apiserver`, `kube-scheduler`, and `kube-controller-manager`. It stores desired state in SQLite/etcd.
* **The Worker Node (The Muscle)**: Has physical CPUs, RAM, and GPUs. Runs the **`kubelet`** daemon and **containerd**.
* **Communication Flow**:
  1. Engineer sends YAML to `kube-apiserver` over HTTPS.
  2. `kube-scheduler` assigns the Pod to a physical node.
  3. The `kubelet` on that node constantly watches the API server via a secure TLS connection.
  4. When the `kubelet` sees a Pod assigned to its node, it orders `containerd` to pull the image and launch the container with the GPU attached.

#### D. Why is There NO `spec:` in `namespace.yaml`?
* `spec` stands for **Specification** — it defines the technical parameters of an **active, executing workload** (container image, memory limits, ports, environment variables).
* A **Namespace is not a program**. It does not run code, consume CPU, or open network ports. It is an administrative label/boundary.
* Therefore, all a Namespace needs is an identity (`metadata.name: llm-serving`). There is literally nothing to specify in a `spec`!

---

### 6. Deep Dive into the `spec` of `cuda-vector-add.yaml`

A Kubernetes Job has a **two-layer nested spec**:
1. **Outer `spec`**: Configuration for the **Job Controller** (how the overall batch execution is managed).
2. **Inner `spec` (`template.spec`)**: Configuration for the **Pod and Container** (what runs on the physical GPU).

```text
Job: cuda-vector-add
│
├── Outer `spec` (Job Controller)
│     ├── ttlSecondsAfterFinished: 600   ──► Automated cleanup after 10 min
│     ├── backoffLimit: 2                ──► Retry count on failure
│     └── template                       ──► The Pod "cookie-cutter"
│
└── Inner `spec` (`template.spec` - Pod & Container)
      ├── restartPolicy: OnFailure       ──► Only restart if crashed
      └── containers[0]
            ├── image: nvcr.io/...       ──► Official NVIDIA CUDA binary
            └── resources
                  ├── requests           ──► Scheduling reservation floor
                  └── limits             ──► Kernel cgroup enforcement ceiling
```

#### Detailed Breakdown of Every Field in the Job `spec`:

##### 1. `ttlSecondsAfterFinished: 600` (Automated Garbage Collection)
* `ttl` = Time To Live.
* In Kubernetes, when a Pod inside a Job completes, it enters the `Completed` state and stays there forever so engineers can read logs.
* In high-throughput CI/CD or benchmarking, leaving thousands of completed pods consumes API server memory (etcd storage).
* `ttlSecondsAfterFinished: 600` activates the Kubernetes **TTL Controller**: exactly 600 seconds (10 minutes) after the job finishes, Kubernetes automatically cascades and deletes the Job and its Pods, keeping the cluster clean.

##### 2. `backoffLimit: 2` (Exponential Failure Backoff)
* If the container fails (e.g., driver missing, bad CUDA call), Kubernetes doesn't hammer the node with hundreds of restarts.
* It retries at most **2 times** using an **exponential backoff delay** (10s, 20s, 40s...).
* If the job fails on the second retry, the Job status switches to `Failed`, and the controller stops trying.

##### 3. `template.spec.restartPolicy: OnFailure`
* In a Kubernetes **Deployment**, `restartPolicy` defaults to `Always` (if the process stops, always restart it).
* In a **Job**, `restartPolicy: Always` is **illegal** and rejected by the API server schema because a Job is designed to stop!
* `OnFailure` tells Kubernetes: *"If the container exits with code 0 (success), leave it terminated. Only restart if exit code $\neq$ 0."*

##### 4. The `resources` Deep Dive (Linux Cgroups & CFS Scheduler)
How does Kubernetes enforce CPU and Memory at the Linux OS level?

| Field | Declared Value | Linux Kernel Mechanism | What Happens Under the Hood |
| :--- | :--- | :--- | :--- |
| **`requests.cpu`** | `"500m"` (500 millicores) | `cpu.shares` | Guaranteed CPU time. 500m = 0.5 of a physical CPU thread. If other processes compete for CPU, this container is guaranteed at least 50% of one core. |
| **`limits.cpu`** | `"1"` (1000 millicores) | `cpu.cfs_quota_us` / `cpu.cfs_period_us` | Hard ceiling via the Linux CFS (Completely Fair Scheduler). If the container tries to consume more than 100% of 1 core, the kernel throttles its execution cycles. |
| **`requests.memory`** | `512Mi` | Scheduling filter | Informs kube-scheduler: *"Do not place this pod on a node unless it has at least 512MiB of unreserved host RAM."* |
| **`limits.memory`** | `1Gi` | `memory.max` (cgroups v2) | Hard ceiling. If the container allocates 1025 MiB, the Linux kernel triggers **OOM-Killer (Out of Memory)**, kills the container instantly with **Exit Code 137**, and records an `OOMKilled` event. |
| **`requests.nvidia.com/gpu`** | `1` | Extended Resource filter | Scheduler verifies node has `status.allocatable['nvidia.com/gpu'] >= 1`. |
| **`limits.nvidia.com/gpu`** | `1` | NVIDIA Device Plugin & CDI | Containerd calls NVIDIA Container Toolkit to mount `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`. The container is isolated to only GPU 0. |

##### 5. What Does the CUDA Vector Addition Program Actually Test?
The image `nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04` executes a classic CUDA verification algorithm:
1. Allocates memory on host RAM for two arrays of 50,000 float numbers: $A$ and $B$.
2. Uses `cudaMalloc()` to allocate memory on GPU VRAM ($d\_A$, $d\_B$, $d\_C$).
3. Copies arrays from host RAM to GPU VRAM across the **PCIe bus** (`cudaMemcpyHostToDevice`).
4. Launches a CUDA kernel with multiple thread blocks to compute $C[i] = A[i] + B[i]$ in parallel across physical GPU streaming multiprocessors (SMs).
5. Copies results back to host RAM (`cudaMemcpyDeviceToHost`).
6. Compares GPU results against CPU addition to confirm arithmetic precision.
7. Prints `Test PASSED`.

> **Why this matters**: If this test passes, you have mathematically verified that:
> - The host kernel driver is working.
> - The PCIe bus communication is operational.
> - The container toolkit successfully injected device nodes.
> - The CUDA runtime successfully compiled and executed GPU kernels.

---

### 7. End-to-End Execution Flow (What Happens on `kubectl apply`)
```
1. Engineer runs `kubectl apply -f cuda-vector-add.yaml`
                     │
2. Kube-API Server validates syntax and records Job object
                     │
3. Job Controller detects Job and creates Pod matching `template`
                     │
4. Kube-Scheduler inspects Pod `requests: { nvidia.com/gpu: 1 }`
   Checks nodes: finds node with 1 allocatable GPU -> Binds Pod to Node
                     │
5. Kubelet on GPU Node calls NVIDIA Device Plugin via gRPC (`Allocate()`)
   Device plugin returns device node: `/dev/nvidia0`
                     │
6. Containerd launches container using `nvidia-container-runtime`
   Mounts `/dev/nvidia0` and links host `libcuda.so.1`
                     │
7. C++ VectorAdd binary runs on GPU CUDA cores
   Verifies math -> Prints "Test PASSED" -> Exits with code 0
                     │
8. Job Controller marks Job status as `Completed`
```

---

---

### 8. Distributed AI Infrastructure: Multi-GPU Training & Serving Architecture

In AI Infrastructure and Forward Deployed Engineering interviews, a premier evaluation question is:  
> *"Llama-3-70B requires 140 GB in FP16, but our largest GPU (H100) has only 80 GB of VRAM. How do you partition and scale this model across multiple GPUs?"*

Here is the comprehensive architectural breakdown.

---

#### A. The Hardware Bottleneck: The "Memory Wall"
* A 70 Billion parameter model in **FP16** (16 bits = 2 bytes/param) requires:
  $$\text{Model Weights} = 70 \times 10^9 \times 2\text{ bytes} \approx \mathbf{140\text{ GB}}$$
* The largest single commercial GPU (NVIDIA H100) has **80 GB of HBM3 VRAM**.
* **$140\text{ GB} > 80\text{ GB}$**: A single GPU physically cannot hold the model weights for inference.
* **During Training, the VRAM Footprint Explodes**:
  In addition to weights, training requires storing:
  1. **Gradients**: $140\text{ GB}$ (2 bytes/param)
  2. **Adam Optimizer States**: $560\text{ GB}$ (8 bytes/param for FP32 master weights, momentum, and variance)
  3. **Activation Memory**: Hundreds of gigabytes depending on sequence length and batch size.
  * *Total VRAM required to train a 70B model*: **$\mathbf{> 1,000\text{ GB}}$ (Over 1 Terabyte of VRAM)**!

To overcome this, AI systems engineers deploy **Distributed Parallelism**.

---

#### B. The 4 Fundamental Parallelism Paradigms

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DISTRIBUTED PARALLELISM STRATEGIES                   │
├──────────────────────────┬─────────────────────────────────────────────┤
│ 1. Data Parallelism (DDP)│ Clone model on all GPUs, split data batch   │
│ 2. Tensor Parallelism(TP)│ Slice individual weight matrices within layer│
│ 3. Pipeline Parall. (PP) │ Split layers sequentially across GPUs       │
│ 4. FSDP / DeepSpeed ZeRO │ Shard weights, gradients, optimizer states  │
└──────────────────────────┴─────────────────────────────────────────────┘
```

##### 1. Data Parallelism (DDP) — *"Clone the Model, Split the Batch"*
* **When to use**: When the model fits inside a single GPU, but you have huge datasets and want to train faster.
* **Mechanism**:
  * An identical copy of the full model is loaded onto each GPU.
  * Each GPU receives a distinct micro-batch of training data.
  * After calculating the forward and backward passes, the GPUs communicate via **NCCL (NVIDIA Collective Communications Library)** using an **All-Reduce** operation to average gradients across all GPUs before updating weights.

##### 2. Tensor Parallelism (TP) — *"Slice Each Matrix Across GPUs (Intra-Layer)"*
* **When to use**: When individual weight matrices are too large for one GPU, or for **ultra-low-latency serving** (the core technique in **vLLM** and **TensorRT-LLM**).
* **Mechanism (Megatron-LM Style)**:
  * In a Transformer, attention and MLP layers are matrix multiplications: $Y = X \times W$.
  * If weight matrix $W$ has 16,000 columns, we slice it across 2 GPUs:
    * **GPU 0**: Holds Columns $1 \dots 8,000$ ($W_1$). Computes $Y_1 = X \times W_1$.
    * **GPU 1**: Holds Columns $8,001 \dots 16,000$ ($W_2$). Computes $Y_2 = X \times W_2$.
  * An **All-Reduce** operation sums or concatenates the partial outputs to reconstruct $Y$.
* **Hardware Requirement**: Extremely high inter-GPU bandwidth. Tensor Parallelism **must run over NVLink** (up to 900 GB/s on H100) within the same physical node. Running TP across Ethernet or InfiniBand between different nodes is too slow and stalls the GPU compute cores.
* **Serving with vLLM**:
  ```bash
  vllm serve meta-llama/Llama-3-70B --tensor-parallel-size 4
  ```

##### 3. Pipeline Parallelism (PP) — *"The Factory Assembly Line (Inter-Layer)"*
* **When to use**: When a model has dozens of layers (e.g. 80 Transformer blocks) that exceed a single node's total memory.
* **Mechanism**:
  * Layers are partitioned sequentially across GPUs:
    * **GPU 0**: Layers $1 \dots 20$
    * **GPU 1**: Layers $21 \dots 40$
    * **GPU 2**: Layers $41 \dots 60$
    * **GPU 3**: Layers $61 \dots 80$
  * The prompt enters GPU 0 $\rightarrow$ activations are transmitted over network $\rightarrow$ GPU 1 $\rightarrow$ GPU 2 $\rightarrow$ GPU 3 produces the final logits.
* **The "Bubble" Problem**: In a naive implementation, GPU 3 sits completely idle waiting for Layers 1–60 to finish. Modern systems use **1F1B (One Forward, One Backward)** micro-batch scheduling to interleave forward and backward passes and minimize idle time.

##### 4. FSDP & DeepSpeed ZeRO — *"The Modern Standard: Shard Everything"*
* **Origin**: Pioneered by Microsoft (**DeepSpeed ZeRO**: Zero Redundancy Optimizer) and standardized in PyTorch as **FSDP (Fully Sharded Data Parallel)**.
* **Mechanism**:
  * Standard DDP duplicates weights, gradients, and optimizer states across every GPU, wasting memory.
  * **ZeRO Stage 1**: Shards the 8 bytes/param Adam optimizer states across all GPUs (cuts memory by $4\times$).
  * **ZeRO Stage 2**: Shards optimizer states + gradients.
  * **ZeRO Stage 3 / FSDP**: **Shards everything!** Weights, gradients, and optimizer states are split across all GPUs.
  * *Just-In-Time Parameter Prefetching*: When GPU 0 needs to compute Layer 5, it executes an **All-Gather** to fetch the missing parameters from peer GPUs, computes the layer, and **instantly discards them from VRAM** before moving to Layer 6!

---

#### C. Comparison Matrix for Technical Interviews

| Parallelism Scheme | What is Partitioned? | Primary Use Case | Interconnect Requirement | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Data Parallelism (DDP)** | Input Data Batches | High-throughput training of smaller models | Standard PCIe / 10GbE | Gradient All-Reduce time at large cluster scale |
| **Tensor Parallelism (TP)** | Weight Matrices (within layers) | Low-latency inference (vLLM) & large model training | **NVLink (Intra-Node only: 900 GB/s)** | High communication frequency per layer |
| **Pipeline Parallelism (PP)**| Transformer Layers (inter-layer) | Splitting multi-layer models across physical servers | InfiniBand / RoCE (Inter-Node) | Pipeline bubble (idle GPU wait states) |
| **FSDP / ZeRO-3** | Weights + Gradients + Optimizer States | State-of-the-art LLM Training (Llama 3, DeepSeek) | High-Speed InfiniBand (400 Gbps) | All-Gather network latency during forward/backward |

---

#### D. How Kubernetes Manages Multi-GPU Clusters
1. **Multi-GPU Pod Scheduling**:
   * A single Pod can request all GPUs on an 8-GPU server:
     ```yaml
     resources:
       limits:
         nvidia.com/gpu: 8   # Schedules onto an 8x H100 node with NVLink mesh
     ```
2. **Kubernetes Multi-Node Training Operators**:
   * **KubeRay / RayJob**: Deploys a Ray cluster across multiple GPU nodes for distributed inference and fine-tuning.
   * **Kubeflow Training Operator (`PyTorchJob`)**: Automates distributed master/worker pods, configuring `RANK`, `WORLD_SIZE`, and `MASTER_ADDR` environment variables for PyTorch distributed backends.
3. **High-Performance Networking in K8s**:
   * Production GPU Kubernetes clusters deploy the **NVIDIA Network Operator** alongside the GPU Operator to enable **GPUDirect RDMA** (Remote Direct Memory Access), allowing GPUs on different servers to transfer memory directly over InfiniBand without bouncing through host CPU or RAM.

---

### 9. Master Learning Syllabus & Video Library

To master both standard Kubernetes and GPU AI infrastructure, use this curated reference library:

#### Track A: Standard Kubernetes Mastery
* **[Kubernetes Tutorial for Beginners — TechWorld with Nana (YouTube)](https://www.youtube.com/watch?v=X48VuDVv0do)**: Visual, end-to-end breakdown of Pods, Deployments, Services, and Ingress.
* **[Kubernetes Concepts & Architecture (Official Docs)](https://kubernetes.io/docs/concepts/architecture/)**: Master reference for control plane and worker node components.
* **[iximiuz Labs (Ivan Velichko's Blog)](https://iximiuz.com/en/categories/?category=kubernetes)**: Hand-drawn architectural illustrations of containers, cgroups, and namespaces.
* **[Killercoda Kubernetes Playground](https://killercoda.com/playgrounds/scenario/kubernetes)**: Free in-browser interactive Kubernetes terminal.
* **[Kubernetes the Hard Way by Kelsey Hightower](https://github.com/kelseyhightower/kubernetes-the-hard-way)**: Step-by-step manual cluster construction for deep systems understanding.

#### Track B: GPU in Containers & AI Serving Specialization
* **[How to Use NVIDIA GPUs with Docker & Containers — Christian Lempa (YouTube)](https://www.youtube.com/watch?v=k_l3FzB070M)**: 15-minute hands-on guide to host drivers, NVIDIA Container Toolkit, and `/dev/nvidia*` mounts.
* **[Demystifying GPU Support in Kubernetes — KubeCon + CloudNativeCon (YouTube)](https://www.youtube.com/watch?v=i9B3pX1G9Q4)**: Deep architectural conference talk explaining Kubelet device plugin gRPC calls and extended resources.
* **[NVIDIA GPU Operator on Kubernetes — NVIDIA Developer (YouTube)](https://www.youtube.com/watch?v=0k5iZ_t077g)**: Walkthrough of Helm-based GPU Operator deployment.
* **[vLLM & PagedAttention — Sky Computing Lab / Woosuk Kwon (YouTube)](https://www.youtube.com/watch?v=5ZlavKF_98U)**: The creator of vLLM explains PagedAttention, KV caching, and continuous batching.
* **[Eugene Yan's Guide to LLM Serving](https://eugeneyan.com/writing/llm-serving/)**: Essential industry reading on TTFT, ITL, memory bottlenecks, and quantization.
* **[Baseten Engineering Blog](https://www.baseten.co/blog/)**: Case studies on model cold-starts, TensorRT-LLM optimization, and cloud GPU economics.
* **[NVIDIA Container Toolkit Architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html)**: Official specification on CDI (Container Device Interface) and runtime injection.

---

### 10. Kubernetes Storage Architecture & Pod Lifecycle (Why PVCs are Mandatory in AI Serving)

In modern cloud-native systems, Pods are treated as **ephemeral ("cattle, not pets")**—they are designed to be stopped, evicted, upgraded, and destroyed automatically. 

Understanding how storage and secrets work during pod lifecycle events is essential for AI Infrastructure interviews.

---

#### A. The "Hotel Room" Ephemeral Storage Problem
* Standard container images (`vllm/vllm-openai`) are composed of immutable, **read-only layers**.
* At container startup, the container runtime (containerd) attaches a temporary, read-write layer on top via **OverlayFS**.
* When vLLM downloads the 15.2 GB of Qwen2.5-7B weights from Hugging Face, they are written to this temporary layer.
* **The Fatal Flaw**: When a pod terminates or crashes, containerd destroys that temporary read-write layer. **All 15.2 GB of downloaded model weights are wiped out instantly.**

---

#### B. The 5 Real-World Scenarios Where Pods are Destroyed & Recreated

1. **Traffic Spikes & Out-of-Memory Crashes (`OOMKilled`)**:
   * Multiple concurrent users submit large context prompts (e.g. 8,000 tokens). The attention KV cache exhausts VRAM or host memory. The Linux kernel sends `SIGKILL` (**Exit Code 137**).
   * Kubernetes immediately cleans up the dead pod and creates a replacement pod.
2. **Configuration Updates & Rolling Deployments**:
   * An engineer changes a flag in `deployment.yaml` (e.g. `--max-model-len 8192` or testing a new quantization scheme).
   * Kubernetes executes a **Rolling Update**, provisioning a new pod and terminating the old one.
3. **Dynamic Autoscaling (Scaling Up & Down)**:
   * During low nighttime traffic, KEDA scales replicas down from 3 pods to 1 pod (destroying Pods 2 and 3).
   * In the morning rush, KEDA scales back up from 1 pod to 3 pods (creating Pods 2 and 3).
4. **Cloud Preemptible / Spot GPU Instance Reclamation**:
   * To save 60%–80% on compute bills, AI clusters run on Spot GPU instances (AWS Spot, RunPod Spot). When cloud providers reclaim a node, pods are evicted and rescheduled onto another available node.
5. **Kubelet DiskPressure Evictions**:
   * If a container downloads 15 GB directly into ephemeral storage, it writes directly to the host OS root filesystem (`/var/lib/containerd`). If disk usage crosses 85%, Kubelet enters `DiskPressure` state and forcibly **evicts (kills)** the largest pods to protect the host OS from crashing.

---

#### C. The Solution: The Kubernetes Storage Trinity

To decouple model weight storage from the fragile container lifecycle, Kubernetes uses three decoupled abstractions:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. StorageClass (The Storage Policy / Provisioner)                     │
│    • In k3s: `local-path`. Tells K8s to carve space out of the host's  │
│      local high-speed NVMe SSD.                                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Dynamically Provisions
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. PersistentVolume - PV (The Physical Disk Directory)                 │
│    • A permanent host directory: `/var/lib/rancher/k3s/storage/pvc-...`│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Bound to
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. PersistentVolumeClaim - PVC (The Storage Claim / Request)           │
│    • Declares: "Request 50Gi of ReadWriteOnce storage."                │
│    • Mounted inside the vLLM pod at `/root/.cache/huggingface`         │
└────────────────────────────────────────────────────────────────────────┘
```

##### Impact Comparison: With vs. Without PVC

| Operational Event | Without PVC (Amateur Setup) | With PVC (Production AI Infra) |
| :--- | :--- | :--- |
| **Pod OOM Crash Recovery** | 12 minutes downtime (re-download 15 GB) | **15 seconds recovery (local disk cache hit)** |
| **YAML Config Update** | 12 minutes delay per rollout | **Instant rolling update with zero user downtime** |
| **Morning Scale-Up** | New pods lag by 15 minutes before ready | **New pods ready to serve traffic in 20 seconds** |
| **Hugging Face API Limits**| Risk of HTTP 429 Too Many Requests | **Zero downloads needed after initial cache** |
| **Cloud Bandwidth Costs** | Gigabytes of repeated egress bandwidth | **$0 wasted network bandwidth** |

---

#### D. The Secrets Paradigm: Why `kind: Secret` Matters

* **The Security Threat**: Hugging Face API tokens (`hf_...`) allow access to model weights and gated repos. Automated bot networks continuously scan public GitHub commits in real time (within 2 seconds of pushing). Hardcoding a secret in a git-tracked YAML leads to immediate account compromise and API quota exhaustion.
* **The Kubernetes Fix**:
  * Passwords and tokens are stored in an encrypted/isolated Kubernetes object: `kind: Secret`.
  * The Deployment YAML references the secret via `valueFrom.secretKeyRef` without ever exposing the raw string.
  * At container boot, Kubernetes mounts the secret as an in-memory environment variable (`HF_TOKEN`) inside the isolated container Linux namespace.
* **The GitOps Template Pattern**:
  * We commit `hf-secret.template.yaml` with placeholder strings (`PASTE_YOUR_HF_TOKEN_HERE`).
  * On the actual VM, engineers copy this to `hf-secret.yaml` (which is excluded by `.gitignore`) and apply it. Version control remains 100% clean and audit-compliant.

---

### 11. Manifest Deep Dive: `model-cache-pvc.yaml` & `hf-secret.template.yaml` (Part 1.3)

---

#### Manifest A: `k8s/base/model-cache-pvc.yaml`
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
  namespace: llm-serving
  labels:
    app.kubernetes.io/name: model-cache
    app.kubernetes.io/part-of: llm-serving-k8s
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-path
  resources:
    requests:
      storage: 50Gi
```

##### Line-by-Line Breakdown:
* `apiVersion: v1`: Storage claims are core Kubernetes resources in the `v1` API group.
* `kind: PersistentVolumeClaim`: An authorization ticket/request for storage. Decouples the pod from the underlying cloud hardware (whether local SSD, AWS EBS, or Ceph).
* `metadata.name: model-cache-pvc`: The handle that our vLLM deployment will reference in its `volumes` array.
* `metadata.namespace: llm-serving`: Bound to our isolated project namespace.
* `spec.accessModes: [ReadWriteOnce]`:
  * **ReadWriteOnce (RWO)**: The volume can be mounted with read-write permissions by **a single Kubernetes node** at a time.
  * *Why RWO?* Our single GPU VM runs all pods on that one machine. Local NVMe SSD storage is inherently node-local and offers the maximum sequential read speed (3,000–7,000 MB/s), which is crucial for fast model weight loading.
  * *(Interview Note)*: If we had a 100-node cluster sharing a centralized weight cache, we would use **ReadWriteMany (RWX)** via a distributed file system like NFS, CephFS, or AWS EFS.
* `spec.storageClassName: local-path`:
  * Directs Kubernetes to use k3s’s built-in **Rancher Local Path Provisioner**.
  * Dynamically creates a persistent host directory at `/var/lib/rancher/k3s/storage/` on the host VM's NVMe drive.
* `spec.resources.requests.storage: 50Gi`:
  * Carves out **50 Gibibytes** ($50 \times 1024^3$ bytes).
  * *Storage Budget*:
    * Qwen2.5-7B FP16: **~15.2 GB**
    * Qwen2.5-7B INT4 AWQ: **~5.3 GB**
    * Qwen2.5-7B INT4 GPTQ: **~5.2 GB**
    * Total for all 3 models = **~25.7 GB**, leaving $> 24\text{ GB}$ of buffer for tokenizer configs, temporary staging files, and OS overhead.

---

#### Manifest B: `k8s/base/hf-secret.template.yaml`
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-secret
  namespace: llm-serving
  labels:
    app.kubernetes.io/name: hf-secret
    app.kubernetes.io/part-of: llm-serving-k8s
type: Opaque
stringData:
  HF_TOKEN: "PASTE_YOUR_HUGGING_FACE_TOKEN_HERE"
```

##### Line-by-Line Breakdown:
* `kind: Secret`: Stores sensitive authentication data securely in the Kubernetes etcd/SQLite database rather than plaintext manifest configs.
* `type: Opaque`: The standard, default Kubernetes secret type for arbitrary user-defined key-value strings (passwords, tokens, API keys).
* `stringData`:
  * Standard Kubernetes secrets use `data:`, which **requires base64-encoded strings** (e.g. `echo -n "my-token" | base64`).
  * `stringData` allows writing the token in **plain human-readable text** in the local file. When you run `kubectl apply`, Kubernetes automatically base64-encodes the string before writing it to the cluster! This eliminates human copy-paste errors during token creation.

---

### 12. Interview Questions & Model Answers (Part 1.3)

#### Q1: "What failure modes occur if model weights are downloaded to ephemeral container storage instead of a dedicated PersistentVolume?"
> **Model Answer**:
> *"If model weights are written to ephemeral container storage, they live in the container's temporary OverlayFS read-write layer. Any lifecycle event that terminates the container—such as an Out-of-Memory (OOM) kill, a rolling deployment update, a node preemption, or autoscaling down—destroys that layer permanently. When the replacement pod boots, it must re-download the 15GB model over the public internet, inflating cold-start times from 15 seconds to 12 minutes, causing HTTP 504 gateway timeouts for users, and risking API rate-limiting from Hugging Face."*

#### Q2: "How does `HF_HOME` share cached model snapshots between different pods or restart cycles?"
> **Model Answer**:
> *"Hugging Face Hub uses a content-addressable cache layout governed by the `HF_HOME` environment variable. Under `$HF_HOME/hub/models--<org>--<repo>/`, files are deduplicated by SHA-256 hash in a `blobs/` directory, while `snapshots/<commit-hash>/` contains symlinks to those blobs. By mounting our 50GB PersistentVolume to `/root/.cache/huggingface` and pointing `HF_HOME` to that mount path, vLLM checks existing blob hashes at boot. If the snapshot is complete, it skips network downloads entirely and immediately memory-maps the weights from disk into GPU VRAM."*

#### Q3: "What is the difference between ReadWriteOnce (RWO) and ReadWriteMany (RWX), and when would each be used in LLM serving?"
> **Model Answer**:
> *"ReadWriteOnce (RWO) means the volume can be mounted with read-write permissions by only a single Kubernetes node at a time, whereas ReadWriteMany (RWX) allows simultaneous read-write mounts across multiple nodes. For single-node serving or node-local storage (like k3s `local-path` on an NVMe SSD), RWO is preferred because local disks maximize sequential read throughput (several gigabytes per second) directly over the PCIe bus without network hop latency. In large multi-node clusters where 50 inference replicas must share a single model cache, RWX via a distributed network file system (like NFS, AWS EFS, or CephFS) is used so models are downloaded once for the entire fleet."*

---

*(Next up: Part 1.4 — Serving Manifests: vLLM FP16 Deployment, Service & Test Client [Laptop])*





