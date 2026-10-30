# Software Version Registry

To ensure full reproducibility and avoid the notorious "works on my machine" or subtle ABI/CUDA driver mismatch errors, all software, charts, and base images in this repository are strictly pinned.

| Component | Pinned Version | Purpose / Source | Notes & Release Links |
| :--- | :--- | :--- | :--- |
| **Operating System** | `Ubuntu 22.04 LTS` | Host GPU Virtual Machine OS | Linux Kernel 5.15 / 6.5 (Standard LTS cloud image) |
| **NVIDIA Host Driver** | `>= 535.183` (or `550.x`) | Host kernel graphics & compute driver | Must be preinstalled on cloud VM; provides `libcuda.so` |
| **CUDA Toolkit / Runtime** | `12.4` | Compute engine target ABI | Compatible with vLLM 0.6.x PyTorch 2.4+ builds |
| **k3s (Lightweight K8s)** | `v1.30.4+k3s1` | Cloud-native container orchestrator | [k3s Releases](https://github.com/k3s-io/k3s/releases) — light footprint, native containerd |
| **Helm** | `v3.15.4` | Kubernetes package manager | [Helm Releases](https://github.com/helm/helm/releases) |
| **NVIDIA GPU Operator** | `v24.6.2` | Manages GPU device plugin & DCGM | [GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) (`driver.enabled=false`) |
| **vLLM Container** | `vllm/vllm-openai:v0.6.3.post1` | Inference serving engine with OpenAI API | [vLLM Releases](https://github.com/vllm-project/vllm/releases) — PagedAttention & Marlin kernels |
| **GuideLLM** | `guidellm:v0.7.0` | Load generation & concurrency benchmark | [Neural Magic GuideLLM](https://github.com/vllm-project/guidellm) — uses `guidellm run --target` syntax |
| **kube-prometheus-stack** | `61.3.2` | Prometheus + Alertmanager + Grafana | Helm chart from `prometheus-community` |
| **KEDA** | `v2.15.1` | Event-driven pod autoscaler | [KEDA Documentation](https://keda.sh/) — scales on vLLM queue length |
| **lm-evaluation-harness**| `v0.4.4` | Accuracy benchmarking (GSM8K, MMLU) | [EleutherAI lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) |
| **Baseten Truss** | `truss >= 0.9.30` | Model packaging for TRT-LLM | [Truss Documentation](https://truss.baseten.co/) |

---

### Target Models (Hugging Face)

| Precision / Format | Hugging Face Model ID | Context Length | Weight Footprint |
| :--- | :--- | :--- | :--- |
| **FP16 Baseline** | `Qwen/Qwen2.5-7B-Instruct` | 4,096 tokens (pinned) | ~15.2 GB |
| **INT4 AWQ** | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 4,096 tokens (pinned) | ~5.3 GB |
| **INT4 GPTQ** | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | 4,096 tokens (pinned) | ~5.2 GB |
| **FP8 (Hopper)** | `neuralmagic/Qwen2.5-7B-Instruct-FP8` | 4,096 tokens (pinned) | ~7.8 GB |

