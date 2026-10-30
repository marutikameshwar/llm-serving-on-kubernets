#!/usr/bin/env bash
# ==============================================================================
# setup-gpu-vm.sh
# Bootstraps k3s, Helm, NVIDIA Container Toolkit, and NVIDIA GPU Operator
# Target OS: Ubuntu 22.04 LTS on Cloud GPU VM (L4, A10, H100)
# ==============================================================================

set -euo pipefail

# Pinned Software Versions (matching docs/versions.md)
K3S_VERSION="v1.30.4+k3s1"
HELM_VERSION="v3.15.4"
GPU_OPERATOR_VERSION="v24.6.2"

echo "=========================================================="
echo " Starting GPU Kubernetes Node Bootstrapping"
echo " Target k3s version: ${K3S_VERSION}"
echo " Target Helm version: ${HELM_VERSION}"
echo " Target GPU Operator: ${GPU_OPERATOR_VERSION}"
echo "=========================================================="

# ------------------------------------------------------------------------------
# 1. Host Sanity Checks
# ------------------------------------------------------------------------------
echo "==> [Step 1/6] Checking host NVIDIA driver and GPU visibility..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This script requires preinstalled NVIDIA host drivers."
    echo "Ensure your cloud VM was provisioned with NVIDIA drivers."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "Host NVIDIA driver is operational."

# ------------------------------------------------------------------------------
# 2. Install Host Dependencies & NVIDIA Container Toolkit
# ------------------------------------------------------------------------------
echo "==> [Step 2/6] Installing prerequisites & NVIDIA Container Toolkit..."
sudo apt-get update -y
sudo apt-get install -y curl git jq ca-certificates apt-transport-https

# Add NVIDIA Container Toolkit repository if not already installed
if ! command -v nvidia-ctk &> /dev/null; then
    echo "Installing nvidia-container-toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
else
    echo "nvidia-container-toolkit is already installed."
fi

# Configure Container Device Interface (CDI) support
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml || true

# ------------------------------------------------------------------------------
# 3. Configure k3s Containerd Template for NVIDIA Runtime
# ------------------------------------------------------------------------------
echo "==> [Step 3/6] Setting up containerd config template for k3s..."
sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd/

# k3s allows overriding containerd configuration via config.toml.tmpl.
# We inject the nvidia-container-runtime so pods requesting GPUs receive them.
sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl > /dev/null << 'EOF'
{{ template "base" . }}

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia"]
  runtime_type = "io.containerd.runc.v2"
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia".options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
EOF

# ------------------------------------------------------------------------------
# 4. Install k3s
# ------------------------------------------------------------------------------
echo "==> [Step 4/6] Installing k3s (${K3S_VERSION})..."
if ! command -v k3s &> /dev/null; then
    curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="${K3S_VERSION}" K3S_KUBECONFIG_MODE="644" sh -s - \
      --write-kubeconfig-mode 644 \
      --disable traefik \
      --disable servicelb
else
    echo "k3s is already installed. Restarting service to pick up containerd changes..."
    sudo systemctl restart k3s
fi

# Set up user kubeconfig
mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
export KUBECONFIG="$HOME/.kube/config"

echo "Waiting for k3s node to reach Ready state..."
kubectl wait --for=condition=Ready node --all --timeout=60s

# ------------------------------------------------------------------------------
# 5. Install Helm
# ------------------------------------------------------------------------------
echo "==> [Step 5/6] Installing Helm (${HELM_VERSION})..."
if ! command -v helm &> /dev/null; then
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | DESIRED_VERSION="${HELM_VERSION}" bash
else
    echo "Helm is already installed: $(helm version --short)"
fi

# ------------------------------------------------------------------------------
# 6. Install NVIDIA GPU Operator via Helm
# ------------------------------------------------------------------------------
echo "==> [Step 6/6] Deploying NVIDIA GPU Operator (${GPU_OPERATOR_VERSION})..."
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update

# Key Configuration Rationale:
# - driver.enabled=false: Use preinstalled host NVIDIA drivers.
# - toolkit.enabled=false: Use host-installed nvidia-container-toolkit via containerd template.
# - dcgmExporter.enabled=true: Expose GPU hardware telemetry for Prometheus.
# - devicePlugin.enabled=true: Expose nvidia.com/gpu resources to Kubernetes.
helm upgrade --install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --version "${GPU_OPERATOR_VERSION}" \
  --set driver.enabled=false \
  --set toolkit.enabled=false \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --wait --timeout=300s

echo "=========================================================="
echo " Cluster Setup Complete!"
echo " Checking allocatable GPU resources on node:"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU_ALLOCATABLE:.status.allocatable.'nvidia\.com/gpu'
echo "=========================================================="
echo "You can now run:"
echo "  kubectl apply -f k8s/base/namespace.yaml"
echo "  kubectl apply -f k8s/base/cuda-vector-add.yaml"
