#!/usr/bin/env bash
# ==============================================================================
# Colab Model Station - Automated Non-Interactive Environment Bootstrap
# ==============================================================================
set -e
export DEBIAN_FRONTEND=noninteractive

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

echo "================================================================================"
echo "Colab Model Station - Environment Bootstrap [Mode: $MODE]"
echo "================================================================================"

function install_system_tools() {
    echo "[SETUP] Updating system apt repositories and core utilities..."
    apt-get update -qq > /dev/null 2>&1 || true
    apt-get install -y -qq curl wget git jq psmisc > /dev/null 2>&1 || true
}

function install_tailscale() {
    if ! command -v tailscale > /dev/null 2>&1 || ! command -v tailscaled > /dev/null 2>&1; then
        echo "[SETUP] Installing Tailscale binary..."
        curl -fsSL https://tailscale.com/install.sh | sh > /dev/null 2>&1
        echo "[SETUP] Tailscale installed successfully: $(tailscale version | head -n 1)"
    else
        echo "[SETUP] Tailscale is already installed: $(tailscale version | head -n 1)"
    fi
}

function install_cloudflared() {
    if ! command -v cloudflared > /dev/null 2>&1; then
        echo "[SETUP] Installing cloudflared binary..."
        curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
        dpkg -i /tmp/cloudflared.deb > /dev/null 2>&1
        rm -f /tmp/cloudflared.deb
        echo "[SETUP] cloudflared installed successfully: $(cloudflared --version)"
    else
        echo "[SETUP] cloudflared is already installed: $(cloudflared --version)"
    fi
}

function install_diffusers_deps() {
    echo "[SETUP] Provisioning Python dependencies for Diffusers Engine..."
    pip install -q -r "$DIR/configs/requirements.lock"
    echo "[SETUP] Diffusers dependencies installed."
}

function install_comfyui() {
    COMFY_DIR="/content/ComfyUI"
    if [ ! -d "$COMFY_DIR" ]; then
        echo "[SETUP] Cloning ComfyUI repository into $COMFY_DIR..."
        git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    else
        echo "[SETUP] ComfyUI directory already exists at $COMFY_DIR."
    fi

    echo "[SETUP] Installing ComfyUI dependencies..."
    pip install -q -r "$COMFY_DIR/requirements.txt"
    pip install -q websocket-client
    echo "[SETUP] ComfyUI installation ready."
}

case "$MODE" in
    tailscale)
        install_system_tools
        install_tailscale
        ;;
    diffusers)
        install_system_tools
        install_diffusers_deps
        ;;
    comfyui)
        install_system_tools
        install_comfyui
        ;;
    all)
        install_system_tools
        install_tailscale
        install_cloudflared
        install_diffusers_deps
        ;;
    *)
        echo "Usage: bash scripts/setup.sh [diffusers|comfyui|tailscale|all]"
        exit 1
        ;;
esac

echo "================================================================================"
echo "Colab Model Station - Bootstrap Completed Successfully!"
echo "================================================================================"
