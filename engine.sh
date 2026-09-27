#!/usr/bin/env bash
# ==============================================================================
# Colab Model Station - Master Engine & Tunnel Lifecycle Controller
# Supports Universal Multi-Engine (vLLM, Ollama, Diffusers, ComfyUI)
# and Tri-Networking (Tailscale Mesh, Cloudflare, VS Code Remote Tunnel)
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$DIR/configs"
SCRIPTS_DIR="$DIR/scripts"
ENGINES_DIR="$DIR/engines"
TOOLS_DIR="$DIR/tools"
LOG_DIR="$DIR/logs"

mkdir -p "$LOG_DIR"
cd "$DIR"

if [ -f "$DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$DIR/.env"
    set +a
fi

function detect_hardware() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
        GPU_MEM="${GPU_MEM:-0}"
    else
        GPU_NAME="None (CPU Mode)"
        GPU_MEM="0"
    fi

    echo "=== Hardware Profile ==="
    echo "Detected GPU: $GPU_NAME (${GPU_MEM} MB VRAM)"

    if [[ "$GPU_NAME" =~ "A100" ]] || [[ "$GPU_NAME" =~ "H100" ]] || [[ "$GPU_MEM" -gt 35000 ]]; then
        echo "Recommendation: Enterprise GPU (A100). Suitable for vLLM (BF16/AWQ 70B) & FLUX.1 unquantized (BF16)."
    elif [[ "$GPU_NAME" =~ "L4" ]] || [[ "$GPU_MEM" -gt 20000 ]]; then
        echo "Recommendation: High-tier GPU (L4). Suitable for vLLM (14B/32B AWQ), Ollama (32B), & FLUX.1 FP8 (Model Offload)."
    elif [[ "$GPU_NAME" =~ "T4" ]] || [[ "$GPU_MEM" -gt 12000 ]]; then
        echo "Recommendation: Standard GPU (T4). Suitable for Ollama (7B/14B Q4), vLLM (7B AWQ), & FLUX.1 FP8 (Sequential Offload) or SDXL."
    else
        echo "Recommendation: CPU mode detected. Use mock mode or lightweight models for testing."
    fi
}

function show_help() {
    echo "================================================================================"
    echo "Colab Model Station - Unified Multi-Engine & Tunnel CLI"
    echo "================================================================================"
    echo "Usage: ./engine.sh [command] [options...]"
    echo ""
    echo "General System:"
    echo "  status                           Show comprehensive system health and all engines"
    echo "  list                             Display catalog of supported LLM & Diffusion models"
    echo "  setup [mode]                     Bootstrap dependencies (diffusers|comfyui|vllm|ollama|tunnels|all)"
    echo "  teardown                         Terminate all services and unassign Colab VM"
    echo ""
    echo "Diffusion Engines (Images & LoRAs):"
    echo "  diffusers start [--model ID] [--precision fp8|fp16|bf16] [--mock]"
    echo "  diffusers stop | status | logs [-f]"
    echo "  comfyui start [--port 8188] [--vram auto|low|high]"
    echo "  comfyui stop | status | logs [-f]"
    echo ""
    echo "LLM Engines (Text, Code & OpenAI API):"
    echo "  vllm start [--model ID] [--max-len INT] [--gpu-util FLOAT] [--quantization awq|auto]"
    echo "  vllm stop | status | logs [-f] | chat [model] | bench [model]"
    echo "  ollama start | stop | status | logs [-f] | list | pull <model> | chat [model] | bench [model]"
    echo ""
    echo "Network & Remote IDE Tunnels:"
    echo "  tunnel tailscale up [authkey]    Connect to Tailscale mesh (Userspace mode)"
    echo "  tunnel tailscale down | status | serve [port]"
    echo "  tunnel cloudflare up [port] | down"
    echo "  tunnel vscode [setup|login|start [name]|status|stop]"
    echo ""
    echo "Storage, Cache & Safeguards:"
    echo "  cache prefetch <model_repo_id>   Accelerated download via HF Transfer"
    echo "  cache lora <source_url_or_repo>  Download LoRA weights to local registry"
    echo "  cache stats                      Show storage consumption"
    echo "  watchdog start [--timeout SECS]  Start automated compute unit guardian"
    echo "  watchdog stop | status"
    echo "================================================================================"
}

CMD="${1:-help}"

case "$CMD" in
    setup)
        bash "$SCRIPTS_DIR/setup.sh" "${@:2}"
        ;;

    status)
        detect_hardware
        echo -e "\n=== Engine Status ==="
        # Diffusers
        if curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"engine":"diffusers"'; then
            MODEL_NAME=$(curl -s http://127.0.0.1:8000/health | jq -r '.active_model // "none"')
            echo "[RUNNING] Diffusers Service (Port: 8000, Model: $MODEL_NAME)"
        elif curl -s http://127.0.0.1:8000/v1/models &> /dev/null; then
            echo "[RUNNING] vLLM Service (Port: 8000, OpenAI API Compatible)"
        else
            echo "[STOPPED] Port 8000 Engine (Diffusers / vLLM)"
        fi

        # ComfyUI
        if curl -s http://127.0.0.1:8188/system_stats &> /dev/null; then
            echo "[RUNNING] ComfyUI Service (Port: 8188)"
        else
            echo "[STOPPED] ComfyUI Service"
        fi

        # Ollama
        if curl -s http://127.0.0.1:11434/api/version &> /dev/null; then
            echo "[RUNNING] Ollama Service (Port: 11434, GGUF Models)"
        else
            echo "[STOPPED] Ollama Service"
        fi

        echo -e "\n=== Network Tunnel Status ==="
        if tailscale status 2>/dev/null | grep -qv "Logged out"; then
            echo "[ACTIVE]  Tailscale Mesh Network"
            tailscale status | head -n 3
        else
            echo "[INACTIVE] Tailscale Mesh Network"
        fi

        CF_URL=$(grep -o 'https://[-a-zA-Z0-9.]*\.trycloudflare\.com' "$LOG_DIR/tunnel.log" 2>/dev/null | tail -n 1 || true)
        if [ -n "$CF_URL" ] && pgrep -f "cloudflared tunnel" > /dev/null; then
            echo "[ACTIVE]  Cloudflare Public Tunnel: $CF_URL"
        else
            echo "[INACTIVE] Cloudflare Public Tunnel"
        fi

        if pgrep -f "code tunnel" > /dev/null; then
            VSCODE_NAME=$(grep -o '"name": *"[^"]*"' /root/.vscode/cli/code_tunnel.json 2>/dev/null | cut -d'"' -f4 || echo "colab-model-station")
            echo "[ACTIVE]  VS Code Remote Tunnel (Name: $VSCODE_NAME)"
            echo "          Endpoint: https://vscode.dev/tunnel/$VSCODE_NAME"
        else
            echo "[INACTIVE] VS Code Remote Tunnel"
        fi

        echo -e "\n=== Watchdog Status ==="
        python3 "$SCRIPTS_DIR/idle_watchdog.py" status
        ;;

    list)
        python3 -c "
import json
from pathlib import Path
cfg = json.loads(Path('$CONFIG_DIR/models.json').read_text())
cats = cfg.get('catalogs', {})

print('=== Diffusion Models ===')
for m, v in cats.get('diffusion', {}).get('models', {}).items():
    print(f'• {m:<45} | {v.get(\"description\", \"\")}')

print('\n=== LLM Models (vLLM Engine) ===')
for m, v in cats.get('llm', {}).get('vllm', {}).get('models', {}).items():
    print(f'• {m:<45} | ~{v.get(\"vram_gb\")}GB | {v.get(\"description\", \"\")}')

print('\n=== LLM Models (Ollama Engine) ===')
for m, v in cats.get('llm', {}).get('ollama', {}).get('models', {}).items():
    print(f'• {m:<25} | ~{v.get(\"vram_gb\")}GB | {v.get(\"description\", \"\")}')

print('\n=== Vision-Language (VLM) Models ===')
for m, v in cats.get('vlm', {}).get('models', {}).items():
    print(f'• {m:<35} | ~{v.get(\"vram_gb\")}GB | {v.get(\"description\", \"\")}')

print('\n=== Audio AI Models ===')
for m, v in cats.get('audio', {}).get('models', {}).items():
    print(f'• {m:<35} | ~{v.get(\"vram_gb\")}GB | {v.get(\"description\", \"\")}')

print('\n=== Dense Embedding & Reranker Models ===')
for m, v in cats.get('embedding', {}).get('models', {}).items():
    print(f'• {m:<45} | ~{v.get(\"vram_gb\")}GB | {v.get(\"description\", \"\")}')

print('\n=== Video Diffusion Models ===')
for m, v in cats.get('video', {}).get('models', {}).items():
    print(f'• {m:<35} | ~{v.get(\"vram_gb\")}GB | {v.get(\"description\", \"\")}')
"
        ;;

    # ==========================================================================
    # Engine: Diffusers (Port 8000)
    # ==========================================================================
    diffusers)
        ACTION="${2:-status}"
        PID_FILE="$LOG_DIR/diffusers.pid"
        LOG_FILE="$LOG_DIR/diffusers.log"

        case "$ACTION" in
            start)
                if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
                    echo "[WARN] Diffusers server is already running (PID: $(cat "$PID_FILE"))."
                    exit 0
                fi

                TARGET_ARGS=()
                MOCK_FLAG=0
                MODEL=""
                PRECISION="auto"

                shift 2
                while [[ $# -gt 0 ]]; do
                    case "$1" in
                        --model)
                            MODEL="$2"
                            TARGET_ARGS+=("--model" "$2")
                            shift 2
                            ;;
                        --precision)
                            PRECISION="$2"
                            TARGET_ARGS+=("--precision" "$2")
                            shift 2
                            ;;
                        --mock)
                            MOCK_FLAG=1
                            TARGET_ARGS+=("--mock")
                            shift
                            ;;
                        *)
                            if [ -z "$MODEL" ] && [[ "$1" != --* ]]; then
                                MODEL="$1"
                                TARGET_ARGS+=("--model" "$1")
                            else
                                TARGET_ARGS+=("$1")
                            fi
                            shift
                            ;;
                    esac
                done

                MODEL="${MODEL:-${STATION_MODEL:-black-forest-labs/FLUX.1-schnell}}"
                if [ "$MOCK_FLAG" -eq 1 ]; then
                    export STATION_MOCK_ENGINE="1"
                fi

                echo "[INFO] Starting Diffusers server (Model: $MODEL, Precision: $PRECISION)..."
                export STATION_MODEL="$MODEL"
                export STATION_PRECISION="$PRECISION"

                setsid python3 "$ENGINES_DIR/diffusers_engine/server.py" "${TARGET_ARGS[@]}" > "$LOG_FILE" 2>&1 &
                PID=$!
                echo $PID > "$PID_FILE"
                sleep 3

                if kill -0 "$PID" 2>/dev/null; then
                    echo "[SUCCESS] Diffusers server started (PID: $PID, Port: 8000)."
                else
                    echo "[ERROR] Server failed to start. Last log lines:"
                    tail -n 15 "$LOG_FILE"
                    exit 1
                fi
                ;;

            stop)
                if [ -f "$PID_FILE" ]; then
                    PID=$(cat "$PID_FILE")
                    echo "[INFO] Stopping Diffusers server (PID: $PID)..."
                    pkill -P "$PID" 2>/dev/null || true
                    kill "$PID" 2>/dev/null || true
                    sleep 1
                    if kill -0 "$PID" 2>/dev/null; then
                        kill -9 "$PID" 2>/dev/null || true
                    fi
                    rm -f "$PID_FILE"
                    echo "[SUCCESS] Stopped Diffusers server."
                else
                    echo "[WARN] No PID file found."
                fi
                ;;

            status)
                if curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"engine":"diffusers"'; then
                    echo "[RUNNING] Diffusers service is responsive."
                    curl -s http://127.0.0.1:8000/health | jq .
                else
                    echo "[STOPPED] Diffusers service is not reachable."
                fi
                ;;

            logs)
                FOLLOW="${3:-}"
                if [ "$FOLLOW" == "-f" ]; then
                    tail -f "$LOG_FILE"
                else
                    tail -n 40 "$LOG_FILE"
                fi
                ;;
        esac
        ;;

    # ==========================================================================
    # Engine: ComfyUI (Port 8188)
    # ==========================================================================
    comfyui)
        ACTION="${2:-status}"
        case "$ACTION" in
            start)
                python3 "$ENGINES_DIR/comfyui_engine/runner.py" start "${@:3}"
                ;;
            stop)
                python3 "$ENGINES_DIR/comfyui_engine/runner.py" stop
                ;;
            status)
                python3 "$ENGINES_DIR/comfyui_engine/runner.py" status
                ;;
            logs)
                FOLLOW="${3:-}"
                if [ "$FOLLOW" == "-f" ]; then
                    tail -f "$LOG_DIR/comfyui.log"
                else
                    tail -n 40 "$LOG_DIR/comfyui.log"
                fi
                ;;
        esac
        ;;

    # ==========================================================================
    # Engine: vLLM (Port 8000, OpenAI API)
    # ==========================================================================
    vllm)
        ACTION="${2:-status}"
        case "$ACTION" in
            start|serve)
                python3 "$ENGINES_DIR/llm_engine/runner.py" vllm start "${@:3}"
                ;;
            stop)
                python3 "$ENGINES_DIR/llm_engine/runner.py" vllm stop
                ;;
            status)
                python3 "$ENGINES_DIR/llm_engine/runner.py" vllm status
                ;;
            logs)
                FOLLOW="${3:-}"
                if [ "$FOLLOW" == "-f" ]; then
                    tail -f "$LOG_DIR/vllm.log"
                else
                    tail -n 40 "$LOG_DIR/vllm.log"
                fi
                ;;
            chat)
                MODEL="${3:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
                python3 "$TOOLS_DIR/chat.py" --endpoint "http://127.0.0.1:8000" --model "$MODEL"
                ;;
            bench)
                MODEL="${3:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
                python3 "$TOOLS_DIR/token_benchmark.py" --endpoint "http://127.0.0.1:8000" --model "$MODEL"
                ;;
        esac
        ;;

    # ==========================================================================
    # Engine: Ollama (Port 11434, GGUF)
    # ==========================================================================
    ollama)
        ACTION="${2:-status}"
        case "$ACTION" in
            start|serve)
                python3 "$ENGINES_DIR/llm_engine/runner.py" ollama start "${@:3}"
                ;;
            stop)
                python3 "$ENGINES_DIR/llm_engine/runner.py" ollama stop
                ;;
            status)
                python3 "$ENGINES_DIR/llm_engine/runner.py" ollama status
                ;;
            logs)
                FOLLOW="${3:-}"
                if [ "$FOLLOW" == "-f" ]; then
                    tail -f "$LOG_DIR/ollama.log"
                else
                    tail -n 40 "$LOG_DIR/ollama.log"
                fi
                ;;
            pull)
                MODEL="${3:-qwen2.5-coder:7b}"
                python3 "$ENGINES_DIR/llm_engine/runner.py" ollama pull "$MODEL"
                ;;
            list)
                python3 "$ENGINES_DIR/llm_engine/runner.py" ollama list
                ;;
            chat)
                MODEL="${3:-qwen2.5-coder:7b}"
                python3 "$TOOLS_DIR/chat.py" --endpoint "http://127.0.0.1:11434" --model "$MODEL"
                ;;
            bench)
                MODEL="${3:-qwen2.5-coder:7b}"
                python3 "$TOOLS_DIR/token_benchmark.py" --endpoint "http://127.0.0.1:11434" --model "$MODEL"
                ;;
        esac
        ;;

    # ==========================================================================
    # Network & Remote IDE Tunnels
    # ==========================================================================
    tunnel)
        TARGET="${2:-help}"
        case "$TARGET" in
            tailscale)
                ACTION="${3:-status}"
                case "$ACTION" in
                    up)
                        AUTHKEY="${4:-$TAILSCALE_AUTHKEY}"
                        mkdir -p "$DIR/.tailscale"
                        if ! command -v tailscale > /dev/null 2>&1 || ! command -v tailscaled > /dev/null 2>&1; then
                            echo "[INFO] Installing Tailscale..."
                            curl -fsSL https://tailscale.com/install.sh | sh
                        fi

                        if ! pgrep -f "tailscaled" > /dev/null; then
                            echo "[INFO] Starting tailscaled daemon (Userspace networking mode)..."
                            nohup tailscaled --tun=userspace-networking \
                                --state="$DIR/.tailscale/tailscaled.state" \
                                --socks5-server=localhost:1055 \
                                --outbound-http-proxy-listen=localhost:1055 > "$LOG_DIR/tailscaled.log" 2>&1 &
                            sleep 2
                        fi

                        TS_HOSTNAME="${TAILSCALE_HOSTNAME:-colab-model-station}"
                        if [ -n "$AUTHKEY" ]; then
                            echo "[INFO] Authenticating Tailscale with Auth Key (Hostname: $TS_HOSTNAME)..."
                            tailscale up --authkey="$AUTHKEY" --hostname="$TS_HOSTNAME" --ssh --accept-risk=all
                        else
                            echo "[INFO] Tailscale Auth Key not provided. Interactive login URL:"
                            tailscale up --hostname="$TS_HOSTNAME" --qr --ssh --accept-risk=all
                        fi
                        ;;
                    down)
                        tailscale down || true
                        ;;
                    serve)
                        PORT="8000"
                        if [ "${4:-}" == "--port" ] && [ -n "${5:-}" ]; then
                            PORT="$5"
                        elif [ -n "${4:-}" ]; then
                            PORT="$4"
                        fi
                        echo "[INFO] Proxying port $PORT to Tailscale private mesh via HTTP reverse proxy..."
                        tailscale serve reset > /dev/null 2>&1 || true
                        tailscale serve --bg --http="$PORT" "$PORT"
                        tailscale serve status
                        ;;
                    status)
                        tailscale status || true
                        ;;
                esac
                ;;

            cloudflare)
                ACTION="${3:-up}"
                case "$ACTION" in
                    up)
                        PORT="8000"
                        if [ "${4:-}" == "--port" ] && [ -n "${5:-}" ]; then
                            PORT="$5"
                        elif [ -n "${4:-}" ]; then
                            PORT="$4"
                        fi
                        if ! command -v cloudflared > /dev/null 2>&1; then
                            echo "[INFO] Installing cloudflared..."
                            bash "$SCRIPTS_DIR/setup.sh" tunnels
                        fi
                        pkill -f "cloudflared tunnel" || true
                        echo "[INFO] Opening Cloudflare Tunnel to port $PORT..."
                        nohup cloudflared tunnel --url "http://127.0.0.1:$PORT" --logfile "$LOG_DIR/tunnel.log" > /dev/null 2>&1 &
                        sleep 6
                        CF_URL=$(grep -o 'https://[-a-zA-Z0-9.]*\.trycloudflare\.com' "$LOG_DIR/tunnel.log" | tail -n 1 || true)
                        echo "================================================================"
                        echo "Cloudflare Public Tunnel Established:"
                        echo "  Endpoint: $CF_URL"
                        echo "================================================================"
                        ;;
                    down)
                        pkill -f "cloudflared tunnel" || true
                        echo "[INFO] Cloudflare tunnel terminated."
                        ;;
                esac
                ;;

            vscode)
                ACTION="${3:-status}"
                PERSIST_DIR="/content/drive/MyDrive/.vscode_colab"
                case "$ACTION" in
                    setup)
                        echo "[INFO] Checking VS Code CLI..."
                        if ! command -v code > /dev/null 2>&1; then
                            echo "[INFO] Downloading VS Code CLI..."
                            curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /tmp/vscode_cli.tar.gz
                            tar -xf /tmp/vscode_cli.tar.gz -C /usr/local/bin
                            chmod +x /usr/local/bin/code
                            rm -f /tmp/vscode_cli.tar.gz
                        fi
                        mkdir -p "$PERSIST_DIR"
                        mkdir -p /root/.vscode/cli
                        if [ -f "$PERSIST_DIR/token.json" ]; then
                            echo "[INFO] Restoring credentials from Google Drive ($PERSIST_DIR)..."
                            cp -rn "$PERSIST_DIR"/* /root/.vscode/cli/ 2>/dev/null || true
                        fi
                        echo "[SUCCESS] VS Code CLI ready: $(code --version | head -n 1)"
                        ;;
                    login)
                        mkdir -p "$PERSIST_DIR"
                        mkdir -p /root/.vscode/cli
                        cp -rn "$PERSIST_DIR"/* /root/.vscode/cli/ 2>/dev/null || true
                        if code tunnel user show >/dev/null 2>&1; then
                            echo "[INFO] Already authenticated to VS Code Tunnel via GitHub:"
                            code tunnel user show
                        else
                            echo "[INFO] Authenticating VS Code Tunnel via GitHub..."
                            code tunnel user login --provider github
                            cp -f /root/.vscode/cli/token.json /root/.vscode/cli/code_tunnel.json "$PERSIST_DIR/" 2>/dev/null || true
                            echo "[SUCCESS] Credentials saved to $PERSIST_DIR for persistence across sessions."
                        fi
                        ;;
                    start)
                        NAME="${4:-colab-model-station}"
                        mkdir -p "$PERSIST_DIR"
                        mkdir -p /root/.vscode/cli
                        cp -rn "$PERSIST_DIR"/* /root/.vscode/cli/ 2>/dev/null || true
                        if pgrep -f "code tunnel" > /dev/null; then
                            echo "[WARN] VS Code Tunnel is already running."
                        else
                            echo "[INFO] Starting VS Code Remote Tunnel (Name: $NAME)..."
                            nohup code tunnel --accept-server-license-terms --name "$NAME" > "$LOG_DIR/vscode_tunnel.log" 2>&1 &
                            sleep 4
                        fi
                        echo "================================================================"
                        echo "VS Code Remote Tunnel Status:"
                        echo "  Machine Name: $NAME"
                        echo "  Web URL:      https://vscode.dev/tunnel/$NAME"
                        echo "================================================================"
                        ;;
                    stop)
                        pkill -f "code tunnel" || true
                        echo "[SUCCESS] VS Code Remote Tunnel stopped."
                        ;;
                    status)
                        if pgrep -f "code tunnel" > /dev/null; then
                            echo "[RUNNING] VS Code Tunnel is active."
                        else
                            echo "[STOPPED] VS Code Tunnel is not running."
                        fi
                        ;;
                esac
                ;;
        esac
        ;;

    # ==========================================================================
    # Storage, Cache & Safeguards
    # ==========================================================================
    cache)
        ACTION="${2:-stats}"
        case "$ACTION" in
            prefetch)
                REPO_ID="${3:-}"
                [ -z "$REPO_ID" ] && { echo "Usage: ./engine.sh cache prefetch <model_repo_id>"; exit 1; }
                python3 "$SCRIPTS_DIR/cache_manager.py" model "$REPO_ID"
                ;;
            lora)
                LORA_SRC="${3:-}"
                [ -z "$LORA_SRC" ] && { echo "Usage: ./engine.sh cache lora <source_url_or_repo>"; exit 1; }
                python3 "$SCRIPTS_DIR/cache_manager.py" lora "$LORA_SRC"
                ;;
            stats)
                python3 "$SCRIPTS_DIR/cache_manager.py" stats
                ;;
        esac
        ;;

    watchdog)
        ACTION="${2:-status}"
        case "$ACTION" in
            start)
                python3 "$SCRIPTS_DIR/idle_watchdog.py" start "${@:3}"
                ;;
            stop)
                python3 "$SCRIPTS_DIR/idle_watchdog.py" stop
                ;;
            status)
                python3 "$SCRIPTS_DIR/idle_watchdog.py" status
                ;;
        esac
        ;;

    teardown)
        echo "[SYSTEM] Initiating complete system teardown..."
        python3 "$SCRIPTS_DIR/idle_watchdog.py" stop || true
        bash "$DIR/engine.sh" diffusers stop || true
        bash "$DIR/engine.sh" comfyui stop || true
        bash "$DIR/engine.sh" vllm stop || true
        bash "$DIR/engine.sh" ollama stop || true
        bash "$DIR/engine.sh" tunnel vscode stop || true
        pkill -f "cloudflared tunnel" || true
        tailscale down 2>/dev/null || true

        echo "[SYSTEM] Invoking Google Colab runtime unassign..."
        python3 -c "
try:
    from google.colab import runtime
    runtime.unassign()
    print('[SYSTEM] Runtime unassigned.')
except ImportError:
    print('[SYSTEM] Not running in Colab; local processes halted.')
"
        ;;

    help|*)
        show_help
        ;;
esac
