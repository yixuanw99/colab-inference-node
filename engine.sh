#!/usr/bin/env bash
# ==============================================================================
# Colab Model Station - Unified Lifecycle & Tunnel Controller
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$DIR/configs"
SCRIPTS_DIR="$DIR/scripts"
ENGINES_DIR="$DIR/engines"
LOG_DIR="$DIR/logs"

mkdir -p "$LOG_DIR"
cd "$DIR"

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
        echo "Recommendation: A100 detected. FLUX unquantized (BF16) or SDXL at maximum batch size."
    elif [[ "$GPU_NAME" =~ "L4" ]] || [[ "$GPU_MEM" -gt 20000 ]]; then
        echo "Recommendation: L4 detected. FLUX (FP8) with model CPU offloading."
    elif [[ "$GPU_NAME" =~ "T4" ]] || [[ "$GPU_MEM" -gt 12000 ]]; then
        echo "Recommendation: T4 detected. FLUX (FP8 / NF4) with sequential CPU offload, or SDXL (FP16)."
    else
        echo "Recommendation: CPU mode detected. Use mock mode or lightweight models for testing."
    fi
}

function show_help() {
    echo "================================================================================"
    echo "Colab Model Station - Engine & Tunnel CLI"
    echo "================================================================================"
    echo "Usage: ./engine.sh [command] [options...]"
    echo ""
    echo "Engine: Diffusers (Port 8000, FastAPI REST / OpenAI Compatible):"
    echo "  diffusers start [--model ID] [--precision fp8|fp16|bf16] [--mock]"
    echo "  diffusers stop"
    echo "  diffusers status"
    echo "  diffusers logs [-f]"
    echo ""
    echo "Engine: ComfyUI (Port 8188, Headless API & Node Graphs):"
    echo "  comfyui start [--port 8188] [--vram auto|low|high]"
    echo "  comfyui stop"
    echo "  comfyui status"
    echo "  comfyui logs [-f]"
    echo ""
    echo "Network Tunnels:"
    echo "  tunnel tailscale up [authkey]    Connect to Tailscale mesh in Userspace mode"
    echo "  tunnel tailscale down            Disconnect Tailscale"
    echo "  tunnel tailscale serve [port]    Proxy port to private Tailnet"
    echo "  tunnel tailscale status          Show Tailscale connection status"
    echo "  tunnel cloudflare up [port]      Create public HTTPS Cloudflare tunnel"
    echo "  tunnel cloudflare down           Stop Cloudflare tunnel"
    echo ""
    echo "Storage & Cache:"
    echo "  cache prefetch <model_repo_id>   Accelerated download via HF Transfer"
    echo "  cache lora <source_url_or_repo>  Download LoRA weights to local registry"
    echo "  cache stats                      Show disk space and cache usage"
    echo ""
    echo "Compute Protection & System:"
    echo "  watchdog start [--timeout SECS]  Start automated idle teardown daemon"
    echo "  watchdog stop                    Stop watchdog daemon"
    echo "  watchdog status                  Check watchdog daemon status"
    echo "  status                           Show comprehensive system health"
    echo "  teardown                         Terminate services and unassign Colab VM"
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
        if curl -s http://127.0.0.1:8000/health &> /dev/null; then
            MODEL_NAME=$(curl -s http://127.0.0.1:8000/health | jq -r '.active_model // "none"')
            echo "[RUNNING] Diffusers Service (Port: 8000, Model: $MODEL_NAME)"
        else
            echo "[STOPPED] Diffusers Service"
        fi

        if curl -s http://127.0.0.1:8188/system_stats &> /dev/null; then
            echo "[RUNNING] ComfyUI Service (Port: 8188)"
        else
            echo "[STOPPED] ComfyUI Service"
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

        echo -e "\n=== Watchdog Status ==="
        python3 "$SCRIPTS_DIR/idle_watchdog.py" status
        ;;

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
                    rm -f "$PID_FILE"
                    echo "[SUCCESS] Stopped Diffusers server."
                else
                    echo "[WARN] No PID file found."
                fi
                ;;

            status)
                if curl -s http://127.0.0.1:8000/health &> /dev/null; then
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
                        PORT="${4:-8000}"
                        echo "[INFO] Proxying port $PORT to Tailscale private mesh..."
                        tailscale serve --bg --tcp "$PORT" "$PORT"
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
                        PORT="${4:-8000}"
                        if ! command -v cloudflared > /dev/null 2>&1; then
                            echo "[INFO] Installing cloudflared..."
                            bash "$SCRIPTS_DIR/setup.sh" all
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
        esac
        ;;

    cache)
        ACTION="${2:-stats}"
        case "$ACTION" in
            prefetch)
                REPO_ID="${3:-}"
                if [ -z "$REPO_ID" ]; then
                    echo "Usage: ./engine.sh cache prefetch <model_repo_id>"
                    exit 1
                fi
                python3 "$SCRIPTS_DIR/cache_manager.py" model "$REPO_ID"
                ;;
            lora)
                LORA_SRC="${3:-}"
                if [ -z "$LORA_SRC" ]; then
                    echo "Usage: ./engine.sh cache lora <source_url_or_repo>"
                    exit 1
                fi
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
