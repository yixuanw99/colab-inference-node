# CLI Commands & Flags Reference

Comprehensive command-line interface specification for `engine.sh` and companion tools in `colab-model-station`.

---

## 1. `engine.sh` Master Controller

Master entrypoint for daemon management, inference service lifecycle, networking tunnels, and cost safeguards.

### 1.1 Diffusers Engine (`engine.sh diffusers`)

```bash
bash engine.sh diffusers [start|stop|restart|status|logs] [options]
```

- `start`: Launches FastAPI Diffusers server in the background.
  - `--model <hf_repo_id>`: Hugging Face model repository ID.
    - Default: `black-forest-labs/FLUX.1-dev`
    - Supported: `black-forest-labs/FLUX.1-schnell`, `stabilityai/stable-diffusion-xl-base-1.0`, `cyberdelia/CyberRealisticXL`
  - `--precision <precision>`: Model precision (`bf16`, `fp16`, `fp8`). Default: `bf16` for FLUX, `fp16` for SDXL.
  - `--port <port>`: HTTP bind port. Default: `8000`.
  - `--host <host>`: HTTP bind host. Default: `0.0.0.0`.
  - `--enable-cpu-offload`: Enable sequential model CPU offloading for low-VRAM GPUs (T4/L4).
- `stop`: Gracefully terminates the running Diffusers server and frees GPU VRAM.
- `status`: Checks if the server is healthy and responds on HTTP `/health`.
- `logs`: Streams server log output (`tail -f logs/diffusers.log`).

### 1.2 LLM Engines (`engine.sh vllm` / `engine.sh ollama`)

#### vLLM:
```bash
bash engine.sh vllm [start|stop|restart|status|logs] [options]
```
- `start`:
  - `--model <hf_model_id>`: Hugging Face model ID (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`).
  - `--quantization <method>`: Quantization method (`awq`, `gptq`, `bitsandbytes`, `none`). Default: `awq`.
  - `--port <port>`: OpenAI-compatible HTTP port. Default: `8000`.
  - `--gpu-memory-utilization <float>`: Target VRAM utilization (0.1 - 0.95). Default: `0.90`.
  - `--max-model-len <int>`: Maximum context window length. Default: `8192`.

#### Ollama:
```bash
bash engine.sh ollama [start|stop|pull|list|status|logs] [model_tag]
```
- `start`: Starts `ollama serve` on port 11434 in background.
- `pull <model>`: Pulls GGUF weights (e.g., `bash engine.sh ollama pull qwen2.5-coder:7b`).
- `list`: Lists locally cached Ollama models.
- `stop`: Terminates the Ollama daemon.

### 1.3 ComfyUI Engine (`engine.sh comfyui`)

```bash
bash engine.sh comfyui [start|stop|restart|status|logs] [options]
```
- `start`: Starts headless ComfyUI on port 8188 (`--listen 0.0.0.0 --port 8188 --highvram`).
- `stop`: Terminates ComfyUI process.
- `status`: Queries `/system_stats` to verify ComfyUI readiness.

### 1.4 Networking & Tunnels (`engine.sh tunnel`)

```bash
bash engine.sh tunnel [tailscale|cloudflare|vscode] [action] [options]
```

- `tailscale up --authkey <key>`: Starts Tailscale in userspace networking mode using ephemeral authkey.
- `tailscale down`: Disconnects Tailscale mesh.
- `tailscale serve <port>`: Exposes local port to the Tailscale private mesh.
- `cloudflare up --port <port>`: Launches Cloudflare quick tunnel and outputs temporary HTTPS URL.
- `cloudflare down`: Stops Cloudflare tunnel daemon.
- `vscode start [tunnel_name]`: Starts `code tunnel` and outputs device pairing URL/code.
- `vscode stop`: Terminates VS Code tunnel.

### 1.5 Safeguards & Cost Control (`engine.sh watchdog` / `teardown`)

```bash
bash engine.sh watchdog [start|stop|status] [--timeout <seconds>]
bash engine.sh teardown
bash engine.sh status
```
- `watchdog start --timeout 1800`: Monitors network traffic on inference ports; unassigns VM if idle for 30 minutes.
- `status`: Summarizes all running daemons, occupied ports, and GPU VRAM usage.
- `teardown`: Gracefully kills all running engines and unassigns Colab compute runtime.

---

## 2. Companion Python Diagnostic Tools

### 2.1 Precision Inpainting (`tools/inpaint_fill.py`)

```bash
python3 tools/inpaint_fill.py \
  --image <input_path> \
  --mask <binary_mask_path> \
  --prompt <garment_or_object_description> \
  --output <output_path> \
  [--steps 35] \
  [--guidance 30.0] \
  [--feather 15] \
  [--seed 42] \
  [--width <px>] \
  [--height <px>]
```

### 2.2 Text & Watermark Inpainting (`tools/inpaint_text.py`)

```bash
python3 tools/inpaint_text.py \
  --image <input_path> \
  --output <output_path> \
  [--threshold 220] \
  [--dilate 2]
```

### 2.3 Image Generation Client (`tools/generate.py`)

```bash
python3 tools/generate.py \
  --prompt <string> \
  [--negative-prompt <string>] \
  [--output <path>] \
  [--url http://localhost:8000] \
  [--steps 28] \
  [--cfg 3.5] \
  [--width 1024] \
  [--height 1024] \
  [--lora-path <repo_or_file>] \
  [--lora-weight 0.85]
```

### 2.4 Streaming Chat Client (`tools/chat.py`)

```bash
python3 tools/chat.py \
  --model <model_name> \
  [--url http://localhost:8000/v1] \
  [--system-prompt <string>]
```

### 2.5 Benchmarking Tools

```bash
# Diffusion latency and memory peak benchmark
python3 tools/benchmark.py --url http://localhost:8000 --iterations 5

# LLM TTFT (Time-To-First-Token) and generation throughput benchmark
python3 tools/token_benchmark.py --url http://localhost:8000/v1 --model "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
```
