---
name: colab-model-station
description: >-
  Operational runbook and command control interface for colab-model-station.
  Use when deploying, configuring, serving, benchmarking, or troubleshooting deep learning
  models (FLUX.1, FLUX.1-Fill, SDXL, vLLM, Ollama, ComfyUI, Qwen2-VL, Whisper, Wan2.1,
  CogVideoX, BGE-M3), orchestrating remote GPU instances on Google Colab, managing Tailscale/Cloudflare
  tunnels, executing precision inpainting/styling, or controlling compute unit teardown.
---

# Colab Model Station - Agent Skill & Runbook

This skill equips an autonomous AI agent or LLM with the complete operational knowledge required to orchestrate, serve, and debug deep learning models on `colab-model-station` across Google Colab GPU runtimes (A100-80G, L4-24G, T4-16G) and local developer workstations.

---

## 1. Quick Decision Matrix

Select the appropriate engine or workflow based on the user's objective:

| Objective | Target Model / Tool | Execution Command | Port |
| :--- | :--- | :--- | :--- |
| **High-Fidelity Text-to-Image** | FLUX.1-dev (BF16 / FP8) | `bash engine.sh diffusers start --model "black-forest-labs/FLUX.1-dev" --precision bf16` | 8000 |
| **Fast Commercial Photorealism** | CyberRealistic XL (SDXL) | `bash engine.sh diffusers start --model "cyberdelia/CyberRealisticXL" --precision fp16` | 8000 |
| **Garment Replacement / Inpainting** | FLUX.1-Fill-dev (16-Ch Flow) | `python3 tools/inpaint_fill.py --image <img> --mask <mask.png> --prompt "<desc>" --output <out.png>` | CLI |
| **Watermark / Text Inpainting** | Fast Marching PDE | `python3 tools/inpaint_text.py --image <img> --output <out.png>` | CLI |
| **High-Throughput LLM Serving** | vLLM (AWQ / GPTQ / BF16) | `bash engine.sh vllm start --model "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"` | 8000 |
| **Flexible Local/Ollama LLM** | Ollama (GGUF) | `bash engine.sh ollama start && bash engine.sh ollama pull "qwen2.5-coder:7b"` | 11434 |
| **Node Graph Diffusion Workflows** | ComfyUI Headless | `bash engine.sh comfyui start` | 8188 |
| **Mesh VPN Remote Access** | Tailscale (Userspace) | `bash engine.sh tunnel tailscale up --authkey "$TAILSCALE_AUTHKEY"` | Mesh |
| **Public HTTPS Remote Access** | Cloudflare Quick Tunnel | `bash engine.sh tunnel cloudflare up --port 8000` | HTTPS |
| **Remote IDE Development** | VS Code Remote Tunnel | `bash engine.sh tunnel vscode start "colab-station"` | VS Code |
| **Idle Guard / Cost Protection** | Watchdog Daemon | `bash engine.sh watchdog start --timeout 1800` | Background |

---

## 2. Standard Operating Procedures

### 2.1 Managing Diffusion Engines (Diffusers)

Diffusers serves an OpenAPI-compatible HTTP REST server on port 8000.

1. **Start Server with Default or Custom Model**:
   ```bash
   bash engine.sh diffusers start --model "black-forest-labs/FLUX.1-dev" --precision bf16
   ```
2. **Verify Server Health & Model Status**:
   ```bash
   curl -s http://localhost:8000/health
   # Expected JSON: {"status": "ok", "model": "black-forest-labs/FLUX.1-dev", ...}
   ```
3. **Execute Remote Generation CLI**:
   ```bash
   python3 tools/generate.py \
     --prompt "High-end studio portrait of an architect in a minimalist concrete gallery" \
     --output "outputs/portrait.png" \
     --steps 28 \
     --cfg 3.5
   ```
4. **Dynamic LoRA Injection**:
   ```bash
   python3 tools/generate.py \
     --prompt "Photorealistic commercial shot" \
     --lora-path "XLabs-AI/flux-RealismLora" \
     --lora-weight 0.85 \
     --output "outputs/portrait_lora.png"
   ```
5. **Stop Engine & Reclaim GPU VRAM**:
   ```bash
   bash engine.sh diffusers stop
   ```

### 2.2 Precision Inpainting & Garment Replacement (FLUX.1-Fill)

When modifying clothing, accessories, or objects while **preserving 100% of the subject's face, posture, and hands**:

```bash
python3 tools/inpaint_fill.py \
  --image "samples/fashion_model.jpg" \
  --mask "samples/garment_mask.png" \
  --prompt "A tailored charcoal navy double-breasted wool trench coat with structured lapels, horn buttons, premium fabric texture, matching ambient studio lighting" \
  --output "outputs/styled_trench_coat.png" \
  --steps 35 \
  --guidance 30.0 \
  --feather 15 \
  --seed 42
```
*Key Invariant*: All unmasked pixels are bit-for-bit preserved from the source photograph. Gaussian feathering (`--feather 15`) prevents seam lines.

### 2.3 Managing LLM Engines (vLLM & Ollama)

#### vLLM (Production OpenAI-Compatible Endpoint):
```bash
# Start vLLM on port 8000
bash engine.sh vllm start --model "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ" --port 8000

# Stream chat via CLI tool
python3 tools/chat.py --model "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ" --url "http://localhost:8000/v1"

# Stop vLLM
bash engine.sh vllm stop
```

#### Ollama (Dynamic Local Model Runner):
```bash
# Start Ollama daemon on port 11434
bash engine.sh ollama start

# Pull model non-interactively
bash engine.sh ollama pull "qwen2.5-coder:7b"

# Query Ollama via CLI
python3 tools/chat.py --model "qwen2.5-coder:7b" --url "http://localhost:11434/v1"

# Stop Ollama
bash engine.sh ollama stop
```

### 2.4 Managing ComfyUI (Headless)

```bash
# Start ComfyUI daemon on port 8188
bash engine.sh comfyui start

# Verify API is ready
curl -s http://localhost:8188/system_stats

# Run automated workflow test client
python3 tools/test_comfyui.py

# Stop ComfyUI
bash engine.sh comfyui stop
```

### 2.5 Networking & Tunnels

```bash
# Tailscale Mesh (Userspace Networking, No Root TUN device needed)
bash engine.sh tunnel tailscale up --authkey "$TAILSCALE_AUTHKEY"
bash engine.sh tunnel tailscale serve 8000

# Cloudflare Quick Tunnel (Public temporary HTTPS URL)
bash engine.sh tunnel cloudflare up --port 8000

# VS Code Remote Tunnel (Attach desktop VS Code to Colab VM)
bash engine.sh tunnel vscode start "colab-station"
```

### 2.6 Cost Protection & Safe Teardown

```bash
# Launch background idle watchdog (shuts down VM after 30 mins of inactivity)
bash engine.sh watchdog start --timeout 1800

# Check running daemons and memory usage
bash engine.sh status

# Full teardown and unassign Colab compute units
bash engine.sh teardown
```

---

## 3. Mandatory Agent Operational Guardrails

All autonomous agents executing within this environment must adhere strictly to these constraints:

1. **Non-Interactive Execution Only**:
   - Google Colab executes in headless mode without a TTY terminal.
   - Never run interactive commands (`read`, unflagged `apt-get`, interactive `tailscale up`).
   - Always use non-interactive flags (`DEBIAN_FRONTEND=noninteractive`, `--authkey`, `-y`).
2. **Sequential VRAM Management**:
   - Only ONE major GPU engine (FLUX, SDXL, vLLM, ComfyUI) may run at a time on single-GPU instances.
   - Always call `bash engine.sh <current_engine> stop` and verify port release before starting another engine.
3. **No Hardcoded Parent Paths or Drive Mounts**:
   - Always dynamically resolve working directories via relative paths or `Path(__file__).resolve().parent`.
   - Never assume Google Drive is mounted at `/content/drive/MyDrive/...`. Fall back gracefully to `/content/cache`.
4. **Zero CU Waste Guarantee**:
   - Background tasks on Colab continue consuming credits after tab closure. Always verify `watchdog` is running or explicitly call `bash engine.sh teardown` when workflows conclude.
5. **No Decorative Emojis**:
   - Follow standard technical formatting. Avoid emojis in logs, status messages, code, and documentation.

---

## 4. Progressive References

For in-depth technical details, consult the reference sheets:
- [CLI Commands & Flags Reference](references/cli_commands.md)
- [Hardware Profiles & VRAM Allocation Matrix](references/hardware_profiles.md)
- [Troubleshooting & Recovery Procedures](references/troubleshooting.md)
