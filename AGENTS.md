# AGENTS.md

Operating standards and repository guardrails for autonomous AI coding agents working on `colab-model-station` (hosted at `colab-inference-node`).

---

## 1. System Overview & Purpose

`colab-model-station` provides an automated, production-grade infrastructure for deploying arbitrary deep learning models—with primary focus on diffusion models (FLUX.1, FLUX.2, SDXL)—on Google Colab GPU runtimes. It enables private, headless remote access from local developer workstations (macOS, Linux, WSL) via Tailscale Mesh network (Userspace Networking mode) or Cloudflare Tunnels, orchestrated locally via the Google Colab CLI (`google-colab-cli`).

---

## 2. Core Directives & Persona Constraints

All AI agents interacting with this repository or acting within this project must strictly comply with the following standards:

### 2.1 Communication Style & Tone
- **Professional Engineering Standard**: Maintain an objective, concise, and authoritative technical tone. Focus on system architecture, operational precision, and root-cause analysis.
- **Zero Filler**: Avoid conversational pleasantries, cheerleading, and boilerplate platitudes.
- **English-First**: Default language for documentation, code comments, commit messages, and agent reasoning is English. Traditional Chinese (繁體中文) is used exclusively when updating Chinese documentation files (`*_zh.md`, `*-zh.ipynb`) or when the user explicitly requests it.
- **Strictly No Emojis**: Do not use decorative emojis anywhere (no responses, commit messages, code comments, documentation headers, or log messages). Maintain clean, standard technical formatting.
- **Clickable References**: Hyperlink all referenced files, classes, methods, and configurations using Markdown links (e.g., [`engine.sh`](file:///content/colab-inference-node/engine.sh)).

---

## 3. Repository Architecture & File Mapping

```
colab-model-station/
├── station.sh                   # Unified entrypoint for local workstation and Colab
├── station.ps1                  # Local workstation launcher (Windows PowerShell)
├── engine.sh                    # Colab master CLI lifecycle controller (Diffusers, ComfyUI, Tunnels)
├── setup.sh                     # Dependency installer forwarder -> scripts/setup.sh
├── colab_station.ipynb          # Primary interactive deployment notebook (English source of truth)
├── colab_station-zh.ipynb       # Mirrored interactive deployment notebook (Traditional Chinese)
├── AGENTS.md                    # Autonomous agent operating instructions
├── README.md                    # System architecture and deployment guide
├── README_zh.md                 # System architecture and deployment guide (Traditional Chinese)
├── .env.example                 # Template for local workstation and Colab environment variables
├── configs/                     # Central configurations, model catalogs, and hardware profiles
│   ├── models.json              # Model catalog (FLUX.1, SDXL, LoRA metadata)
│   ├── hardware_profiles.json   # VRAM allocation matrix and quantization guidance
│   ├── versions.env             # Pinned package and binary versions
│   └── requirements.lock        # Pinned Python package dependencies
├── engines/                     # Inference engine implementations & wrappers
│   ├── base.py                  # Abstract base engine class
│   ├── diffusers_engine/        # FastAPI + Hugging Face Diffusers adapter
│   │   ├── server.py            # FastAPI service exposing /v1/images/generations & /v1/loras
│   │   ├── pipeline_manager.py  # Model loading, FP8 casting, offload, and LoRA switcher
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   └── requirements.txt     # Engine dependencies
│   └── comfyui_engine/          # Headless ComfyUI daemon runner & workflow client
│       ├── runner.py            # Headless process supervisor
│       ├── client.py            # WebSocket/HTTP client for queueing prompt graphs
│       ├── workflows/           # Pre-configured API prompt JSON templates
│       └── requirements.txt
├── scripts/                     # Automation, bootstrap, and maintenance utilities
│   ├── setup.sh                 # Environment bootstrap & dependency installer
│   ├── remote_bootstrap.py      # Remote headless bootstrap payload for `colab exec`
│   ├── cache_manager.py         # Multi-threaded Hugging Face & LoRA prefetcher
│   ├── idle_watchdog.py         # Compute unit protection daemon (monitors idle & triggers unassign)
│   └── sync_git.sh              # Non-interactive Git commit & push tool
├── tools/                       # Client diagnostic and testing tools (Workstation side)
│   ├── station_ctl.py           # Core local workstation orchestrator (Google Colab CLI)
│   ├── generate.py              # CLI client for remote image generation & LoRA invocation
│   ├── benchmark.py             # Latency, memory peak, and throughput benchmark utility
│   └── test_inference.py        # Automated test suite for endpoints & LoRA swapping
└── logs/                        # Runtime daemon logs and PID tracking (git-ignored)
```

---

## 4. Execution Guardrails & Environmental Rules

### 4.1 Headless & Non-Interactive Execution
Google Colab operates in a headless Linux container without interactive TTY terminal input during cell execution.
- **Never prompt for user input**: All commands, scripts, and utilities must run non-interactively.
- Use explicit non-interactive flags (e.g., `DEBIAN_FRONTEND=noninteractive apt-get install -y`, `curl -fsSL`).
- Never run bare `tailscale up` without `--authkey` or unattended flags in automated scripts.
- Never write scripts that block indefinitely on `read` or interactive confirmations.

### 4.2 Runtime Filesystem & Path Standards
- **Working Directory**: Dynamically resolved via `$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)` or `Path(__file__).resolve().parent`. Never hardcode static parent paths.
- **No Google Drive Hardcoding**: Never write runtime logic that assumes Google Drive is mounted at `/content/drive/MyDrive/...`. Standalone Colab users clone directly to `/content/colab-inference-node`. If Google Drive is mounted, leverage it optionally for weight caching, but fail gracefully to local `/content/cache` if absent.
- Runtime artifacts (logs, PID files, sockets, temporary images) must reside within `logs/` or `/tmp`, covered by `.gitignore`.

### 4.3 Secrets & Credential Management
- Never commit private tokens, API keys, or Tailscale Auth Keys into git history.
- Read secrets in Colab using `google.colab.userdata`:
  ```python
  from google.colab import userdata
  tailscale_key = userdata.get('TAILSCALE_AUTHKEY')
  hf_token = userdata.get('HF_TOKEN')
  ```
- All temporary auth keys must be treated as ephemeral.

### 4.4 Compute Unit (CU) & Runtime Teardown
- Closing the browser tab does not stop a Google Colab VM; it continues running and consuming paid compute units until timeout.
- Every deployment notebook workflow and CLI orchestrator must provide an automated teardown mechanism:
  ```python
  from google.colab import runtime
  runtime.unassign()
  ```
- An automated idle watchdog (`scripts/idle_watchdog.py`) must monitor request traffic and terminate the runtime after a configurable idle threshold (default: 30 minutes).

### 4.5 Hardware & VRAM Allocation Matrix
Agents configuring models or inference engines must adhere to strict hardware boundaries:

| Hardware Tier | Typical VRAM | Recommended Engines & Formats | Optimization Flags |
|---|---|---|---|
| **CPU / Fallback** | System RAM | Diffusers (Mock / TinySD) | Sequential CPU offload, low steps |
| **Tesla T4** | 16 GB | Diffusers: SDXL (FP16), FLUX.1 (FP8 / NF4)<br>ComfyUI: FLUX.1 GGUF (Q4_K_S) | `enable_sequential_cpu_offload()`, batch size 1 |
| **NVIDIA L4** | 24 GB | Diffusers: FLUX.1 (FP8 `torch.float8_e4m3fn`), SDXL (FP16)<br>ComfyUI: FLUX.1 (FP8 Checkpoint) | `enable_model_cpu_offload()` |
| **A100 (SXM4)** | 40 GB / 80 GB | Diffusers: FLUX.1 (BF16 Native), SDXL (BF16)<br>ComfyUI: FLUX.1 (BF16) | Full GPU residence, optional `torch.compile` |

Inspect GPU hardware using `nvidia-smi` or `torch.cuda.get_device_properties(0)` before launching models.

### 4.6 Dual-Notebook Synchronization Rule
- `colab_station.ipynb` is the primary English source of truth.
- `colab_station-zh.ipynb` is the Traditional Chinese localized mirror.
- **Mandatory**: Any modification to cells, scripts, workflows, or deployment steps in one notebook must be mirrored identically in the other.
- Verify notebook integrity before committing:
  ```bash
  python3 -m json.tool colab_station.ipynb > /dev/null
  python3 -m json.tool colab_station-zh.ipynb > /dev/null
  ```

---

## 5. Standard CLI Operations & Workflows

### 5.1 Service Lifecycle Management (`engine.sh`)
```bash
# Diffusers FastAPI Service
bash engine.sh diffusers start --model "black-forest-labs/FLUX.1-schnell" --precision fp8
bash engine.sh diffusers stop
bash engine.sh diffusers logs -f

# ComfyUI Headless Service
bash engine.sh comfyui start
bash engine.sh comfyui stop
bash engine.sh comfyui logs -f

# Tailscale Mesh Networking (Userspace Mode)
bash engine.sh tunnel tailscale up --authkey "$TAILSCALE_AUTHKEY"
bash engine.sh tunnel tailscale down
bash engine.sh tunnel tailscale serve 8000

# Cloudflare Public Tunnel
bash engine.sh tunnel cloudflare up --port 8000
bash engine.sh tunnel cloudflare down

# Idle Watchdog (Compute Unit Protection)
bash engine.sh watchdog start --timeout 1800
bash engine.sh watchdog status
bash engine.sh watchdog stop

# Diagnostics & Status
bash engine.sh status
```

### 5.2 Git Synchronization
When synchronizing changes:
```bash
bash scripts/sync_git.sh "feat(core): implement dynamic lora switching endpoint"
```
Or standard git operations:
```bash
git add -A
git commit -m "type(scope): concise technical description in english"
git push origin main
```
Commit messages must follow Conventional Commits format in lowercase English, without emojis.
