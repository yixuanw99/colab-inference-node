# Hardware Profiles & VRAM Allocation Matrix

Guidance on GPU memory allocation, model precision selection, and offload policies across Google Colab GPU tiers.

---

## 1. Supported GPU Tiers & VRAM Capacities

| GPU Tier | Architecture | VRAM | Compute Capability | Optimal Workloads |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A100-SXM4-80GB** | Ampere | 80 GB | sm_80 | FLUX.1 [dev] BF16 (Unquantized), 32B LLMs, Video Diffusion |
| **NVIDIA A100-SXM4-40GB** | Ampere | 40 GB | sm_80 | FLUX.1 [dev] FP8, 14B-32B AWQ LLMs, SDXL Photorealism |
| **NVIDIA L4** | Ada Lovelace | 24 GB | sm_89 | FLUX.1 [schnell] FP8, SDXL Base FP16, 7B-14B AWQ LLMs |
| **NVIDIA T4** | Turing | 16 GB | sm_75 | SDXL with CPU Offload, Ollama 7B Q4_K_M, Whisper Tiny/Base |

---

## 2. Model Allocation & Precision Matrix

### 2.1 Diffusion Models

| Model | Recommended GPU | Precision | Native VRAM Peak | Low-VRAM Fallback Strategy |
| :--- | :--- | :--- | :--- | :--- |
| `black-forest-labs/FLUX.1-dev` | A100-80G / A100-40G | `bf16` | ~33 GB | `--precision fp8` (~18 GB) or `--enable-cpu-offload` |
| `black-forest-labs/FLUX.1-Fill-dev` | A100-80G / A100-40G | `bf16` | ~34 GB | Sequential CPU offload via `tools/inpaint_fill.py` |
| `black-forest-labs/FLUX.1-schnell` | A100 / L4 (24G) | `bf16` / `fp8` | ~16 GB (FP8) | Runs comfortably on L4 with FP8 quantization |
| `stabilityai/stable-diffusion-xl-base-1.0` | Any (T4/L4/A100) | `fp16` | ~8.5 GB | Runs natively on T4 (16GB) in FP16 |
| `cyberdelia/CyberRealisticXL` | Any (T4/L4/A100) | `fp16` | ~8.8 GB | Runs natively on T4 (16GB) in FP16 |

### 2.2 Large Language Models (LLM)

| Model | Serving Engine | Quantization | Minimum GPU | Context Window |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | vLLM | AWQ (4-bit) | T4 (16GB) / L4 (24GB) | 8,192 tokens |
| `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | vLLM | AWQ (4-bit) | L4 (24GB) / A100 | 8,192 tokens |
| `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | vLLM | AWQ (4-bit) | A100 (40GB/80GB) | 16,384 tokens |
| `qwen2.5-coder:7b` | Ollama | Q4_K_M (GGUF) | T4 (16GB) | 8,192 tokens |
| `deepseek-r1:8b` | Ollama | Q4_K_M (GGUF) | T4 (16GB) | 8,192 tokens |
| `deepseek-r1:32b` | Ollama | Q4_K_M (GGUF) | A100 (40GB/80GB) | 8,192 tokens |

### 2.3 Multimodal & Specialized Engines

| Modality | Model | Engine / Library | VRAM Peak | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **VLM** | `Qwen/Qwen2-VL-7B-Instruct` | Transformers / vLLM | ~16 GB | High-resolution image OCR & structured extraction |
| **Audio AI** | `openai/whisper-large-v3` | Transformers / Whisper | ~6.5 GB | Speech-to-text with word-level timestamps |
| **Embeddings** | `BAAI/bge-m3` | Sentence-Transformers | ~2.5 GB | 1024-dim dense vectors + hybrid sparse retrieval |
| **Video** | `THUDM/CogVideoX-2b` | Diffusers | ~18 GB | Text-to-video diffusion (2-second clips) |

---

## 3. Automated Fallback Rules for Agents

If the agent detects a lower-tier GPU during initialization (`nvidia-smi` output):
1. **On NVIDIA T4 (16GB)**:
   - For Diffusion: Prefer `cyberdelia/CyberRealisticXL` (FP16) or enforce `--enable-cpu-offload` for FLUX.
   - For LLMs: Never run unquantized 7B/14B; use `AWQ` or `Ollama Q4_K_M`.
2. **On NVIDIA L4 (24GB)**:
   - For FLUX.1: Set `--precision fp8`.
   - For LLMs: 7B and 14B AWQ models fit with full 8k-16k context window.
3. **On NVIDIA A100 (40GB/80GB)**:
   - Default to native `bf16` precision for full fidelity.
   - Enable high concurrency batching (`gpu_memory_utilization = 0.90`).
