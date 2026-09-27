"""
Diffusers Pipeline Manager for colab-model-station.
Handles hardware-aware model loading, quantization, CPU offloading,
dynamic LoRA hot-swapping, and image inference.
"""
from __future__ import annotations

import base64
import gc
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from engines.base import DiffusionEngineInterface


class DiffusersPipelineManager(DiffusionEngineInterface):
    """Manages Hugging Face Diffusers pipelines for FLUX and SDXL."""

    def __init__(self, output_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Any = None
        self.active_model_id: Optional[str] = None
        self.model_family: Optional[str] = None
        self.loaded_loras: Dict[str, Dict[str, Any]] = {}
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "logs" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.last_activity_time = time.time()
        self.is_mock = False

    def initialize(self, **kwargs: Any) -> None:
        """Initialize runtime and inspect hardware environment."""
        self.last_activity_time = time.time()
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect GPU hardware name, total memory, and allocated memory."""
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            vram_alloc_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            vram_res_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
        else:
            gpu_name = "CPU Standard"
            vram_total_mb = 0.0
            vram_alloc_mb = 0.0
            vram_res_mb = 0.0
        return {
            "device": self.device,
            "gpu_name": gpu_name,
            "vram_total_mb": round(vram_total_mb, 2),
            "vram_allocated_mb": round(vram_alloc_mb, 2),
            "vram_reserved_mb": round(vram_res_mb, 2),
        }

    def load_model(
        self,
        model_id: str,
        precision: str = "auto",
        offload_strategy: str = "auto",
        is_mock: bool = False,
    ) -> Dict[str, Any]:
        """Load a Diffusion model pipeline (FLUX or SDXL) with precision and offload strategy."""
        self.last_activity_time = time.time()

        if is_mock or model_id.startswith("mock://") or os.environ.get("STATION_MOCK_ENGINE", "").lower() in ("1", "true"):
            self.is_mock = True
            self.active_model_id = model_id
            self.model_family = "flux" if "flux" in model_id.lower() else "sdxl"
            self.pipe = "MOCK_PIPELINE"
            return {
                "status": "loaded",
                "model_id": model_id,
                "family": self.model_family,
                "precision": "mock",
                "offload": "mock",
                "is_mock": True,
            }

        self.is_mock = False
        self.unload_model()

        # Determine model family
        model_lower = model_id.lower()
        if "flux.2" in model_lower or "flux2" in model_lower:
            self.model_family = "flux2"
        elif "flux" in model_lower:
            self.model_family = "flux"
        elif "xl" in model_lower or "sdxl" in model_lower:
            self.model_family = "sdxl"
        else:
            self.model_family = "generic_diffusion"

        # Determine precision dtype
        if self.device == "cpu":
            target_dtype = torch.float32
            actual_precision = "float32"
        else:
            if precision in ("bf16", "bfloat16") or (precision == "auto" and torch.cuda.is_bf16_supported()):
                target_dtype = torch.bfloat16
                actual_precision = "bfloat16"
            elif precision in ("fp8", "float8") and self.model_family in ("flux", "flux2"):
                target_dtype = torch.bfloat16
                actual_precision = "fp8"
            else:
                target_dtype = torch.float16
                actual_precision = "float16"

        hw_info = self._detect_hardware()
        vram_total = hw_info["vram_total_mb"]

        if self.model_family == "flux2":
            from diffusers import Flux2Pipeline
            load_kwargs: Dict[str, Any] = {"torch_dtype": target_dtype}
            self.pipe = Flux2Pipeline.from_pretrained(model_id, **load_kwargs)
        elif self.model_family == "flux":
            from diffusers import FluxPipeline
            load_kwargs: Dict[str, Any] = {"torch_dtype": target_dtype}
            self.pipe = FluxPipeline.from_pretrained(model_id, **load_kwargs)
        elif self.model_family == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=target_dtype,
                use_safetensors=True,
            )
        else:
            from diffusers import AutoPipelineForText2Image
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=target_dtype,
            )

        # Offload strategy configuration
        applied_offload = offload_strategy
        if self.device == "cuda":
            if offload_strategy == "auto":
                if self.model_family == "flux2":
                    self.pipe.enable_model_cpu_offload()
                    applied_offload = "model"
                elif vram_total < 18000 and self.model_family == "flux":
                    self.pipe.enable_sequential_cpu_offload()
                    applied_offload = "sequential"
                elif vram_total < 32000:
                    self.pipe.enable_model_cpu_offload()
                    applied_offload = "model"
                else:
                    self.pipe.to("cuda")
                    applied_offload = "none"
            elif offload_strategy == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif offload_strategy == "model":
                self.pipe.enable_model_cpu_offload()
            elif offload_strategy == "none":
                self.pipe.to("cuda")
        else:
            self.pipe.to("cpu")
            applied_offload = "cpu"

        self.active_model_id = model_id
        return {
            "status": "loaded",
            "model_id": model_id,
            "family": self.model_family,
            "precision": actual_precision,
            "offload": applied_offload,
            "is_mock": False,
        }

    def unload_model(self) -> None:
        """Unload active model and release allocated memory."""
        self.last_activity_time = time.time()
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.active_model_id = None
        self.model_family = None
        self.loaded_loras.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_lora(
        self,
        lora_id_or_path: str,
        adapter_name: str,
        weight: float = 1.0,
        weight_name: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dynamically load and attach a LoRA adapter."""
        self.last_activity_time = time.time()
        if not self.pipe:
            raise RuntimeError("No active model pipeline loaded. Load a base model first.")

        if self.is_mock or lora_id_or_path.startswith("mock://"):
            self.loaded_loras[adapter_name] = {
                "source": lora_id_or_path,
                "weight": weight,
                "weight_name": weight_name,
                "subfolder": subfolder,
                "loaded_at": time.time(),
            }
            return {
                "status": "loaded",
                "adapter_name": adapter_name,
                "weight": weight,
                "active_loras": self.list_loras(),
            }

        load_kwargs: Dict[str, Any] = {"adapter_name": adapter_name}
        if weight_name:
            load_kwargs["weight_name"] = weight_name
        if subfolder:
            load_kwargs["subfolder"] = subfolder

        self.pipe.load_lora_weights(lora_id_or_path, **load_kwargs)
        self.pipe.set_adapters([adapter_name], adapter_weights=[weight])

        self.loaded_loras[adapter_name] = {
            "source": lora_id_or_path,
            "weight": weight,
            "weight_name": weight_name,
            "subfolder": subfolder,
            "loaded_at": time.time(),
        }

        return {
            "status": "loaded",
            "adapter_name": adapter_name,
            "weight": weight,
            "active_loras": self.list_loras(),
        }

    def unload_lora(self, adapter_name: str) -> Dict[str, Any]:
        """Unload and delete a registered LoRA adapter."""
        self.last_activity_time = time.time()
        if not self.pipe:
            raise RuntimeError("No active model pipeline loaded.")

        if adapter_name not in self.loaded_loras:
            raise ValueError(f"LoRA adapter '{adapter_name}' is not currently loaded.")

        is_mock_lora = self.loaded_loras[adapter_name].get("source", "").startswith("mock://")
        if self.is_mock or is_mock_lora:
            del self.loaded_loras[adapter_name]
            return {
                "status": "unloaded",
                "adapter_name": adapter_name,
                "active_loras": self.list_loras(),
            }

        if hasattr(self.pipe, "delete_adapters"):
            self.pipe.delete_adapters(adapter_name)
        elif hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()

        del self.loaded_loras[adapter_name]

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "status": "unloaded",
            "adapter_name": adapter_name,
            "active_loras": self.list_loras(),
        }

    def list_loras(self) -> List[Dict[str, Any]]:
        """Return list of loaded LoRA adapters."""
        return [
            {"adapter_name": k, **v}
            for k, v in self.loaded_loras.items()
        ]

    def _apply_request_loras(self, requested_loras: Optional[List[Dict[str, Any]]]) -> None:
        """Temporarily scale active LoRA adapters according to inference request."""
        if not requested_loras or not self.pipe or self.is_mock:
            return

        names: List[str] = []
        weights: List[float] = []
        for lora in requested_loras:
            name = lora.get("adapter_name")
            scale = float(lora.get("weight", 1.0))
            if name and name in self.loaded_loras:
                names.append(name)
                weights.append(scale)

        if names and hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters(names, adapter_weights=weights)

    def generate(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text-to-image inference and return generated artifacts."""
        self.last_activity_time = time.time()
        if not self.pipe:
            raise RuntimeError("No model is loaded. Call /v1/models/load first.")

        prompt = request_params.get("prompt", "")
        negative_prompt = request_params.get("negative_prompt")
        height = int(request_params.get("height", 1024))
        width = int(request_params.get("width", 1024))
        num_inference_steps = int(request_params.get("num_inference_steps", 4))
        guidance_scale = float(request_params.get("guidance_scale", 0.0))
        seed = request_params.get("seed")
        num_images = int(request_params.get("num_images", 1))
        return_base64 = bool(request_params.get("return_base64", True))
        requested_loras = request_params.get("loras")

        if seed is None:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())

        self._apply_request_loras(requested_loras)

        t_start = time.time()
        generated_artifacts: List[Dict[str, Any]] = []

        if self.is_mock:
            # Generate deterministic mock test image
            for idx in range(num_images):
                img_seed = seed + idx
                img = Image.new("RGB", (width, height), color=(img_seed % 255, (img_seed * 3) % 255, (img_seed * 7) % 255))
                filename = f"gen_{int(t_start)}_{idx}.png"
                filepath = self.output_dir / filename
                img.save(filepath, format="PNG")

                b64_str = None
                if return_base64:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                generated_artifacts.append({
                    "index": idx,
                    "seed": img_seed,
                    "file_path": str(filepath),
                    "file_name": filename,
                    "base64_data": b64_str,
                })
        else:
            generator_device = "cpu" if self.device == "cpu" else "cuda"
            generator = torch.Generator(device=generator_device).manual_seed(seed)

            infer_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
                "generator": generator,
            }

            if self.model_family == "sdxl" and negative_prompt:
                infer_kwargs["negative_prompt"] = negative_prompt

            with torch.inference_mode():
                output = self.pipe(**infer_kwargs)

            for idx, img in enumerate(output.images):
                img_seed = seed + idx
                filename = f"gen_{int(t_start)}_{idx}.png"
                filepath = self.output_dir / filename
                img.save(filepath, format="PNG")

                b64_str = None
                if return_base64:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                generated_artifacts.append({
                    "index": idx,
                    "seed": img_seed,
                    "file_path": str(filepath),
                    "file_name": filename,
                    "base64_data": b64_str,
                })

        duration = time.time() - t_start
        return {
            "model": self.active_model_id or "unknown",
            "images": generated_artifacts,
            "generation_time_sec": round(duration, 3),
            "parameters": {
                "prompt": prompt,
                "height": height,
                "width": width,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_images": num_images,
            },
            "loras_applied": self.list_loras(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return engine runtime health, loaded model, VRAM usage, and uptime."""
        now = time.time()
        hw_info = self._detect_hardware()
        return {
            "status": "ready" if self.pipe is not None else "idle",
            "engine": "diffusers",
            "active_model": self.active_model_id,
            "active_loras": self.list_loras(),
            "uptime_seconds": round(now - self.start_time, 1),
            "idle_seconds": round(now - self.last_activity_time, 1),
            **hw_info,
        }

    def shutdown(self) -> None:
        """Shutdown engine and release VRAM."""
        self.unload_model()
