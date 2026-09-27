"""
Video Diffusion Engine for colab-model-station.
Provides text-to-video generation using CogVideoX and diffusers on NVIDIA GPU.
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from engines.base import VideoEngineInterface


class VideoEngine(VideoEngineInterface):
    """Production Video Diffusion engine supporting CogVideoX on NVIDIA GPU."""

    def __init__(self, output_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[Any] = None
        self.active_model_id: Optional[str] = None
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "logs" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.last_activity_time = time.time()
        self.is_mock = False

    def initialize(self, **kwargs: Any) -> None:
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    def load_model(
        self,
        model_id: str = "THUDM/CogVideoX-2b",
        precision: str = "bfloat16",
        is_mock: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.unload_model()
        self.last_activity_time = time.time()

        if is_mock or model_id.startswith("mock://") or os.environ.get("STATION_MOCK_ENGINE", "").lower() in ("1", "true"):
            self.is_mock = True
            self.active_model_id = model_id
            self.pipe = "MOCK_VIDEO_PIPELINE"
            return {
                "status": "loaded",
                "engine": "video",
                "model_id": model_id,
                "is_mock": True,
            }

        self.is_mock = False
        dtype = torch.bfloat16 if precision == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16

        from diffusers import CogVideoXPipeline
        self.pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        if self.device == "cuda":
            self.pipe.to("cuda")

        self.active_model_id = model_id
        return {
            "status": "loaded",
            "engine": "video",
            "model_id": model_id,
            "device": self.device,
        }

    def unload_model(self) -> None:
        self.pipe = None
        self.active_model_id = None
        self.is_mock = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def generate_video(
        self,
        prompt: str,
        num_frames: int = 16,
        num_inference_steps: int = 10,
        guidance_scale: float = 6.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.last_activity_time = time.time()
        if not self.pipe:
            raise RuntimeError("No video model loaded. Call load_model() first.")

        out_filename = f"video_{int(time.time())}.mp4"
        out_path = self.output_dir / out_filename

        if self.is_mock:
            out_path.write_bytes(b"MOCK_MP4_PAYLOAD")
            return {
                "engine": "video",
                "model": self.active_model_id,
                "prompt": prompt,
                "frames": num_frames,
                "output_path": str(out_path),
                "is_mock": True,
            }

        from diffusers.utils import export_to_video
        output = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(kwargs.get("seed", 42)),
        ).frames[0]

        export_to_video(output, str(out_path), fps=8)

        return {
            "engine": "video",
            "model": self.active_model_id,
            "prompt": prompt,
            "frames": len(output),
            "output_path": str(out_path),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.pipe else "unloaded",
            "engine": "video",
            "active_model": self.active_model_id,
            "device": self.device,
            "uptime_seconds": round(time.time() - self.start_time, 2),
        }

    def shutdown(self) -> None:
        self.unload_model()


# Aliased stub for backwards compatibility
StubVideoEngine = VideoEngine
