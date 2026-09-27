"""
Pydantic schemas for the Diffusers FastAPI engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    adapter_name: str = Field(..., description="Unique name identifying the LoRA adapter")
    weight: float = Field(default=1.0, ge=-2.0, le=2.0, description="Scale weight applied to the adapter")


class LoRALoadRequest(BaseModel):
    lora_id_or_path: str = Field(..., description="Hugging Face repo ID, local path, or Civitai URL")
    adapter_name: str = Field(..., description="Unique identifier to register the LoRA adapter under")
    weight: float = Field(default=1.0, ge=-2.0, le=2.0, description="Default blending weight")
    weight_name: Optional[str] = Field(default=None, description="Specific .safetensors filename within repo")
    subfolder: Optional[str] = Field(default=None, description="Subfolder path inside the repo")


class LoRAUnloadRequest(BaseModel):
    adapter_name: str = Field(..., description="Identifier of the LoRA adapter to remove")


class ModelLoadRequest(BaseModel):
    model_id: str = Field(..., description="Hugging Face repository ID or local path")
    precision: str = Field(default="auto", description="Precision format: fp8, fp16, bf16, or auto")
    offload_strategy: str = Field(default="auto", description="Offload mode: auto, sequential, model, or none")


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt guiding image generation")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt guidance (SDXL only)")
    height: int = Field(default=1024, ge=256, le=2048, description="Image height in pixels")
    width: int = Field(default=1024, ge=256, le=2048, description="Image width in pixels")
    num_inference_steps: int = Field(default=4, ge=1, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(default=0.0, ge=0.0, le=30.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible outputs")
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    loras: Optional[List[LoRAConfig]] = Field(default=None, description="Active LoRAs and weights to apply")
    return_base64: bool = Field(default=True, description="Whether to include base64-encoded image strings")


class ImageArtifact(BaseModel):
    index: int
    seed: int
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    base64_data: Optional[str] = None


class GenerationResponse(BaseModel):
    model: str
    images: List[ImageArtifact]
    generation_time_sec: float
    parameters: Dict[str, Any]
    loras_applied: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    engine: str
    active_model: Optional[str]
    active_loras: List[Dict[str, Any]]
    device: str
    gpu_name: Optional[str]
    vram_allocated_mb: float
    vram_reserved_mb: float
    vram_total_mb: float
    uptime_seconds: float
    idle_seconds: float


class TeardownRequest(BaseModel):
    reason: str = Field(default="user_request", description="Reason for triggering runtime teardown")
    force: bool = Field(default=False, description="Terminate without waiting for queued jobs")
