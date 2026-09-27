"""
FastAPI Server for colab-model-station Diffusers Engine.
Provides OpenAI-compatible endpoints for image generation,
dynamic LoRA adapter management, and runtime health tracking.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure root repository is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines.diffusers_engine.pipeline_manager import DiffusersPipelineManager
from engines.diffusers_engine.schemas import (
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    LoRALoadRequest,
    LoRAUnloadRequest,
    ModelLoadRequest,
    TeardownRequest,
)

from contextlib import asynccontextmanager

pipeline_mgr = DiffusersPipelineManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize hardware settings and optionally preload default model."""
    pipeline_mgr.initialize()
    default_model = os.environ.get("STATION_MODEL")
    precision = os.environ.get("STATION_PRECISION", "auto")
    is_mock = os.environ.get("STATION_MOCK_ENGINE", "").lower() in ("1", "true")

    if default_model:
        print(f"[ENGINE] Preloading configured model: {default_model} (Precision: {precision})", flush=True)
        try:
            pipeline_mgr.load_model(default_model, precision=precision, is_mock=is_mock)
        except Exception as e:
            print(f"[ENGINE] Warning: Model preloading failed: {e}", file=sys.stderr, flush=True)
    yield
    pipeline_mgr.shutdown()

app = FastAPI(
    title="Colab Model Station - Diffusers Inference API",
    version="1.0.0",
    description="Headless API for FLUX, SDXL, dynamic LoRAs, and remote orchestration.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static outputs directory
OUTPUT_DIR = REPO_ROOT / "logs" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """Return live system health, GPU VRAM metrics, and idle time."""
    return pipeline_mgr.get_status()


@app.get("/v1/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Expose metrics for idle watchdog monitoring."""
    status_data = pipeline_mgr.get_status()
    return {
        "engine": "diffusers",
        "status": status_data["status"],
        "idle_seconds": status_data["idle_seconds"],
        "uptime_seconds": status_data["uptime_seconds"],
        "active_model": status_data["active_model"],
        "vram_total_mb": status_data["vram_total_mb"],
        "vram_allocated_mb": status_data["vram_allocated_mb"],
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    """Return active model and available catalog models."""
    catalog_path = REPO_ROOT / "configs" / "models.json"
    catalog: Dict[str, Any] = {}
    if catalog_path.is_file():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

    status_data = pipeline_mgr.get_status()
    return {
        "active_model": status_data["active_model"],
        "available_models": catalog.get("models", {}),
    }


@app.post("/v1/models/load")
async def load_model_endpoint(req: ModelLoadRequest) -> Dict[str, Any]:
    """Dynamically load or switch base model checkpoint."""
    try:
        res = pipeline_mgr.load_model(
            model_id=req.model_id,
            precision=req.precision,
            offload_strategy=req.offload_strategy,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {req.model_id}: {str(e)}")


@app.delete("/v1/models/unload")
async def unload_model_endpoint() -> Dict[str, Any]:
    """Unload active model and free allocated GPU memory."""
    pipeline_mgr.unload_model()
    return {"status": "unloaded"}


@app.post("/v1/images/generations", response_model=GenerationResponse)
async def generate_images(req: GenerationRequest) -> Dict[str, Any]:
    """Generate images from text prompt with optional LoRA adapters."""
    try:
        params = req.model_dump()
        result = pipeline_mgr.generate(params)
        return result
    except RuntimeError as re:
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/v1/loras/load")
async def load_lora_endpoint(req: LoRALoadRequest) -> Dict[str, Any]:
    """Attach and activate a LoRA adapter dynamically."""
    try:
        res = pipeline_mgr.load_lora(
            lora_id_or_path=req.lora_id_or_path,
            adapter_name=req.adapter_name,
            weight=req.weight,
            weight_name=req.weight_name,
            subfolder=req.subfolder,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LoRA {req.adapter_name}: {str(e)}")


@app.post("/v1/loras/unload")
async def unload_lora_endpoint(req: LoRAUnloadRequest) -> Dict[str, Any]:
    """Unload a specific LoRA adapter and release its weights."""
    try:
        res = pipeline_mgr.unload_lora(req.adapter_name)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload LoRA {req.adapter_name}: {str(e)}")


@app.get("/v1/loras")
async def list_loras_endpoint() -> Dict[str, Any]:
    """List currently active LoRA adapters."""
    return {"active_loras": pipeline_mgr.list_loras()}


@app.post("/v1/system/teardown")
async def system_teardown(req: TeardownRequest) -> Dict[str, Any]:
    """Execute resource release and trigger Google Colab VM unassignment."""
    print(f"[SYSTEM] Teardown requested. Reason: {req.reason}", flush=True)
    pipeline_mgr.shutdown()

    # Trigger Colab unassign if running inside Google Colab environment
    try:
        from google.colab import runtime
        runtime.unassign()
        return {"status": "unassigning_runtime", "message": "Colab runtime unassigned successfully."}
    except ImportError:
        return {"status": "shutdown", "message": "Not in Google Colab environment. Engine terminated."}


def main():
    parser = argparse.ArgumentParser(description="Colab Model Station - Diffusers Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--model", default=None, help="Base model identifier to preload")
    parser.add_argument("--precision", default="auto", help="Precision (fp8, fp16, bf16, auto)")
    parser.add_argument("--mock", action="store_true", help="Start in mock mode for non-GPU testing")
    args = parser.parse_args()

    if args.model:
        os.environ["STATION_MODEL"] = args.model
    if args.precision:
        os.environ["STATION_PRECISION"] = args.precision
    if args.mock:
        os.environ["STATION_MOCK_ENGINE"] = "1"

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
