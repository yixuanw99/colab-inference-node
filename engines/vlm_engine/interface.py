"""
Vision-Language Model (VLM) engine interface placeholder for colab-model-station.
Reserved for Qwen2-VL, InternVL2.5, and Florence-2 document parsing.
"""
from __future__ import annotations

from typing import Any, Dict
from engines.base import VLMEngineInterface


class StubVLMEngine(VLMEngineInterface):
    """Placeholder implementation for future VLM integration."""

    def initialize(self, **kwargs: Any) -> None:
        pass

    def load_model(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "planned", "engine": "vlm", "model_id": model_id}

    def unload_model(self) -> None:
        pass

    def analyze_visual(self, image_input: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("VLM engine is scheduled for future release.")

    def parse_document(self, document_path: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Document parsing engine is scheduled for future release.")

    def get_status(self) -> Dict[str, Any]:
        return {"status": "unloaded", "engine": "vlm"}

    def shutdown(self) -> None:
        pass
