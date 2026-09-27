"""
Generative Video engine interface placeholder for colab-model-station.
Reserved for Wan2.1, CogVideoX, and LTX-Video generation pipelines.
"""
from __future__ import annotations

from typing import Any, Dict
from engines.base import VideoEngineInterface


class StubVideoEngine(VideoEngineInterface):
    """Placeholder implementation for future video generation integration."""

    def initialize(self, **kwargs: Any) -> None:
        pass

    def load_model(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "planned", "engine": "video", "model_id": model_id}

    def unload_model(self) -> None:
        pass

    def generate_video(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Video generation engine is scheduled for future release.")

    def get_status(self) -> Dict[str, Any]:
        return {"status": "unloaded", "engine": "video"}

    def shutdown(self) -> None:
        pass
