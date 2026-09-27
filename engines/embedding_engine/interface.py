"""
Embedding & Reranker engine interface placeholder for colab-model-station.
Reserved for BGE-M3, ModernBERT, and Hugging Face TEI / Infinity.
"""
from __future__ import annotations

from typing import Any, Dict, List
from engines.base import EmbeddingEngineInterface


class StubEmbeddingEngine(EmbeddingEngineInterface):
    """Placeholder implementation for future embedding/reranker integration."""

    def initialize(self, **kwargs: Any) -> None:
        pass

    def load_model(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "planned", "engine": "embedding", "model_id": model_id}

    def unload_model(self) -> None:
        pass

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        raise NotImplementedError("Embedding engine is scheduled for future release.")

    def rerank(self, query: str, documents: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError("Reranker engine is scheduled for future release.")

    def get_status(self) -> Dict[str, Any]:
        return {"status": "unloaded", "engine": "embedding"}

    def shutdown(self) -> None:
        pass
