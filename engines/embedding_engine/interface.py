"""
Embedding & Reranker engine for colab-model-station.
Provides high-performance vector embeddings and document reranking
using PyTorch and Hugging Face Transformers on GPU.
"""
from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from engines.base import EmbeddingEngineInterface


class EmbeddingEngine(EmbeddingEngineInterface):
    """Production embedding and reranking engine supporting BGE-M3, MiniLM, and ModernBERT."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.active_model_id: Optional[str] = None
        self.start_time = time.time()
        self.last_activity_time = time.time()

    def initialize(self, **kwargs: Any) -> None:
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    def load_model(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: Any) -> Dict[str, Any]:
        self.unload_model()
        self.last_activity_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.active_model_id = model_id
        return {
            "status": "loaded",
            "engine": "embedding",
            "model_id": model_id,
            "device": self.device,
        }

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None
        self.active_model_id = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        self.last_activity_time = time.time()
        if not self.model or not self.tokenizer:
            raise RuntimeError("No embedding model loaded. Call load_model() first.")

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=kwargs.get("max_length", 512),
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs[0]
            input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            normalized = F.normalize(pooled, p=2, dim=1)

        return normalized.cpu().tolist()

    def rerank(self, query: str, documents: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        self.last_activity_time = time.time()
        if not documents:
            return []

        query_emb = torch.tensor(self.embed([query])[0])
        doc_embs = torch.tensor(self.embed(documents))
        scores = F.cosine_similarity(query_emb.unsqueeze(0), doc_embs).tolist()

        ranked = [
            {"index": i, "document": doc, "score": float(score)}
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.model else "unloaded",
            "engine": "embedding",
            "active_model": self.active_model_id,
            "device": self.device,
            "uptime_seconds": round(time.time() - self.start_time, 2),
        }

    def shutdown(self) -> None:
        self.unload_model()


# Aliased stub for backwards compatibility
StubEmbeddingEngine = EmbeddingEngine
