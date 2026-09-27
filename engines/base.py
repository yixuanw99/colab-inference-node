"""
Universal Base Engine interfaces for colab-model-station.
Provides standardized abstract contracts for LLM, Diffusion, VLM, Audio,
Embedding, and Video inference engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEngine(ABC):
    """Core lifecycle interface that all model station inference engines must satisfy."""

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        """Initialize the engine runtime environment and hardware allocation."""
        pass

    @abstractmethod
    def load_model(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Load a base model checkpoint into memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload active model and release allocated GPU VRAM."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return engine runtime health, loaded model, VRAM usage, and metrics."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully terminate engine and release all system resources."""
        pass


class DiffusionEngineInterface(BaseEngine):
    """Specialized contract for Diffusion and Image Generation backends."""

    @abstractmethod
    def load_lora(self, lora_id_or_path: str, adapter_name: str, weight: float = 1.0) -> Dict[str, Any]:
        """Load a LoRA adapter dynamically into the active pipeline."""
        pass

    @abstractmethod
    def unload_lora(self, adapter_name: str) -> Dict[str, Any]:
        """Unload a specific LoRA adapter and release its weights."""
        pass

    @abstractmethod
    def list_loras(self) -> List[Dict[str, Any]]:
        """List currently mounted LoRA adapters."""
        pass

    @abstractmethod
    def generate(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text-to-image or image-to-image inference."""
        pass


class LLMEngineInterface(BaseEngine):
    """Specialized contract for Large Language Model backends (vLLM, Ollama)."""

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Execute multi-turn chat completion with optional streaming."""
        pass

    @abstractmethod
    def completion(self, prompt: str, **kwargs: Any) -> Any:
        """Execute raw text completion."""
        pass


class VLMEngineInterface(BaseEngine):
    """Extensible contract for Vision-Language Models (Qwen2-VL, Florence-2)."""

    @abstractmethod
    def analyze_visual(self, image_input: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Process visual input alongside textual instructions."""
        pass

    @abstractmethod
    def parse_document(self, document_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Extract structured text, tables, and LaTeX from document images/PDFs."""
        pass


class AudioEngineInterface(BaseEngine):
    """Extensible contract for Audio AI models (Faster-Whisper, F5-TTS)."""

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Transcribe speech to text with timestamps."""
        pass

    @abstractmethod
    def synthesize(self, text: str, reference_audio: Optional[str] = None, **kwargs: Any) -> bytes:
        """Synthesize text into speech audio bytes."""
        pass


class EmbeddingEngineInterface(BaseEngine):
    """Extensible contract for Dense Embedding and Reranking models (BGE-M3)."""

    @abstractmethod
    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Compute vector embeddings for batch texts."""
        pass

    @abstractmethod
    def rerank(self, query: str, documents: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Re-rank candidate documents against a search query."""
        pass


class VideoEngineInterface(BaseEngine):
    """Extensible contract for Video Diffusion models (Wan2.1, CogVideoX)."""

    @abstractmethod
    def generate_video(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate video frames from text or initial image."""
        pass
