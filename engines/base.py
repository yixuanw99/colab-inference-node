"""
Base engine abstraction for colab-model-station.
Defines the standard interface for all model inference engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEngine(ABC):
    """Abstract base class that all model station inference engines must implement."""

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
        """Execute inference generation using the loaded pipeline."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return engine runtime health, loaded model, VRAM usage, and metrics."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully terminate engine and release all resources."""
        pass
