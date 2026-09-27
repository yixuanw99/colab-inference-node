"""
Audio AI engine interface placeholder for colab-model-station.
Reserved for Faster-Whisper ASR transcription and F5-TTS / CosyVoice synthesis.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from engines.base import AudioEngineInterface


class StubAudioEngine(AudioEngineInterface):
    """Placeholder implementation for future Audio AI integration."""

    def initialize(self, **kwargs: Any) -> None:
        pass

    def load_model(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "planned", "engine": "audio", "model_id": model_id}

    def unload_model(self) -> None:
        pass

    def transcribe(self, audio_path: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Audio transcription engine is scheduled for future release.")

    def synthesize(self, text: str, reference_audio: Optional[str] = None, **kwargs: Any) -> bytes:
        raise NotImplementedError("TTS synthesis engine is scheduled for future release.")

    def get_status(self) -> Dict[str, Any]:
        return {"status": "unloaded", "engine": "audio"}

    def shutdown(self) -> None:
        pass
