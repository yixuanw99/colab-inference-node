"""
Vision-Language Model (VLM) engine for colab-model-station.
Provides visual reasoning, OCR, and document understanding
using Qwen2-VL on NVIDIA GPU.
"""
from __future__ import annotations

import base64
import gc
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from engines.base import VLMEngineInterface


class VLMEngine(VLMEngineInterface):
    """Production Vision-Language inference engine supporting Qwen2-VL."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[Qwen2VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self.active_model_id: Optional[str] = None
        self.start_time = time.time()
        self.last_activity_time = time.time()

    def initialize(self, **kwargs: Any) -> None:
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    def _resolve_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if path.is_file():
                return Image.open(path).convert("RGB")
            # Try decoding base64 string
            try:
                img_bytes = base64.b64decode(str(image_input))
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass
        raise ValueError(f"Cannot resolve image input: {type(image_input)}")

    def load_model(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.unload_model()
        self.last_activity_time = time.time()
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.active_model_id = model_id

        return {
            "status": "loaded",
            "engine": "vlm",
            "model_id": model_id,
            "device": self.device,
        }

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self.active_model_id = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def analyze_visual(self, image_input: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.last_activity_time = time.time()
        if not self.model or not self.processor:
            raise RuntimeError("No VLM loaded. Call load_model() first.")

        image = self._resolve_image(image_input)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        max_new_tokens = kwargs.get("max_new_tokens", 256)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get("do_sample", False),
            )

        # Trim prompt tokens from generation output
        trimmed_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return {
            "engine": "vlm",
            "model": self.active_model_id,
            "prompt": prompt,
            "response": output_text,
            "image_size": f"{image.width}x{image.height}",
        }

    def parse_document(self, document_path: str, **kwargs: Any) -> Dict[str, Any]:
        prompt = kwargs.pop(
            "prompt",
            "Read and extract all visible text and layout information from this document image.",
        )
        res = self.analyze_visual(document_path, prompt=prompt, **kwargs)
        return {
            "document": str(document_path),
            "parsed_text": res["response"],
            "model": res["model"],
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.model else "unloaded",
            "engine": "vlm",
            "active_model": self.active_model_id,
            "device": self.device,
            "uptime_seconds": round(time.time() - self.start_time, 2),
        }

    def shutdown(self) -> None:
        self.unload_model()


# Aliased stub for backwards compatibility
StubVLMEngine = VLMEngine
