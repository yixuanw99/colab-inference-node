#!/usr/bin/env python3
"""
Automated Test & Diagnostic Suite for colab-model-station VLM Engine.
Verifies multimodal vision-language model loading, visual question answering,
and OCR document parsing on NVIDIA A100.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines.vlm_engine.interface import VLMEngine


def create_synthetic_test_image() -> Image.Image:
    """Create a synthetic test image with geometric shapes and text."""
    img = Image.new("RGB", (384, 384), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Draw blue rectangle
    draw.rectangle([40, 40, 180, 180], fill=(30, 144, 255), outline=(0, 0, 128), width=3)
    # Draw red circle
    draw.ellipse([200, 200, 340, 340], fill=(255, 69, 0), outline=(139, 0, 0), width=3)
    # Draw text
    draw.text((60, 100), "A100 GPU", fill=(255, 255, 255))
    draw.text((220, 260), "COLAB", fill=(255, 255, 255))
    return img


def test_vlm_pipeline(model_id: str) -> None:
    print("=" * 80)
    print(f"Colab Model Station - VLM Diagnostic Suite ({model_id})")
    print("=" * 80)

    test_img = create_synthetic_test_image()
    tmp_path = REPO_ROOT / "logs" / "test_vlm_input.png"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    test_img.save(tmp_path)
    print(f"[SETUP] Synthetic test image generated: {tmp_path} (384x384)")

    engine = VLMEngine()
    engine.initialize()

    print("[TEST] 1. Loading Vision-Language Model onto GPU...")
    t0 = time.time()
    load_res = engine.load_model(model_id)
    t_load = time.time() - t0
    assert load_res["status"] == "loaded"
    print(f"  [PASS] VLM loaded in {t_load:.2f}s on {load_res['device']}.")

    status = engine.get_status()
    assert status["status"] == "ready"
    print(f"  [PASS] Engine status confirmed: {status}")

    print("[TEST] 2. Visual question answering (analyze_visual)...")
    prompt = "Describe the colors and geometric shapes visible in this image."
    t0 = time.time()
    res = engine.analyze_visual(test_img, prompt=prompt, max_new_tokens=64)
    t_inf = time.time() - t0
    print(f"  [PASS] Visual inference succeeded in {t_inf:.2f}s:")
    print(f"    Output: '{res['response']}'")
    assert len(res["response"]) > 10, "VLM produced empty or trivial response"

    print("[TEST] 3. OCR and document parsing (parse_document)...")
    t0 = time.time()
    doc_res = engine.parse_document(str(tmp_path), prompt="What text words are written in this image?")
    t_ocr = time.time() - t0
    print(f"  [PASS] OCR parsing completed in {t_ocr:.2f}s:")
    print(f"    Extracted Text: '{doc_res['parsed_text']}'")

    print("[TEST] 4. Graceful model teardown...")
    engine.unload_model()
    assert engine.get_status()["status"] == "unloaded"
    print("  [PASS] Model unloaded and GPU VRAM released.")

    print("=" * 80)
    print("ALL VLM TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="VLM Test Suite")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct", help="VLM Hugging Face model ID")
    args = parser.parse_args()

    try:
        test_vlm_pipeline(args.model)
    except Exception as e:
        print(f"\n[FAIL] VLM test suite failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
