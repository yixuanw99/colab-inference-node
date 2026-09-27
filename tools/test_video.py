#!/usr/bin/env python3
"""
Automated Test & Diagnostic Suite for colab-model-station Video Engine.
Verifies text-to-video generation pipeline using CogVideoX on NVIDIA GPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines.video_engine.interface import VideoEngine


def test_video_pipeline(model_id: str, is_mock: bool = False) -> None:
    print("=" * 80)
    print(f"Colab Model Station - Video Engine Diagnostic Suite ({model_id})")
    print(f"Mode: {'MOCK' if is_mock else 'LIVE GPU'}")
    print("=" * 80)

    engine = VideoEngine()
    engine.initialize()

    print("[TEST] 1. Loading Video Diffusion Model...")
    t0 = time.time()
    load_res = engine.load_model(model_id, is_mock=is_mock)
    t_load = time.time() - t0
    assert load_res["status"] == "loaded"
    print(f"  [PASS] Video pipeline loaded in {t_load:.2f}s.")

    status = engine.get_status()
    assert status["status"] == "ready"
    print(f"  [PASS] Engine status confirmed: {status}")

    print("[TEST] 2. Generating video from text prompt...")
    prompt = "A glowing golden jellyfish floating gracefully in deep ocean waters."
    t0 = time.time()
    res = engine.generate_video(prompt, num_frames=8, num_inference_steps=2)
    t_gen = time.time() - t0
    assert "output_path" in res
    out_file = Path(res["output_path"])
    assert out_file.is_file(), f"Output video file not found: {out_file}"
    assert out_file.stat().st_size > 0, "Output video file is empty"
    print(f"  [PASS] Video generation completed in {t_gen:.2f}s ({out_file.stat().st_size} bytes):")
    print(f"    Path:   {res['output_path']}")
    print(f"    Frames: {res['frames']}")

    print("[TEST] 3. Graceful model teardown...")
    engine.unload_model()
    assert engine.get_status()["status"] == "unloaded"
    print("  [PASS] Video model unloaded and GPU VRAM released.")

    print("=" * 80)
    print("ALL VIDEO TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Video Diffusion Test Suite")
    parser.add_argument("--model", default="THUDM/CogVideoX-2b", help="Video model ID")
    parser.add_argument("--mock", action="store_true", help="Run with mock pipeline for fast validation")
    args = parser.parse_args()

    try:
        test_video_pipeline(args.model, is_mock=args.mock)
    except Exception as e:
        print(f"\n[FAIL] Video test suite failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
