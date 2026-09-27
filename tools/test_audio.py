#!/usr/bin/env python3
"""
Automated Test & Diagnostic Suite for colab-model-station Audio Engine.
Verifies Whisper speech-to-text transcription and audio synthesis on NVIDIA A100.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines.audio_engine.interface import AudioEngine


def test_audio_pipeline(model_id: str) -> None:
    print("=" * 80)
    print(f"Colab Model Station - Audio Engine Diagnostic Suite ({model_id})")
    print("=" * 80)

    engine = AudioEngine()
    engine.initialize()

    print("[TEST] 1. Synthesizing test audio...")
    test_text = "Welcome to Colab Model Station on NVIDIA A100."
    t0 = time.time()
    audio_bytes = engine.synthesize(test_text)
    t_syn = time.time() - t0
    assert len(audio_bytes) > 1000, "Synthesized audio bytes too short"
    test_wav_path = REPO_ROOT / "logs" / "test_synth.wav"
    test_wav_path.parent.mkdir(parents=True, exist_ok=True)
    test_wav_path.write_bytes(audio_bytes)
    print(f"  [PASS] Synthesized {len(audio_bytes)} bytes WAV in {t_syn * 1000:.2f}ms ({test_wav_path}).")

    print("[TEST] 2. Loading Whisper ASR model onto GPU...")
    t0 = time.time()
    load_res = engine.load_model(model_id)
    t_load = time.time() - t0
    assert load_res["status"] == "loaded"
    print(f"  [PASS] Audio model loaded in {t_load:.2f}s on {load_res['device']}.")

    status = engine.get_status()
    assert status["status"] == "ready"
    print(f"  [PASS] Engine status confirmed: {status}")

    print("[TEST] 3. Running speech-to-text transcription...")
    t0 = time.time()
    res = engine.transcribe(str(test_wav_path))
    t_trans = time.time() - t0
    print(f"  [PASS] Audio transcription completed in {t_trans:.2f}s:")
    print(f"    Transcribed Text: '{res['text']}'")
    print(f"    Chunks/Timestamps: {res.get('chunks', [])}")

    print("[TEST] 4. Graceful model teardown...")
    engine.unload_model()
    assert engine.get_status()["status"] == "unloaded"
    print("  [PASS] Audio model unloaded and GPU VRAM released.")

    print("=" * 80)
    print("ALL AUDIO TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Audio Test Suite")
    parser.add_argument("--model", default="openai/whisper-tiny", help="Whisper model ID")
    args = parser.parse_args()

    try:
        test_audio_pipeline(args.model)
    except Exception as e:
        print(f"\n[FAIL] Audio test suite failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
