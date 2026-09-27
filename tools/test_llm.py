#!/usr/bin/env python3
"""
Automated Test & Diagnostic Suite for colab-model-station LLM Engines (vLLM & Ollama).
Verifies service health, model catalog, non-streaming completion, and streaming chat.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def test_models_endpoint(base_url: str) -> list[str]:
    print("[TEST] 1. Verifying /v1/models endpoint...")
    url = f"{base_url.rstrip('/')}/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer token"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id") for m in data.get("data", [])]
        print(f"  [PASS] Models endpoint responsive. Discovered {len(models)} model(s): {models}")
        return models


def test_chat_completion(base_url: str, model: str) -> None:
    print(f"[TEST] 2. Verifying /v1/chat/completions (non-streaming, model: {model})...")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Respond with exactly the word: 'PONG'"}],
        "temperature": 0.0,
        "max_tokens": 16,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer token"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        assert choices, "No choices returned in chat completion"
        content = choices[0].get("message", {}).get("content", "").strip()
        elapsed = time.time() - t0
        print(f"  [PASS] Chat response received in {elapsed:.2f}s: '{content}'")


def test_streaming_chat(base_url: str, model: str) -> None:
    print(f"[TEST] 3. Verifying /v1/chat/completions (streaming, model: {model})...")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5 separated by spaces."}],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": 32,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer token"},
    )
    t0 = time.time()
    chunks_received = 0
    full_text = ""
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        chunks_received += 1
                        full_text += content
            except json.JSONDecodeError:
                pass

    elapsed = time.time() - t0
    assert chunks_received > 0, "No streaming tokens received"
    print(f"  [PASS] Streaming verified ({chunks_received} chunks in {elapsed:.2f}s): '{full_text.strip()}'")


def main():
    parser = argparse.ArgumentParser(description="LLM Automated Test Suite")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434", help="API base URL")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Model name to verify")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Colab Model Station - LLM Test Suite running against {args.endpoint}")
    print("=" * 80)

    try:
        models = test_models_endpoint(args.endpoint)
        target_model = args.model if args.model in models else (models[0] if models else args.model)
        test_chat_completion(args.endpoint, target_model)
        test_streaming_chat(args.endpoint, target_model)
        print("=" * 80)
        print("ALL LLM TESTS PASSED SUCCESSFULLY.")
        print("=" * 80)
    except Exception as e:
        print(f"\n[FAIL] LLM test suite failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
