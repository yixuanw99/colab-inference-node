#!/usr/bin/env python3
"""
Automated Test & Diagnostic Suite for colab-model-station.
Verifies service health, model catalog, text2img generation,
dynamic LoRA attachment, and graceful adapter detachment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def test_endpoint_health(base_url: str) -> dict:
    print("[TEST] 1. Verifying /health endpoint...")
    url = f"{base_url.rstrip('/')}/health"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))
        assert "status" in data
        assert "engine" in data
        print(f"  [PASS] Status: {data['status']}, Engine: {data['engine']}, Active Model: {data['active_model']}")
        return data


def test_list_models(base_url: str) -> dict:
    print("[TEST] 2. Verifying /v1/models endpoint...")
    url = f"{base_url.rstrip('/')}/v1/models"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "available_models" in data
        count = len(data["available_models"])
        print(f"  [PASS] Available models catalog verified ({count} models found).")
        return data


def test_generation(base_url: str) -> dict:
    print("[TEST] 3. Verifying /v1/images/generations endpoint...")
    url = f"{base_url.rstrip('/')}/v1/images/generations"
    payload = json.dumps({
        "prompt": "Test diagnostic image generation",
        "num_inference_steps": 2,
        "width": 512,
        "height": 512,
        "return_base64": True,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "images" in data
        assert len(data["images"]) >= 1
        elapsed = time.time() - t0
        print(f"  [PASS] Generation successful in {elapsed:.2f}s (Artifacts: {len(data['images'])}).")
        return data


def test_lora_lifecycle(base_url: str) -> None:
    print("[TEST] 4. Verifying dynamic LoRA loading and unloading...")

    # Load LoRA
    load_url = f"{base_url.rstrip('/')}/v1/loras/load"
    load_payload = json.dumps({
        "lora_id_or_path": "mock://test-lora",
        "adapter_name": "diagnostic_adapter",
        "weight": 0.8,
    }).encode("utf-8")

    req = urllib.request.Request(load_url, data=load_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        load_data = json.loads(resp.read().decode("utf-8"))
        assert load_data.get("status") == "loaded"
        print("  [PASS] LoRA adapter dynamically attached.")

    # List LoRAs
    list_url = f"{base_url.rstrip('/')}/v1/loras"
    with urllib.request.urlopen(urllib.request.Request(list_url), timeout=5) as resp:
        assert resp.status == 200
        list_data = json.loads(resp.read().decode("utf-8"))
        active_names = [a.get("adapter_name") for a in list_data.get("active_loras", [])]
        assert "diagnostic_adapter" in active_names
        print(f"  [PASS] Active LoRA registry verified: {active_names}")

    # Unload LoRA
    unload_url = f"{base_url.rstrip('/')}/v1/loras/unload"
    unload_payload = json.dumps({"adapter_name": "diagnostic_adapter"}).encode("utf-8")
    req = urllib.request.Request(unload_url, data=unload_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        unload_data = json.loads(resp.read().decode("utf-8"))
        assert unload_data.get("status") == "unloaded"
        print("  [PASS] LoRA adapter detached and resources cleared.")


def main():
    parser = argparse.ArgumentParser(description="Inference Test Suite")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    print("================================================================================")
    print(f"Colab Model Station - Test Suite running against {args.endpoint}")
    print("================================================================================")

    try:
        test_endpoint_health(args.endpoint)
        test_list_models(args.endpoint)
        test_generation(args.endpoint)
        test_lora_lifecycle(args.endpoint)
        print("================================================================================")
        print("ALL TESTS PASSED SUCCESSFULLY.")
        print("================================================================================")
    except Exception as e:
        print(f"\n[FAIL] Test suite failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
