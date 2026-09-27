#!/usr/bin/env python3
"""
Automated Test & Diagnostic Suite for colab-model-station ComfyUI Engine.
Verifies service health, system stats, node object info, and queue availability.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def test_system_stats(base_url: str) -> dict:
    print("[TEST] 1. Verifying /system_stats endpoint...")
    url = f"{base_url.rstrip('/')}/system_stats"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))
        devices = data.get("devices", [])
        assert len(devices) > 0, "No compute devices found in ComfyUI"
        gpu_name = devices[0].get("name", "Unknown")
        vram_total_gb = devices[0].get("vram_total", 0) / (1024 ** 3)
        print(f"  [PASS] ComfyUI responsive. Hardware: {gpu_name} ({vram_total_gb:.1f} GB VRAM)")
        return data


def test_object_info(base_url: str) -> None:
    print("[TEST] 2. Verifying /object_info endpoint (node definitions)...")
    url = f"{base_url.rstrip('/')}/object_info"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        node_count = len(data)
        assert node_count > 50, f"Too few nodes loaded ({node_count})"
        print(f"  [PASS] Node registry verified ({node_count} nodes available).")


def test_queue_status(base_url: str) -> None:
    print("[TEST] 3. Verifying /queue endpoint...")
    url = f"{base_url.rstrip('/')}/queue"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "queue_running" in data
        assert "queue_pending" in data
        print("  [PASS] Queue status verified (running: {}, pending: {}).".format(
            len(data["queue_running"]), len(data["queue_pending"])
        ))


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Automated Test Suite")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Colab Model Station - ComfyUI Test Suite running against {args.endpoint}")
    print("=" * 80)

    try:
        test_system_stats(args.endpoint)
        test_object_info(args.endpoint)
        test_queue_status(args.endpoint)
        print("=" * 80)
        print("ALL COMFYUI TESTS PASSED SUCCESSFULLY.")
        print("=" * 80)
    except Exception as e:
        print(f"\n[FAIL] ComfyUI test suite failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
