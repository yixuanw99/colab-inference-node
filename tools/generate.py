#!/usr/bin/env python3
"""
CLI Client for Remote Image Generation on colab-model-station.
Sends generation requests with optional LoRA parameters over Tailscale mesh
or direct HTTP endpoint, saving returned image files locally.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def generate_image(
    endpoint: str,
    prompt: str,
    output_path: Path,
    negative_prompt: str | None = None,
    steps: int = 4,
    guidance: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    lora: str | None = None,
    lora_weight: float = 1.0,
) -> None:
    """Send generation request to diffusers endpoint and save output image."""
    url = f"{endpoint.rstrip('/')}/v1/images/generations"

    payload: dict = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "return_base64": True,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed

    if lora:
        payload["loras"] = [{"adapter_name": lora, "weight": lora_weight}]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    print(f"[CLIENT] Sending generation request to {url}...")
    print(f"  Prompt:   '{prompt}'")
    print(f"  Steps:    {steps} | Guidance: {guidance} | Size: {width}x{height}")
    if lora:
        print(f"  LoRA:     {lora} (weight: {lora_weight})")

    t_start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"[ERROR] HTTP {e.code}: {err_body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    total_time = time.time() - t_start
    gen_time = res_json.get("generation_time_sec", 0.0)
    images = res_json.get("images", [])

    print(f"[SUCCESS] Generation completed in {total_time:.2f}s (Engine compute: {gen_time:.2f}s).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for img_meta in images:
        b64_data = img_meta.get("base64_data")
        if b64_data:
            img_bytes = base64.b64decode(b64_data)
            output_file = output_path
            if len(images) > 1:
                idx = img_meta.get("index", 0)
                output_file = output_path.parent / f"{output_path.stem}_{idx}{output_path.suffix}"
            output_file.write_bytes(img_bytes)
            print(f"[SAVED] Image written to: {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Colab Model Station - Remote Generation Client")
    parser.add_argument("--endpoint", default=os.environ.get("STATION_ENDPOINT", "http://127.0.0.1:8000"), help="Base endpoint URL")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt")
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps (Default: 4 for FLUX.1 schnell)")
    parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale (Default: 0.0 for FLUX schnell, 3.5 for FLUX dev, 7.5 for SDXL)")
    parser.add_argument("--width", type=int, default=1024, help="Width in pixels")
    parser.add_argument("--height", type=int, default=1024, help="Height in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--lora", default=None, help="Active LoRA adapter name")
    parser.add_argument("--lora-weight", type=float, default=1.0, help="LoRA adapter weight")
    parser.add_argument("--output", default="output.png", help="Path to save output image")

    args = parser.parse_args()
    generate_image(
        endpoint=args.endpoint,
        prompt=args.prompt,
        output_path=Path(args.output),
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
        lora=args.lora,
        lora_weight=args.lora_weight,
    )


if __name__ == "__main__":
    main()
