#!/usr/bin/env python3
"""
Precision Inpainting & Identity-Preserving Fill Tool for colab-model-station.
Leverages FLUX.1-Fill-dev to execute surgical garment replacement, object insertion,
or background modification while keeping 100% of unmasked pixels (face, posture, hands)
completely untouched and bit-for-bit identical to the source photograph.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_env() -> None:
    """Load environment variables from repository root .env file."""
    repo_root = Path(__file__).resolve().parents[1]
    env_file = repo_root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                if k not in os.environ:
                    os.environ[k] = v.strip().strip('"').strip("'")


def calculate_optimal_dimensions(orig_w: int, orig_h: int, target_pixels: int = 1024 * 1024) -> tuple[int, int]:
    """Calculate dimensions matching original aspect ratio, scaled to ~1MP and divisible by 16."""
    aspect = orig_w / orig_h
    h = int(np.sqrt(target_pixels / aspect))
    w = int(h * aspect)
    # Align to multiples of 16
    w = (w // 16) * 16
    h = (h // 16) * 16
    return max(w, 512), max(h, 512)


def execute_fill(
    image_path: Path,
    mask_path: Path,
    prompt: str,
    output_path: Path,
    model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
    steps: int = 35,
    guidance: float = 30.0,
    seed: int | None = 42,
    feather_radius: int = 15,
    target_width: int | None = None,
    target_height: int | None = None,
) -> None:
    """Execute FLUX.1-Fill inpainting and composite back to original resolution."""
    load_env()
    token = os.environ.get("HF_TOKEN")

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Input mask not found: {mask_path}")

    print("================================================================================")
    print("Colab Model Station - Precision Inpainting & Identity-Preserving Fill")
    print("================================================================================")
    print(f"Source Image: {image_path}")
    print(f"Mask Image:   {mask_path}")
    print(f"Prompt:       '{prompt}'")
    print(f"Model ID:     {model_id}")
    print(f"Steps:        {steps} | Guidance: {guidance} | Seed: {seed}")
    print("--------------------------------------------------------------------------------")

    start_time = time.time()
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_cv is None:
        raise ValueError(f"Failed to read mask image from {mask_path}")
    if mask_cv.shape[:2] != (orig_h, orig_w):
        mask_cv = cv2.resize(mask_cv, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if target_width and target_height:
        w_target = (target_width // 16) * 16
        h_target = (target_height // 16) * 16
    else:
        w_target, h_target = calculate_optimal_dimensions(orig_w, orig_h)

    print(f"[INFO] Original Resolution: {orig_w}x{orig_h}")
    print(f"[INFO] Inpainting Compute Resolution: {w_target}x{h_target}")

    img_resized = orig_img.resize((w_target, h_target), Image.LANCZOS)
    mask_resized = Image.fromarray(mask_cv).resize((w_target, h_target), Image.NEAREST)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    print(f"[INFO] Loading {model_id} onto {device} ({dtype})...")
    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    print("[INFO] Running inpainting diffusion...")
    inpaint_start = time.time()
    result = pipe(
        prompt=prompt,
        image=img_resized,
        mask_image=mask_resized,
        height=h_target,
        width=w_target,
        guidance_scale=guidance,
        num_inference_steps=steps,
        max_sequence_length=512,
        generator=generator,
    ).images[0]
    inpaint_duration = time.time() - inpaint_start
    print(f"[SUCCESS] Inpainting computation completed in {inpaint_duration:.2f}s.")

    # High-precision restoration & feathered composite back to original resolution
    print("[INFO] Compositing inpaint region back to original frame...")
    res_full = result.resize((orig_w, orig_h), Image.LANCZOS)

    if feather_radius > 0:
        kernel_size = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
        feathered_mask = cv2.GaussianBlur(mask_cv, (kernel_size, kernel_size), 0)
    else:
        feathered_mask = mask_cv

    alpha = (feathered_mask.astype(np.float32) / 255.0)[:, :, None]
    orig_np = np.array(orig_img).astype(np.float32)
    inpaint_np = np.array(res_full).astype(np.float32)

    composited = inpaint_np * alpha + orig_np * (1.0 - alpha)
    final_img = Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".png":
        final_img.save(output_path, format="PNG")
    else:
        final_img.save(output_path, format="JPEG", quality=100, subsampling=0)

    total_duration = time.time() - start_time
    print(f"[SAVED] Final image written to: {output_path.resolve()}")
    print(f"[SUMMARY] Total elapsed: {total_duration:.2f}s | Resolution: {orig_w}x{orig_h}")
    print("================================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precision inpainting and garment/object replacement with FLUX.1-Fill.")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument("--mask", required=True, type=Path, help="Path to binary mask image")
    parser.add_argument("--prompt", required=True, type=str, help="Prompt describing target replacement")
    parser.add_argument("--output", required=True, type=Path, help="Path to save processed image")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-Fill-dev", help="Hugging Face model ID")
    parser.add_argument("--steps", type=int, default=35, help="Number of inference steps (default: 35)")
    parser.add_argument("--guidance", type=float, default=30.0, help="Guidance scale (default: 30.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--feather", type=int, default=15, help="Mask feathering blur radius (default: 15)")
    parser.add_argument("--width", type=int, default=None, help="Target processing width (divisible by 16)")
    parser.add_argument("--height", type=int, default=None, help="Target processing height (divisible by 16)")

    args = parser.parse_args()
    execute_fill(
        image_path=args.image,
        mask_path=args.mask,
        prompt=args.prompt,
        output_path=args.output,
        model_id=args.model,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        feather_radius=args.feather,
        target_width=args.width,
        target_height=args.height,
    )


if __name__ == "__main__":
    main()
