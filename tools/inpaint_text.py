#!/usr/bin/env python3
"""
Automated watermark and text removal tool for colab-model-station.
Performs surgical inpainting to remove text overlays without altering
the original resolution, subject fidelity, or unmasked pixels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def remove_text_watermark(
    image_path: Path,
    output_path: Path,
    region_box: tuple[float, float, float, float] | None = None,
    threshold: int = 215,
    dilate_kernel_size: int = 7,
    inpaint_radius: int = 9,
) -> None:
    """
    Remove bright text overlays from the designated image region.

    :param image_path: Path to source image.
    :param output_path: Path to save the processed image.
    :param region_box: (ymin, ymax, xmin, xmax) normalized fractions [0.0 - 1.0].
                       Default targets bottom 12% vertical, center 60% horizontal.
    :param threshold: Brightness threshold for text detection [0-255].
    :param dilate_kernel_size: Kernel size for text mask dilation.
    :param inpaint_radius: Inpainting neighborhood radius.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    h, w, _ = img_bgr.shape

    if region_box is None:
        # Default: Bottom 12% vertically, center 60% horizontally
        y1, y2 = int(h * 0.88), int(h * 0.98)
        x1, x2 = int(w * 0.20), int(w * 0.80)
    else:
        y1, y2 = int(h * region_box[0]), int(h * region_box[1])
        x1, x2 = int(w * region_box[2]), int(w * region_box[3])

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Threshold bright text inside ROI
    roi_gray = gray[y1:y2, x1:x2]
    _, thresh = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY)

    # Dilate mask to encompass anti-aliased character edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    mask[y1:y2, x1:x2] = dilated

    # Perform Fast Marching Navier-Stokes inpainting
    result_bgr = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".png":
        cv2.imwrite(str(output_path), result_bgr)
    else:
        cv2.imwrite(str(output_path), result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print(f"[SUCCESS] Text removed successfully.")
    print(f"  Source: {image_path} ({w}x{h})")
    print(f"  Target: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove text overlays and watermarks.")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument("--output", required=True, type=Path, help="Path to save output image")
    parser.add_argument("--threshold", type=int, default=215, help="Brightness threshold (default: 215)")
    args = parser.parse_args()

    remove_text_watermark(args.image, args.output, threshold=args.threshold)


if __name__ == "__main__":
    main()
