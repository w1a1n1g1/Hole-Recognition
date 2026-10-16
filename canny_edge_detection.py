#!/usr/bin/env python3
"""Canny edge detection experiment for industrial part images.

This script runs Canny edge detection with multiple threshold combinations
and a noise-vs-blur comparison, saving side-by-side result figures.

Usage::

    python canny_edge_detection.py [image_path]

If *image_path* is omitted, ``2.png`` in the current directory is used.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib

# Use a non-interactive backend so the script runs in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Avoid CJK font errors: use English labels everywhere.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class ThresholdCombo:
    name: str
    threshold1: int
    threshold2: int

    @property
    def ratio(self) -> float:
        return self.threshold2 / self.threshold1 if self.threshold1 else 0.0


DEFAULT_COMBOS: List[ThresholdCombo] = [
    ThresholdCombo("combo1", 30, 90),
    ThresholdCombo("combo2", 40, 100),
    ThresholdCombo("combo3", 50, 150),
    ThresholdCombo("combo4", 60, 120),
    ThresholdCombo("combo5", 20, 80),
]


def load_image(path: str) -> "cv2.Mat":
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def canny_edges(
    img,
    threshold1: int,
    threshold2: int,
    *,
    blur_kernel: Tuple[int, int] = (3, 3),
    sigma: float = 1.0,
    use_blur: bool = True,
):
    blurred = cv2.GaussianBlur(img, blur_kernel, sigma) if use_blur else img
    return cv2.Canny(blurred, threshold1, threshold2)


def save_pair(original, edges, path: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    image_path = argv[0] if argv else "2.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    os.makedirs("results", exist_ok=True)
    img = load_image(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Processing: {image_path} ({img.shape[1]}x{img.shape[0]})")

    # 1. Threshold combination sweep
    for combo in DEFAULT_COMBOS:
        edges = canny_edges(img, combo.threshold1, combo.threshold2, use_blur=True)
        out = f"results/{stem}_{combo.name}_blur.jpg"
        title = (
            f"Canny t1={combo.threshold1} t2={combo.threshold2} "
            f"(ratio {combo.ratio:.1f}, blurred)"
        )
        save_pair(img, edges, out, title)
        print(f"  saved {out}")

    # 2. Noise influence: best params with and without blur
    best = DEFAULT_COMBOS[2]  # combo3: 50/150
    edges_no_blur = canny_edges(img, best.threshold1, best.threshold2, use_blur=False)
    out = f"results/{stem}_no_blur.jpg"
    save_pair(img, edges_no_blur, out, f"Canny t1={best.threshold1} t2={best.threshold2} (no blur)")
    print(f"  saved {out}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
