#!/usr/bin/env python3
"""Command-line interface for the brake-disc hole analyzer.

Usage::

    python cli.py 2.png
    python cli.py 2.png --json result.json
    python cli.py 2.png --csv holes.csv --output annotated.jpg
    python cli.py images/ --output-dir results/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import cv2

from brake_disc import (
    BrakeDiscDetector,
    DetectorConfig,
    draw_result,
    result_summary_text,
)


def build_config(args: argparse.Namespace) -> DetectorConfig:
    """Build a :class:`DetectorConfig` from CLI arguments."""
    cfg = DetectorConfig()
    overrides = {
        "blur_kernel": args.blur_kernel,
        "adaptive_block_size": args.block_size,
        "adaptive_c": args.c_value,
        "min_circularity": args.min_circularity,
        "hough_param2": args.hough_param2,
        "hough_min_radius": args.hough_min_r,
        "hough_max_radius": args.hough_max_r,
        "use_blob": not args.no_blob,
        "use_hough": not args.no_hough,
        "min_bolt_radius_ratio": args.min_bolt_ratio,
        "max_bolt_radius_ratio": args.max_bolt_ratio,
        "pcd_tolerance": args.pcd_tolerance,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def collect_images(paths: List[str]) -> List[Path]:
    """Expand input paths into a list of image files."""
    images: List[Path] = []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in paths:
        path = Path(p)
        if path.is_dir():
            images.extend(
                sorted(f for f in path.iterdir() if f.suffix.lower() in exts)
            )
        elif path.is_file():
            images.append(path)
        else:
            logging.warning("Path not found: %s", path)
    return images


def save_json(result, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


def save_csv(result, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["hole_id", "center_x", "center_y", "radius", "diameter", "type"]
        )
        if result.central_bore:
            c = result.central_bore
            writer.writerow(
                ["bore", c.center[0], c.center[1], c.radius, c.diameter, "bore"]
            )
        for i, c in enumerate(result.bolt_holes, 1):
            writer.writerow(
                [i, c.center[0], c.center[1], c.radius, c.diameter, "bolt"]
            )
        if result.outer_diameter:
            c = result.outer_diameter
            writer.writerow(
                ["od", c.center[0], c.center[1], c.radius, c.diameter, "outer"]
            )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="brake-disc",
        description="Detect and measure holes in brake-disc images.",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image file(s) or directory to analyze.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path for the annotated output image (single image only).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for annotated output images (batch mode).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Write analysis results as JSON to this path.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Write hole measurements as CSV to this path.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Draw all raw circle detections (debug).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    # --- detection parameters ---
    det = parser.add_argument_group("detection parameters")
    det.add_argument("--blur-kernel", type=int, help="Gaussian blur kernel size.")
    det.add_argument("--block-size", type=int, help="Adaptive threshold block size.")
    det.add_argument("--c-value", type=int, help="Adaptive threshold C constant.")
    det.add_argument("--min-circularity", type=float, help="Minimum contour circularity.")
    det.add_argument("--hough-param2", type=float, help="Hough circle accumulator threshold.")
    det.add_argument("--hough-min-r", type=int, help="Hough minimum radius (px).")
    det.add_argument("--hough-max-r", type=int, help="Hough maximum radius (px).")
    det.add_argument("--no-hough", action="store_true", help="Disable Hough circle detection.")
    det.add_argument("--no-blob", action="store_true", help="Disable blob detector.")

    # --- classification parameters ---
    cls = parser.add_argument_group("classification parameters")
    cls.add_argument("--min-bolt-ratio", type=float, help="Min bolt radius / OD radius.")
    cls.add_argument("--max-bolt-ratio", type=float, help="Max bolt radius / OD radius.")
    cls.add_argument("--pcd-tolerance", type=float, help="PCD relative std tolerance.")

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    images = collect_images(args.images)
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    cfg = build_config(args)
    detector = BrakeDiscDetector(cfg)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read: {img_path}", file=sys.stderr)
            continue

        print(f"\n=== {img_path.name} ===")
        result = detector.analyze(image)
        print(result_summary_text(result))
        results[img_path.name] = result.to_dict()

        # Annotated image
        annotated = draw_result(image, result, draw_all_circles=args.show_all)
        if args.output and len(images) == 1:
            out_path = Path(args.output)
        elif output_dir:
            out_path = output_dir / f"{img_path.stem}_result.jpg"
        else:
            out_path = img_path.parent / f"{img_path.stem}_result.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"Annotated image saved: {out_path}")

        # CSV
        if args.csv_path and len(images) == 1:
            save_csv(result, Path(args.csv_path))
            print(f"CSV saved: {args.csv_path}")

    # JSON output (all results)
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results saved: {json_path}")

    if args.csv_path and len(images) > 1:
        print("\nWarning: --csv only applies to single-image mode.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
