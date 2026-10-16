"""Visualization helpers for brake-disc analysis results."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from .core import BrakeDiscResult, Circle, DetectorConfig

# BGR colour palette
COLOR_OUTER = (0, 255, 255)      # yellow
COLOR_BORE = (255, 0, 255)       # magenta
COLOR_BOLT = (0, 255, 0)         # green
COLOR_PCD = (0, 165, 255)        # orange
COLOR_TEXT = (255, 255, 255)     # white
COLOR_BG = (0, 0, 0)             # black
COLOR_RAW = (180, 180, 180)      # gray


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _put_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: tuple,
    font_scale: float = 0.5,
    thickness: int = 1,
    color: tuple = COLOR_TEXT,
) -> None:
    """Draw text with a black background for readability."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(
        img, (x, y - th - baseline), (x + tw + 4, y + baseline), COLOR_BG, -1
    )
    cv2.putText(
        img,
        text,
        (x + 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_result(
    image: np.ndarray,
    result: BrakeDiscResult,
    *,
    draw_all_circles: bool = False,
) -> np.ndarray:
    """Render the analysis result onto a copy of *image*.

    Parameters
    ----------
    image:
        Original image (BGR or grayscale).
    result:
        Analysis result produced by :class:`BrakeDiscDetector`.
    draw_all_circles:
        If True, also draw every raw detection (useful for debugging).
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    # --- optional: all raw detections ---
    if draw_all_circles:
        for c in result.all_circles:
            cv2.circle(
                out,
                (int(c.center[0]), int(c.center[1])),
                int(c.radius),
                (180, 180, 180),
                1,
            )

    # --- outer diameter ---
    if result.outer_diameter:
        c = result.outer_diameter
        cv2.circle(
            out, (int(c.center[0]), int(c.center[1])), int(c.radius),
            COLOR_OUTER, 2,
        )
        _put_text_with_bg(
            out,
            f"OD: {c.diameter:.0f}px",
            (int(c.center[0] - c.radius), int(c.center[1] - c.radius - 5)),
            color=COLOR_OUTER,
        )

    # --- central bore ---
    if result.central_bore:
        c = result.central_bore
        cv2.circle(
            out, (int(c.center[0]), int(c.center[1])), int(c.radius),
            COLOR_BORE, 2,
        )
        _put_text_with_bg(
            out,
            f"Bore: {c.diameter:.0f}px",
            (int(c.center[0] - c.radius), int(c.center[1] + c.radius + 15)),
            color=COLOR_BORE,
        )

    # --- PCD circle ---
    if result.pcd_center and result.pcd_radius > 0:
        cx, cy = result.pcd_center
        cv2.circle(
            out, (int(cx), int(cy)), int(result.pcd_radius),
            COLOR_PCD, 2, cv2.LINE_AA,
        )
        _put_text_with_bg(
            out,
            f"PCD: {result.pcd_diameter:.0f}px",
            (int(cx + 5), int(cy - result.pcd_radius - 5)),
            color=COLOR_PCD,
        )
        # PCD center cross-hair
        s = 6
        cv2.line(out, (int(cx) - s, int(cy)), (int(cx) + s, int(cy)),
                 COLOR_PCD, 1)
        cv2.line(out, (int(cx), int(cy) - s), (int(cx), int(cy) + s),
                 COLOR_PCD, 1)

    # --- bolt holes ---
    for i, c in enumerate(result.bolt_holes, 1):
        cv2.circle(
            out, (int(c.center[0]), int(c.center[1])), int(c.radius),
            COLOR_BOLT, 2,
        )
        # center cross
        s = max(3, int(c.radius * 0.3))
        cv2.line(
            out,
            (int(c.center[0]) - s, int(c.center[1])),
            (int(c.center[0]) + s, int(c.center[1])),
            COLOR_BOLT, 1,
        )
        cv2.line(
            out,
            (int(c.center[0]), int(c.center[1]) - s),
            (int(c.center[0]), int(c.center[1]) + s),
            COLOR_BOLT, 1,
        )
        _put_text_with_bg(
            out,
            f"#{i} r={c.radius:.0f}",
            (int(c.center[0] + c.radius + 3), int(c.center[1])),
            color=COLOR_BOLT,
        )

    return out


def result_summary_text(result: BrakeDiscResult) -> str:
    """Return a human-readable multi-line summary of the result."""
    lines = ["===== Brake Disc Analysis Result ====="]
    lines.append(
        f"Image size: {result.image_shape[1]}x{result.image_shape[0]}"
    )

    if result.outer_diameter:
        c = result.outer_diameter
        lines.append(
            f"Outer Diameter : {c.diameter:.1f} px "
            f"(center=({c.center[0]:.1f}, {c.center[1]:.1f}))"
        )
    if result.central_bore:
        c = result.central_bore
        lines.append(
            f"Central Bore   : {c.diameter:.1f} px "
            f"(center=({c.center[0]:.1f}, {c.center[1]:.1f}))"
        )

    lines.append(f"Bolt holes     : {result.num_bolt_holes}")
    if result.bolt_holes:
        lines.append(
            f"Avg bolt dia   : {result.bolt_hole_avg_diameter:.1f} px"
        )
    if result.pcd_diameter > 0:
        lines.append(
            f"PCD diameter   : {result.pcd_diameter:.1f} px "
            f"(std={result.pcd_std:.1f})"
        )
        if result.pcd_center:
            lines.append(
                f"PCD center     : ({result.pcd_center[0]:.1f}, "
                f"{result.pcd_center[1]:.1f})"
            )

    if result.bolt_holes:
        lines.append("\n--- Bolt hole details ---")
        for i, c in enumerate(result.bolt_holes, 1):
            lines.append(
                f"  #{i}: center=({c.center[0]:.1f}, {c.center[1]:.1f}), "
                f"radius={c.radius:.1f}, diameter={c.diameter:.1f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline step visualization (for debugging / UI)
# ---------------------------------------------------------------------------
def generate_pipeline_steps(
    image: np.ndarray,
    result: BrakeDiscResult,
    config: DetectorConfig,
) -> Dict[str, np.ndarray]:
    """Generate intermediate pipeline images for visualization.

    Returns a dict mapping step name -> BGR image array:
        grayscale, blurred, binary, edges, candidate_circles, final
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps: Dict[str, np.ndarray] = {}

    # 1. Grayscale
    steps["grayscale"] = _to_bgr(gray)

    # 2. Blurred
    blurred = cv2.GaussianBlur(
        gray, (config.blur_kernel, config.blur_kernel), config.blur_sigma
    )
    steps["blurred"] = _to_bgr(blurred)

    # 3. Binary (adaptive threshold + morphology)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, config.adaptive_block_size, config.adaptive_c,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel, config.morph_kernel)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps["binary"] = _to_bgr(binary)

    # 4. Canny edges
    edges = cv2.Canny(blurred, int(config.hough_param1 * 0.4), config.hough_param1)
    steps["edges"] = _to_bgr(edges)

    # 5. All candidate circles (raw detections before classification)
    raw = _to_bgr(gray)
    for c in result.all_circles:
        cv2.circle(
            raw, (int(c.center[0]), int(c.center[1])), int(c.radius),
            COLOR_RAW, 1,
        )
    _put_text_with_bg(
        raw, f"candidates: {len(result.all_circles)}", (10, 25),
        color=COLOR_TEXT,
    )
    steps["candidates"] = raw

    # 6. Final annotated result
    steps["final"] = draw_result(image, result, draw_all_circles=False)

    return steps

