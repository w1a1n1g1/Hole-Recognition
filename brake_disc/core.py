"""Brake disc hole detection and measurement library.

This module provides a robust, modular pipeline for analyzing brake disc
images.  It combines three circle-detection strategies (contour analysis,
Hough circle transform and the OpenCV SimpleBlobDetector) and then applies
brake-disc-domain knowledge to identify the outer diameter, the central
bore, the bolt holes and the pitch-circle diameter (PCD).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Circle:
    """A detected circular feature."""

    center: Tuple[float, float]
    radius: float
    method: str = "contour"
    confidence: float = 1.0

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2

    def distance_to(self, other: "Circle") -> float:
        dx = self.center[0] - other.center[0]
        dy = self.center[1] - other.center[1]
        return math.hypot(dx, dy)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BrakeDiscResult:
    """Complete analysis result for a brake disc image."""

    image_shape: Tuple[int, int]
    outer_diameter: Optional[Circle] = None
    central_bore: Optional[Circle] = None
    bolt_holes: List[Circle] = field(default_factory=list)
    pcd_center: Optional[Tuple[float, float]] = None
    pcd_radius: float = 0.0
    pcd_diameter: float = 0.0
    pcd_std: float = 0.0
    bolt_hole_avg_diameter: float = 0.0
    all_circles: List[Circle] = field(default_factory=list)

    @property
    def num_bolt_holes(self) -> int:
        return len(self.bolt_holes)

    def to_dict(self) -> dict:
        return {
            "image_shape": list(self.image_shape),
            "outer_diameter": self.outer_diameter.to_dict() if self.outer_diameter else None,
            "central_bore": self.central_bore.to_dict() if self.central_bore else None,
            "bolt_holes": [c.to_dict() for c in self.bolt_holes],
            "pcd_center": list(self.pcd_center) if self.pcd_center else None,
            "pcd_radius": self.pcd_radius,
            "pcd_diameter": self.pcd_diameter,
            "pcd_std": self.pcd_std,
            "bolt_hole_avg_diameter": self.bolt_hole_avg_diameter,
            "num_bolt_holes": self.num_bolt_holes,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    """Tunable parameters for the brake-disc detector."""

    # --- preprocessing ---
    blur_kernel: int = 3
    blur_sigma: float = 0.0
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    morph_kernel: int = 3

    # --- contour detection ---
    min_area: float = 5.0
    max_area: float = 1_000_000.0
    min_circularity: float = 0.55
    min_aspect_ratio: float = 0.7
    margin: int = 5

    # --- Hough circles ---
    use_hough: bool = True
    hough_dp: float = 1.0
    hough_min_dist: float = 12.0
    hough_param1: float = 50.0
    hough_param2: float = 22.0  # lower = more sensitive (catches bolt holes)
    hough_min_radius: int = 3
    hough_max_radius: int = 80

    # --- blob detector ---
    use_blob: bool = True
    blob_min_area: float = 10.0
    blob_min_circularity: float = 0.5
    blob_min_inertia: float = 0.5

    # --- post-processing / classification ---
    merge_distance_ratio: float = 0.5  # merge circles whose centers are closer than this * radius
    min_bolt_radius_ratio: float = 0.015  # bolt radius >= this * OD radius
    max_bolt_radius_ratio: float = 0.04   # bolt radius <= this * OD radius
    pcd_tolerance: float = 0.12  # relative std tolerance for PCD consistency


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------
class BrakeDiscDetector:
    """Detect and measure holes in a brake-disc image."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image: np.ndarray) -> BrakeDiscResult:
        """Run the full analysis pipeline on *image* (BGR or grayscale)."""
        if image is None:
            raise ValueError("Input image is None")
        gray = (
            image
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
        result = BrakeDiscResult(image_shape=gray.shape)

        all_circles = self._detect_all_circles(gray)
        result.all_circles = all_circles
        logger.info("Detected %d raw circles", len(all_circles))

        # 1. Outer diameter: use the outer disc contour directly.
        outer = self._find_outer_contour(gray)
        if outer is None:
            # Fallback: largest detected circle.
            outer = max(all_circles, key=lambda c: c.radius) if all_circles else None
        result.outer_diameter = outer
        if outer:
            logger.info("Outer diameter: r=%.1f", outer.radius)

        if outer is None:
            return result

        # 2. Central bore: large dark hole near the disc center
        central = self._find_central_bore(gray, all_circles, outer)
        result.central_bore = central
        if central:
            logger.info(
                "Central bore: center=(%.1f, %.1f), r=%.1f",
                central.center[0], central.center[1], central.radius,
            )

        # 3. Bolt holes: medium circles arranged on a PCD around the center
        bolt_holes, pcd_center, pcd_radius, pcd_std = self._find_bolt_holes(
            all_circles, outer, central
        )
        result.bolt_holes = bolt_holes
        result.pcd_center = pcd_center
        result.pcd_radius = pcd_radius
        result.pcd_diameter = 2.0 * pcd_radius
        result.pcd_std = pcd_std
        if bolt_holes:
            result.bolt_hole_avg_diameter = float(
                np.mean([c.diameter for c in bolt_holes])
            )
        logger.info(
            "Bolt holes: %d found, PCD diameter=%.1f (std=%.1f)",
            len(bolt_holes), result.pcd_diameter, pcd_std,
        )

        return result

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------
    def _detect_all_circles(self, gray: np.ndarray) -> List[Circle]:
        circles: List[Circle] = []

        # Method 1: contour-based
        circles.extend(self._detect_by_contours(gray))

        # Method 2: Hough circles
        if self.config.use_hough:
            circles.extend(self._detect_by_hough(gray))

        # Method 3: SimpleBlobDetector
        if self.config.use_blob:
            circles.extend(self._detect_by_blob(gray))

        # Merge near-duplicate detections from different methods
        circles = self._merge_circles(circles)
        return circles

    def _find_outer_contour(self, gray: np.ndarray) -> Optional[Circle]:
        """Find the outer boundary of the brake disc.

        The largest external contour of the (thresholded) image is fitted to a
        circle.  This is much more reliable than blob/circle detectors for the
        outer diameter, which often gets split or missed.
        """
        # Use Otsu on a slightly blurred image to separate disc from background.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # The disc is bright (255) on a dark background; keep the bright part.
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        # Refine with fitEllipse if the contour is closed and large enough
        if len(largest) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest)
                center, axes, _ = ellipse
                # average of axes gives a more stable radius estimate
                radius = (axes[0] + axes[1]) / 4.0
                cx, cy = center
            except cv2.error:
                pass
        return Circle(
            center=(float(cx), float(cy)),
            radius=float(radius),
            method="outer_contour",
            confidence=1.0,
        )

    def _detect_by_contours(self, gray: np.ndarray) -> List[Circle]:
        blurred = cv2.GaussianBlur(
            gray,
            (self.config.blur_kernel, self.config.blur_kernel),
            self.config.blur_sigma,
        )
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel, self.config.morph_kernel),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        circles: List[Circle] = []
        h, w = gray.shape
        margin = self.config.margin
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area or area > self.config.max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.config.min_circularity:
                continue
            if len(contour) < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(contour)
            except cv2.error:
                continue
            center, axes, _angle = ellipse
            radius = max(axes) / 2.0
            aspect = min(axes) / max(axes) if max(axes) > 0 else 0
            if aspect < self.config.min_aspect_ratio:
                continue
            if not (
                margin < center[0] < w - margin
                and margin < center[1] < h - margin
            ):
                continue
            circles.append(
                Circle(
                    center=(float(center[0]), float(center[1])),
                    radius=float(radius),
                    method="contour",
                    confidence=float(circularity),
                )
            )
        return circles

    def _detect_by_hough(self, gray: np.ndarray) -> List[Circle]:
        circles: List[Circle] = []
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        detected = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.config.hough_dp,
            minDist=self.config.hough_min_dist,
            param1=self.config.hough_param1,
            param2=self.config.hough_param2,
            minRadius=self.config.hough_min_radius,
            maxRadius=self.config.hough_max_radius,
        )
        if detected is None:
            return circles
        detected = np.round(detected[0]).astype("float")
        for x, y, r in detected:
            circles.append(
                Circle(
                    center=(float(x), float(y)),
                    radius=float(r),
                    method="hough",
                    confidence=0.8,
                )
            )
        return circles

    def _detect_by_blob(self, gray: np.ndarray) -> List[Circle]:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.config.blob_min_area
        params.filterByCircularity = True
        params.minCircularity = self.config.blob_min_circularity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.filterByInertia = True
        params.minInertiaRatio = self.config.blob_min_inertia
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        circles: List[Circle] = []
        for kp in keypoints:
            circles.append(
                Circle(
                    center=(float(kp.pt[0]), float(kp.pt[1])),
                    radius=float(kp.size / 2.0),
                    method="blob",
                    confidence=float(kp.response),
                )
            )
        return circles

    # ------------------------------------------------------------------
    # Merging / classification
    # ------------------------------------------------------------------
    def _merge_circles(self, circles: List[Circle]) -> List[Circle]:
        """Merge circles that are very close (likely the same hole)."""
        if len(circles) <= 1:
            return circles
        circles = sorted(circles, key=lambda c: c.radius, reverse=True)
        merged: List[Circle] = []
        for c in circles:
            found = False
            for m in merged:
                dist = c.distance_to(m)
                threshold = self.config.merge_distance_ratio * max(
                    c.radius, m.radius
                )
                if dist < threshold:
                    # weighted average by confidence
                    w_c = max(c.confidence, 1e-6)
                    w_m = max(m.confidence, 1e-6)
                    total = w_c + w_m
                    m.center = (
                        (c.center[0] * w_c + m.center[0] * w_m) / total,
                        (c.center[1] * w_c + m.center[1] * w_m) / total,
                    )
                    m.radius = (c.radius * w_c + m.radius * w_m) / total
                    m.confidence = max(m.confidence, c.confidence)
                    found = True
                    break
            if not found:
                merged.append(c)
        return merged

    def _find_central_bore(
        self, gray: np.ndarray, circles: List[Circle], outer: Circle
    ) -> Optional[Circle]:
        """Find the central bore.

        The central bore is the largest dark circular hole near the disc
        center.  We first try a dedicated dark-region analysis (most reliable
        for a bore), then fall back to the largest detected circle near the
        center.
        """
        disc_center = outer.center
        disc_radius = outer.radius

        # --- Method 1: dark circular region near the center ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, dark = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        best_bore: Optional[Circle] = None
        best_score = -1.0
        for contour in contours:
            area = cv2.contourArea(contour)
            # bore is large but smaller than the disc
            if area < 0.005 * disc_radius ** 2:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.6:
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_to_disc = math.hypot(cx - disc_center[0], cy - disc_center[1])
            # bore must be near the disc center
            if dist_to_disc > 0.2 * disc_radius:
                continue
            radius = math.sqrt(area / math.pi)
            # score: prefer larger, more circular, closer to center
            score = circularity * (1.0 - dist_to_disc / (0.2 * disc_radius)) * radius
            if score > best_score:
                best_score = score
                best_bore = Circle(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    method="bore_dark",
                    confidence=float(circularity),
                )

        if best_bore is not None:
            return best_bore

        # --- Method 2: fallback to largest detected circle near center ---
        candidates = [
            c
            for c in circles
            if c is not outer
            and c.radius >= 0.12 * disc_radius
            and c.distance_to(outer) < 0.25 * disc_radius
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c.distance_to(outer), -c.radius))
        return candidates[0]

    def _find_bolt_holes(
        self,
        circles: List[Circle],
        outer: Circle,
        central: Optional[Circle],
    ) -> Tuple[List[Circle], Optional[Tuple[float, float]], float, float]:
        """Identify bolt holes lying on a pitch circle.

        Uses an iterative centroid approach: candidate bolt holes are those
        whose radius falls in the expected bolt-hole range.  The PCD center is
        estimated as the centroid of the tightest cluster of candidates whose
        distances to that centroid are mutually consistent.

        Returns (bolt_holes, pcd_center, pcd_radius, pcd_std).
        """
        disc_radius = outer.radius
        ref_center = central.center if central else outer.center

        min_bolt_r = self.config.min_bolt_radius_ratio * disc_radius
        max_bolt_r = self.config.max_bolt_radius_ratio * disc_radius

        # Candidate bolt holes: medium-sized, not the outer or bore,
        # and inside the disc.
        candidates = [
            c
            for c in circles
            if c is not outer
            and (central is None or c is not central)
            and min_bolt_r <= c.radius <= max_bolt_r
            and c.distance_to(outer) < disc_radius * 0.85
            and c.distance_to(outer) > disc_radius * 0.04
        ]

        if len(candidates) < 2:
            return candidates, ref_center, 0.0, 0.0

        # --- Iterative centroid + distance clustering ---
        best_group: List[Circle] = []
        best_pcd_radius = 0.0
        best_std = float("inf")
        best_center = ref_center

        # Try multiple starting centers for robustness.
        start_centers = [ref_center]
        # also try centroid of smallest-radius candidates (likely real holes)
        if len(candidates) >= 3:
            smallest = sorted(candidates, key=lambda c: c.radius)[: min(8, len(candidates))]
            start_centers.append(
                (
                    float(np.mean([c.center[0] for c in smallest])),
                    float(np.mean([c.center[1] for c in smallest])),
                )
            )

        for start in start_centers:
            group = list(candidates)
            cx, cy = start
            for _ in range(6):  # iterate to convergence
                dists = np.array(
                    [math.hypot(c.center[0] - cx, c.center[1] - cy) for c in group]
                )
                if len(dists) < 2:
                    break
                med_d = float(np.median(dists))
                mad = float(np.median(np.abs(dists - med_d)))
                # robust tolerance: use median absolute deviation
                tol = max(med_d * self.config.pcd_tolerance, 3.0 * (mad + 1e-6))
                keep = [
                    c for c, d in zip(group, dists) if abs(d - med_d) < tol
                ]
                if len(keep) < 2:
                    break
                if len(keep) == len(group):
                    break
                group = keep
                cx = float(np.mean([c.center[0] for c in group]))
                cy = float(np.mean([c.center[1] for c in group]))

            if len(group) < 2:
                continue

            final_dists = [
                math.hypot(c.center[0] - cx, c.center[1] - cy) for c in group
            ]
            mean_d = float(np.mean(final_dists))
            std_d = float(np.std(final_dists))

            # Prefer the largest group; break ties by lower std.
            if len(group) > len(best_group) or (
                len(group) == len(best_group) and std_d < best_std
            ):
                best_group = group
                best_pcd_radius = mean_d
                best_std = std_d
                best_center = (cx, cy)

        if not best_group:
            return [], ref_center, 0.0, 0.0

        return best_group, best_center, best_pcd_radius, best_std
