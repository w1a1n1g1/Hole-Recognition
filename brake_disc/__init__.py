"""Brake disc hole detection and measurement package."""

from .core import (
    BrakeDiscDetector,
    BrakeDiscResult,
    Circle,
    DetectorConfig,
)
from .visualization import draw_result, generate_pipeline_steps, result_summary_text

__all__ = [
    "BrakeDiscDetector",
    "BrakeDiscResult",
    "Circle",
    "DetectorConfig",
    "draw_result",
    "generate_pipeline_steps",
    "result_summary_text",
]

__version__ = "2.0.0"
