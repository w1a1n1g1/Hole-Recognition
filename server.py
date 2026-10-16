#!/usr/bin/env python3
"""Flask backend for the brake-disc analyzer web UI.

Run with::

    python server.py

Then open http://localhost:5000 in a browser.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import uuid
from dataclasses import asdict
from typing import Any, Dict

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from brake_disc import (
    BrakeDiscDetector,
    DetectorConfig,
    draw_result,
    generate_pipeline_steps,
    result_summary_text,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("brake_disc.server")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


def _config_from_payload(payload: Dict[str, Any]) -> DetectorConfig:
    """Build a DetectorConfig from a JSON payload (only set provided keys)."""
    cfg = DetectorConfig()
    field_names = {f.name for f in DetectorConfig.__dataclass_fields__.values()}
    for key, value in payload.items():
        if key in field_names and value is not None:
            try:
                setattr(cfg, key, value)
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring invalid config %s=%r: %s", key, value, exc)
    return cfg


def _image_to_base64(image: np.ndarray) -> str:
    """Encode a BGR image array to a base64 data-URI string."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return the default detector configuration as JSON."""
    return jsonify(asdict(DetectorConfig()))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze an uploaded brake-disc image.

    Expects a multipart form with a ``file`` field and optional ``config``
    JSON field.  Returns the annotated image (base64) and measurements.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Cannot decode image"}), 400

    # Optional config
    cfg = DetectorConfig()
    config_raw = request.form.get("config")
    if config_raw:
        try:
            cfg = _config_from_payload(json.loads(config_raw))
        except json.JSONDecodeError:
            logger.warning("Invalid config JSON, using defaults")

    draw_all = request.form.get("draw_all", "false").lower() == "true"
    include_steps = request.form.get("steps", "false").lower() == "true"

    # Run detection
    try:
        detector = BrakeDiscDetector(cfg)
        result = detector.analyze(image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    annotated = draw_result(image, result, draw_all_circles=draw_all)

    response = {
        "ok": True,
        "image_shape": list(image.shape[:2]),
        "annotated": _image_to_base64(annotated),
        "result": result.to_dict(),
        "summary": result_summary_text(result),
    }

    if include_steps:
        steps = generate_pipeline_steps(image, result, cfg)
        response["steps"] = {
            name: _image_to_base64(img) for name, img in steps.items()
        }

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting brake-disc analyzer on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
