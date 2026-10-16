# Brake Disc Hole Analyzer

A computer-vision tool for detecting and measuring holes in brake-disc images.
It identifies the **outer diameter (OD)**, the **central bore**, the **bolt
holes**, and computes the **pitch-circle diameter (PCD)** from the bolt-hole
centers.

## Features

- **Multi-method circle detection** — combines contour analysis, Hough circle
  transform, and OpenCV's `SimpleBlobDetector`, then merges duplicates.
- **Brake-disc-aware geometry** — finds the OD via the outer contour, the
  central bore via dark-region analysis, and groups bolt holes onto their PCD
  using an iterative centroid + robust-distance clustering.
- **Multiple interfaces** — a modern **web UI** (`server.py`), a command-line
  tool (`cli.py`), a Tkinter GUI (`gui.py`), and a reusable Python library
  (`brake_disc/`).
- **Exportable results** — annotated images, plus JSON and CSV measurements.
- **Canny edge-detection experiment** — `canny_edge_detection.py` sweeps
  threshold combinations and compares blurred vs. unblurred results.

## Project layout

```
.
├── brake_disc/              # Core library
│   ├── __init__.py
│   ├── core.py              # Detector, config, data classes
│   └── visualization.py     # Annotated rendering + text summary
├── static/                  # Web UI assets
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── server.py                # Flask web server (Web UI)
├── cli.py                   # Command-line interface
├── gui.py                   # Tkinter graphical interface
├── canny_edge_detection.py  # Canny threshold experiment
├── requirements.txt
├── README.md
├── 2.png                    # Sample brake-disc image
└── results/                 # Generated outputs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web UI (推荐)

```bash
python server.py
```

然后在浏览器中打开 http://localhost:5000 。

功能：
- 拖拽或点击上传刹车盘图像
- 实时调节检测参数（圆度、Hough 阈值、螺栓孔半径范围等）
- 一键分析，自动标注外径、中心孔、螺栓孔与节圆（PCD）
- 查看测量结果卡片、螺栓孔明细表与统计信息
- 切换"标注结果 / 原始图像"视图
- 导出标注图像或 JSON 测量数据

### Command line

```bash
# Analyze a single image (prints measurements + saves annotated image)
python cli.py 2.png

# Save JSON and CSV measurements
python cli.py 2.png --json results/out.json --csv results/holes.csv

# Batch-process a directory
python cli.py images/ --output-dir results/

# Tweak detection parameters
python cli.py 2.png --hough-param2 25 --min-bolt-ratio 0.02 --max-bolt-ratio 0.045
```

### GUI

```bash
python gui.py
```

Select an image, click **Analyze**, inspect the measurements, and save the
annotated result.

### Library

```python
import cv2
from brake_disc import BrakeDiscDetector, draw_result, result_summary_text

img = cv2.imread("2.png")
detector = BrakeDiscDetector()
result = detector.analyze(img)

print(result_summary_text(result))
annotated = draw_result(img, result)
cv2.imwrite("annotated.jpg", annotated)
```

## Measurements

All sizes are reported in **pixels**. Key outputs:

| Quantity | Description |
|----------|-------------|
| Outer Diameter (OD) | Diameter of the disc's outer edge |
| Central Bore | Diameter of the center hole |
| Bolt holes | Count, center, radius, and diameter of each bolt hole |
| PCD diameter | Diameter of the circle through all bolt-hole centers |
| PCD std | Standard deviation of bolt-hole distances from the PCD center |

## Canny edge-detection experiment

```bash
python canny_edge_detection.py [image_path]
```

Runs Canny with five threshold combinations (with Gaussian blur) plus a
no-blur comparison, saving side-by-side figures in `results/`.

## Configuration

`DetectorConfig` (in `brake_disc/core.py`) exposes tunables for preprocessing,
each detection method, and bolt-hole classification. Pass overrides via the
CLI flags or by constructing a custom config in code.

## Example output

```
===== Brake Disc Analysis Result =====
Image size: 720x481
Outer Diameter : 864.5 px (center=(359.5, 240.0))
Central Bore   : 88.1 px (center=(341.9, 253.9))
Bolt holes     : 5
Avg bolt dia   : 23.8 px
PCD diameter   : 159.9 px (std=1.6)
PCD center     : (339.9, 253.6)

--- Bolt hole details ---
  #1: center=(395.1, 311.9), radius=13.8, diameter=27.5
  ...
```
