<p align="center">
  <img src="assets/icons/LogoBetterBrush.png" alt="Better Brush" width="128" />
</p>

<h1 align="center">Better Brush</h1>

<p align="center">
  <b>AI-powered rotoscoping desktop app</b> · PySide6 + SAM 2
  <br/>
  Click. Segment. Propagate. Correct. Export.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BUSL--1.1-blue" alt="License: BUSL-1.1" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/version-0.1.0--beta-orange" alt="Version: 0.1.0-beta" /></a>
  <img src="https://img.shields.io/badge/platform-Windows-informational" alt="Platform: Windows" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python: 3.11+" />
  <img src="https://img.shields.io/badge/framework-PySide6-9cf" alt="PySide6" />
</p>

---

Better Brush turns a video clip into **production-ready alpha mattes**. Point at
your subject, let SAM 2 generate the mask, propagate it across the timeline, fix
edges with the brush, and export transparent ProRes 4444 / PNG sequences — in one
desktop app.

---

## Features

| | Capability |
|---|---|
| **Point prompting** | left-click to include, right-click to exclude; SAM 2 masks in real time |
| **Brush correction** | paint or erase masks frame-by-frame |
| **Video propagation** | across the whole clip or between correction keyframes |
| **Content presets** | Anime / Human / Object tune SAM 2 thresholds per use case |
| **Auto model selection** | detects VRAM and picks the best SAM 2.1 model with safe headroom |
| **Accelerated inference** | bf16 + TF32/CuDNN on NVIDIA, cached predictor state |
| **Pro export dialog** | Ctrl+E — PNG alpha, ProRes 4444, QuickTime Animation, H.264, audio mux |
| **Project files** | `.bbproj` saves every point, keyframe, mask and view state |
| **Hardware detection** | CUDA, DirectML, or CPU with live GPU info |
| **Multi-object scenes** | several subjects, each its own mask layer; export combined / single / separate |
| **Undo / Redo** | full history for points and brush strokes |
| **Zoom & pan** | Ctrl+scroll to zoom, middle-drag to pan, from any tool |
| **Keyboard shortcuts** | Space/P play, B brush/points, arrows navigate, `[` `]` brush size, Delete remove point, `I`/`O`/`X` range, Ctrl+S save, Ctrl+E export |

## Workflow

```
Open Video  →  Click Subject  →  Segment  →  Propagate  →  Correct  →  Export
```

## Requirements

- Python 3.11+
- PySide6 ≥ 6.7.0
- PyTorch ≥ 2.3.0 (CUDA recommended)
- SAM 2 (installed from GitHub)
- FFmpeg (for video export)

## Installation

```bash
# Clone the repository
git clone https://github.com/SnegOwo/Better-Brush.git
cd Better-Brush

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Validate imports
python test_imports.py

# Launch the app
python -m src.main
```

The first segmentation downloads the SAM 2.1 checkpoint (~150 MB for tiny) into `checkpoints/`.

### GPU setup (optional)

For NVIDIA GPUs, install CUDA-enabled PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For AMD/Intel GPUs on Windows, install DirectML support:

```bash
pip install torch-directml
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Architecture

```
src/
  core/           App bootstrap, configuration, constants
  hardware/       GPU/RAM detection, SAM 2 model recommendations
  models/         Checkpoint manager, SAM 2 image/video wrappers
  processing/     Video reader, frame buffer, inference workers (QThread)
  ui/             Main window, video canvas, timeline, theme, components
  export/         PNG mask + transparent video exporter
  utils/          Logging configuration
```

All heavy processing (SAM 2 inference, propagation, export) runs in QThread workers to keep the GUI responsive.

## How It Works

- **Hardware detection** picks the best backend (CUDA > DirectML > CPU) and conservatively recommends a SAM 2.1 model size, keeping ~35% VRAM headroom to prevent GPU overheat and driver crashes.
- **Video pipeline** uses OpenCV for reading and an LRU frame buffer for caching. SAM 2's video predictor works with JPEG frame extraction.
- **Workers** handle segmentation, propagation, and export off the main thread with live progress reporting.
- **Inference speed** — CUDA sessions enable cuDNN benchmark mode, TF32 matmul, and bfloat16 autocast (supported by RTX 30-series+ GPUs); fall back to full precision automatically on older cards. The video predictor state is cached per source file, so repeat propagations on the same clip skip re-initialization.
- **Export** dialog produces PNG sequences with alpha channels, transparent video via ffmpeg (ProRes 4444 with qtrle fallback), or H.264 MP4. With "Include Original Audio" checked, ffmpeg re-muxes the source audio (AAC, 192 kbps) without re-encoding the video.

## Packaging

Build a standalone, self-contained Windows bundle (FFmpeg, PySide6, PyTorch and
SAM 2 are all shipped inside — end users install nothing extra):

```bash
pip install pyinstaller
pyinstaller --clean better_brush.spec
```

Output lands in `dist/BetterBrush/` as a ready-to-run folder (`BetterBrush.exe`).

## License

Business Source License 1.1 (BUSL-1.1) — see [LICENSE](LICENSE).

© 2026 Sneg · solo developer — [sneg.dev](https://sneg.dev) · Sneg HQ, a company of Lugunica.

Non-production, personal, evaluation, and educational use is free.
Commercial production use requires a license from the Licensor.
Converts to Apache License 2.0 on 2030-09-11.