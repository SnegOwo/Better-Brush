# Better Brush

AI-powered rotoscoping desktop application built with PySide6 and SAM 2 (Segment Anything Model 2).

Click on a video frame, let SAM 2 generate a mask, propagate it across the timeline, correct with the brush tool, and export production-ready alpha mattes.

## Features

- **Point prompting** -- left-click to include, right-click to exclude. SAM 2 generates masks in real time.
- **Brush correction** -- paint or erase masks frame-by-frame when SAM 2 misses edges.
- **Video propagation** -- propagate masks across the entire video or between correction keyframes.
- **Content-aware presets** -- Anime, Human, and Object modes tune SAM 2 parameters (IoU threshold, mask tokens, stability) for each use case.
- **Auto model selection** -- detects your GPU VRAM and picks the best SAM 2.1 model size with conservative headroom.
- **Export** -- professional export dialog (Ctrl+E): PNG alpha sequences, ProRes 4444, QuickTime Animation, or H.264 (MP4), with resolution scaling and size estimates.
- **Include original audio** -- checkbox in the export dialog muxes the source video's audio track into your export (video formats only).
- **Timeline range selection** -- Shift+drag on the timeline to draw/exact range, or use `I` / `O` to set in/out points at the playhead and `X` to clear. The selection drives the export range and is also remembered in the export dialog (Ctrl+E).
- **Accelerated inference** -- on NVIDIA GPUs SAM 2 runs mixed-precision (bfloat16) with cuDNN/TF32 speedups, and reuses its internal state so correcting a frame and propagating again doesn't re-analyze the video.
- **Project files** -- save your whole session as a `.bbproj` (Ctrl+S): the video reference, every point, keyframe, mask and view state. Reopen it later and keep working.
- **Welcome screen** -- opens on launch with your recent projects plus quick "Start Blank" / "Import Video" actions, and a short tutorial walks new users through the workflow.
- **Preview modes** -- overlay, cutout on checkerboard, or raw video.
- **Hardware detection** -- CUDA, DirectML, or CPU with live GPU info.
- **Drag and drop** -- drop a video file anywhere on the app to open it.
- **Keyboard shortcuts** -- Space or P (play/pause), B (toggle brush/points), arrows (navigate), `[` `]` (brush size), Delete (remove selected point), `I`/`O`/`X` (timeline export range), Ctrl+S (save project).

## Workflow

```
Open Video  ->  Click Subject  ->  Segment  ->  Propagate  ->  Correct  ->  Export
```

## Requirements

- Python 3.11+
- PySide6 >= 6.7.0
- PyTorch >= 2.3.0 (CUDA recommended)
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
python -m pytest tests/ -v                       # all tests
python -m tests.test_brush_editing               # brush + toolbar wiring
python -m tests.test_hardware                     # hardware detection
python -m tests.test_gui_smoke                    # full GUI smoke test (offscreen)
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
- **Inference speed** -- CUDA sessions enable cuDNN benchmark mode, TF32 matmul, and bfloat16 autocast (supported by RTX 30-series+ GPUs); fall back to full precision automatically on older cards. The video predictor state is cached per source file, so repeat propagations on the same clip skip re-initialization.
- **Export** dialog produces PNG sequences with alpha channels, transparent video via ffmpeg (ProRes 4444 with qtrle fallback), or H.264 MP4. With "Include Original Audio" checked, ffmpeg re-muxes the source audio (AAC, 192 kbps) without re-encoding the video.

## Packaging

Build a standalone Windows executable (does not modify the repo, outputs to `dist/`):

```bash
pip install pyinstaller
pyinstaller better_brush.spec
```

## License

Business Source License 1.1 (BUSL-1.1) — see [LICENSE](LICENSE).

Non-production, personal, evaluation, and educational use is free.
Commercial production use requires a license from the Licensor.
Converts to Apache License 2.0 on 2030-09-11.
