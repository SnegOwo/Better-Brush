# Better Brush — Rotoscoping Desktop App

Open-source rotoscoping application powered by SAM 2 (Segment Anything Model 2).
Click on a video frame to prompt SAM 2, generate a mask, propagate it across the
whole video, then export PNG alpha masks or a transparent ProRes 4444 video.

## Dev Setup (Windows / Python 3.13+)

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# 2. Install dependencies (SAM 2 is installed from GitHub)
pip install -r requirements.txt

# 3. Validate imports (PySide6, torch, SAM 2, ...)
python test_imports.py

# 4. Run the tests
python -m tests.test_hardware       # hardware detection + recommendations
python -m tests.test_integration    # full pipeline: video -> SAM 2 -> export
python -m tests.test_gui_smoke      # headless GUI wiring test (offscreen)

# 5. Launch the app
python -m src.main
```

The first segmentation/propagation downloads the SAM 2.1 checkpoint
(~150 MB for `tiny`) into `checkpoints/`.

## How it works

- **Hardware detection** (`src/hardware/`) picks the best backend
  (CUDA > DirectML > CPU) and recommends a SAM 2.1 model size for your VRAM.
- **Video pipeline** (`src/processing/`): OpenCV `VideoReader` + LRU `FrameBuffer`;
  `init_video()` transparently extracts frames to a JPEG folder
  (`.better_brush_frames/` next to the video) because SAM 2's video loader only
  supports JPEG folders on modern Python (its `.mp4` path requires the
  unmaintained `decord` package).
- **Workers** (`src/processing/inference_worker.py`): segmentation, propagation
  and export run in `QThread` workers with live progress; the model weights are
  loaded once and shared between image and video predictors.
- **Export** (`src/export/exporter.py`): PNG masks with alpha channel and
  transparent video (ProRes 4444, fallback `qtrle`) via ffmpeg.

## Project Architecture

```
src/
├── core/       app bootstrap, config, constants
├── hardware/   GPU/RAM detection + SAM 2 recommendations
├── models/     checkpoint manager + SAM 2 image/video wrappers
├── processing/ video reader, frame buffer, QThread workers
├── ui/         main window, video canvas, timeline
├── export/     PNG + transparent video exporter
└── utils/      logging
```

Heavy processing (SAM 2 inference, propagation, export) always runs in
`QThread` workers to keep the GUI responsive.

## Known limitations

- Multi-object tracking is not implemented yet (`obj_id=1`); brush-based mask
  correction is on the roadmap.
- DirectML requires `torch-directml`, which does not support Python 3.13 yet.
- PyTorch CUDA builds for Python 3.13 currently max out at 2.11.0+cu128
  (install with `--index-url https://download.pytorch.org/whl/cu128`).

## License

MIT — see `LICENSE`.
