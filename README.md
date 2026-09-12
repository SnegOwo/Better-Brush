<p align="center">
  <img src="assets/icons/LogoBetterBrush.png" alt="Better Brush" width="128" />
</p>

<h1 align="center">Better Brush</h1>

<p align="center">
  <b>AI rotoscoping studio</b> for video editors · Windows desktop app
  <br/>
  Click. Segment. Propagate. Correct. Export.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BUSL--1.1-blue" alt="License: BUSL-1.1" /></a>
  <a href="https://sneg.dev/BetterBrush"><img src="https://img.shields.io/badge/get%20pro-%E2%82%AC5-green" alt="Get Pro" /></a>
  <img src="https://img.shields.io/badge/platform-Windows-informational" alt="Platform: Windows" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python: 3.11+" />
</p>

---

Better Brush is a professional AI rotoscoping tool from **Sneg HQ**. It turns any
video clip into production-ready alpha mattes with a few clicks: prompt SAM 2 to
segment your subject, propagate the mask across the timeline, refine edges with
the brush, and export transparent ProRes 4444 or PNG sequences.

Everything runs locally — images never leave your machine.

---

## Editions

| | **Lite** — free | **Pro** — €5 one-time |
|---|---|---|
| Point-prompt segmentation | ✔ | ✔ |
| Brush correction & propagation | ✔ | ✔ |
| PNG / ProRes / MP4 export | ✔ | ✔ (up to 8K) |
| **Watermark on exports** | ✔ | ✘ removed |
| Multi-object scenes | ✘ | ✔ |
| Original audio muxing | ✘ | ✔ |
| Library video recoding | ✘ | ✔ |

Pro keys are issued offline and activated in-app. [Get Pro](https://sneg.dev/BetterBrush).

## Highlights

- **Point-prompt masks** — click to include, right-click to exclude; live SAM 2 inference
- **Auto model selection** — detects your GPU and picks the best SAM 2.1 model with safe VRAM headroom
- **Video propagation** — masks follow the subject across the whole clip
- **Multi-object scenes** — independent layers per subject; export combined, single, or separate
- **Pro export pipeline** — PNG alpha sequences, ProRes 4444, QuickTime Animation, H.264, audio mux
- **Project files** — `.bbproj` saves every point, keyframe, mask, and view state
- **CUI-friendly shortcuts** — full keyboard workflow with undo/redo and zoom/pan

## Getting Started

```bash
git clone https://github.com/SnegOwo/Better-Brush.git
cd Better-Brush

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
python -m src.main
```

The first segmentation downloads a SAM 2.1 checkpoint (~150 MB for the tiny model)
into `checkpoints/`. For best performance, install CUDA-enabled PyTorch
(see `requirements.txt`). The release bundle ships with FFmpeg built in.

## Development

```bash
python -m pytest tests/ -v     # run the test suite
pyinstaller --clean better_brush.spec   # build the Windows bundle
```

## License

Business Source License 1.1 (BUSL-1.1) — see [LICENSE](LICENSE).

© 2026 Sneg — Sneg HQ, a company of Lugunica · [sneg.dev](https://sneg.dev)

Personal, evaluation, and educational use is free. Commercial production use
requires a license. Converts to Apache License 2.0 on 2030-09-11.