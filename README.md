<p align="center">
  <img src="assets/icons/LogoBetterBrush.png" alt="Better Brush" width="120" />
</p>

<h1 align="center">Better Brush</h1>

<p align="center">
  <b>The AI rotoscoping studio for video editors.</b><br/>
  Click. Segment. Matte. Propagate. Remove. Export.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BUSL--1.1-blue" alt="License: BUSL-1.1" /></a>
  <a href="https://sneg.dev/BetterBrush"><img src="https://img.shields.io/badge/get%20pro-%E2%82%AC5%20one--time-green" alt="Get Pro" /></a>
  <img src="https://img.shields.io/badge/platform-Windows-informational" alt="Platform: Windows" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python: 3.11+" />
  <img src="https://img.shields.io/badge/tests-350%20passing-brightgreen" alt="Tests" />
</p>

<p align="center">
  Everything runs locally — your footage never leaves your machine.
</p>

---

## What it does

Better Brush turns any clip into a production-ready alpha matte with a few
clicks: prompt SAM 2 to segment your subject, build a **soft matte** around it,
propagate across the whole timeline, **paint subjects out** of the shot, refine
edges with the brush, and export transparent ProRes 4444 or PNG sequences.

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

**Segment**
- Point-prompt masks — click to include, right-click to exclude, live inference
- **Five model families** — SAM 2.1 Tiny → Large, SAM 3.1 Object Multiplex,
  and **EfficientTAM** for machines that cannot host SAM 2.1 Tiny
- Auto model selection from your GPU's *free* VRAM with a stability-first headroom policy
- **Train your own anime-tuned model** — Better Brush is the data engine
  (see [Fine-tuning](docs/FINE-TUNING.md))
- Face-part prompts (mouth / eyes), scene-cut detection, shot profiles

**Matte**
- **Learned video matting** — MatAnyone / MatAnyone 2 build soft alpha with
  hair, motion blur and partial coverage, tracked through the clip
- Edge gamma and matte grow controls, per shot
- Optical-flow temporal smoothing, despill, cast neutralization, guided-filter refinement

**Remove**
- **AI object removal** — a distilled video inpainting model paints the masked
  subject out and rebuilds the background, in cross-faded windows
- Auto window sizing from free VRAM, low-VRAM tiled decode, graceful OOM guidance
- Subtitle / logo cleanup with LaMa inpainting on GPU, plus tracking bands and seeds

**Correct**
- Brush (add / erase) on any frame, points, boxes derived from painted masks
- Undo/redo, zoom/pan, overlay / cutout / raw views
- Matte QC inspector with issue overlays and a coach that answers typed complaints

**Export**
- ProRes 4444, QuickTime Animation, H.264 with chroma-key backgrounds, PNG alpha
- Soft mattes flow straight through to the exporter (no double feathering)
- Batch render queue: persist projects, pause, resume, retry
- Version history with checksummed, restorable checkpoints

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

The first segmentation downloads a SAM 2.1 checkpoint (~150 MB) into
`checkpoints/`. For best performance install CUDA-enabled PyTorch; the release
bundle ships FFmpeg built in. Matting, removal and fine-tuning models download
on first use.

## Production Workflow

1. **Open a clip** — scene cuts are detected automatically.
2. **Prompt the subject** — left-click include, right-click exclude, then
   `Propagate to End` (propagation never crosses a cut).
3. **Matte it** — Matting → Run Matting for a soft alpha; tune gamma / grow.
4. **Remove what you don't want** — Object Removal paints masked subjects out.
5. **Clean up** — subtitles / logos, despill, cast fix, edge softness.
6. **Inspect** — Matte QC Inspector (`Ctrl+Shift+M`) before you commit a render.
7. **Export** — `Ctrl+E`, or queue several projects for a batch render.

## Documentation

| Guide | Contents |
|---|---|
| [User Guide](docs/USER-GUIDE.md) | The complete editor-facing workflow |
| [Developer Guide](docs/DEVELOPER-GUIDE.md) | Architecture, algorithms, persistence, build |
| [Fine-tuning](docs/FINE-TUNING.md) | Train your own anime/AMV SAM 2 model on one GPU |
| [Roadmap Ideas](docs/ROADMAP-IDEAS.md) | Next-level backlog |

## Development

```bash
python -m pytest tests/ -v            # 350 tests, no GPU required
python -m ruff check src/             # lint
pyinstaller --clean better_brush.spec # build the Windows bundle
```

Vendored model packages and their licenses are documented in
[src/vendor/README.md](src/vendor/README.md).

## License

Business Source License 1.1 (BUSL-1.1) — see [LICENSE](LICENSE).

© 2026 Sneg — Sneg HQ, a company of Lugunica · [sneg.dev](https://sneg.dev)

Personal, evaluation, and educational use is free. Commercial production use
requires a license. Converts to Apache License 2.0 on 2030-09-11.
