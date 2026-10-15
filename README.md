<p align="center">
  <img src="assets/icons/LogoBetterBrush.png" alt="Better Brush" width="132" />
</p>

<h1 align="center">Better&nbsp;Brush</h1>

<p align="center">
  <b>The AI rotoscoping studio for video editors.</b><br/>
  Click&nbsp;→&nbsp;Segment&nbsp;→&nbsp;Propagate&nbsp;→&nbsp;Refine&nbsp;→&nbsp;Remove&nbsp;→&nbsp;Export.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BUSL--1.1-blue" alt="License: BUSL-1.1" /></a>
  <a href="https://sneg.dev/BetterBrush"><img src="https://img.shields.io/badge/get%20pro-%E2%82%AC5%20one--time-2ea043" alt="Get Pro" /></a>
  <img src="https://img.shields.io/badge/version-0.3.0-blueviolet" alt="Version 0.3.0" />
  <img src="https://img.shields.io/badge/platform-Windows-0078d4" alt="Platform: Windows" />
  <img src="https://img.shields.io/badge/python-3.11%2B-3776ab" alt="Python: 3.11+" />
  <img src="https://img.shields.io/badge/tests-359%20passing-brightgreen" alt="Tests: 359 passing" />
</p>

<p align="center">
  <b>100% local.</b> Your footage never leaves your machine — no account, no upload, no telemetry.
</p>

---

**Better Brush** turns any clip into a production-ready alpha matte in a few
clicks. Prompt SAM&nbsp;2 on a single frame, let the model propagate the subject
across the whole timeline, build a **soft matte** with real hair and motion-blur
detail, **paint unwanted subjects out** of the plate, then export transparent
ProRes&nbsp;4444 or a PNG sequence straight into your edit.

It is built first for **anime and AMV workflows** — crisp line art, flat cel
shading, thin hair strands and hard edges — but it handles live-action subjects
just as well.

<div align="center">

**Segment** · **Propagate** · **Refine** · **Inspect** · **Remove** · **Export**

</div>

---

## Table of contents

<details>
<summary><b>Expand</b></summary>

- [Why Better Brush](#why-better-brush)
- [Feature tour](#feature-tour)
  - [Segment](#-segment)
  - [Matte](#-matte)
  - [Remove](#-remove)
  - [Refine](#-refine)
  - [Inspect & QC](#-inspect--qc)
  - [Export & deliver](#-export--deliver)
- [The workflow](#the-workflow)
- [Supported formats](#supported-formats)
- [Models & hardware](#models--hardware)
- [Install](#install)
- [Quick start](#quick-start)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Fine-tune your own anime SAM 2](#fine-tune-your-own-anime-sam-2)
- [Batch render queue](#batch-render-queue)
- [Project files](#project-files)
- [Editions: Lite & Pro](#editions-lite--pro)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)

</details>

---

## Why Better Brush

- **Local-first and private.** Every model runs on your GPU. No cloud, no
  upload, no per-frame billing. Pull your network cable and keep working.
- **Non-destructive.** The source video is never modified. Masks and settings
  live in a `.bbproj` project file, so any decision can be revisited later.
- **Inspectable.** Automatic choices can be reviewed, scored and reverted. The
  Matte QC Inspector shows you *where* the matte is wrong, not just that it is.
- **Keyframe-driven, not frame-by-frame.** Paint one frame, propagate the rest,
  correct only where it drifts.
- **Tuned for AMV.** Crisp line-art presets, scene-cut awareness, temporal
  smoothing that keeps cel edges sharp instead of blurring them.
- **A data engine for your own model.** Corrected masks export straight into a
  SAM 2 fine-tuning dataset — see [Fine-tuning](#fine-tune-your-own-anime-sam-2).

---

## Feature tour

### 🎯 Segment

- **Point prompts** — left-click to include, right-click to exclude. A single
  include point is often enough; add negatives to trim background takeovers.
- **Prompt-aware mask selection** — SAM 2's candidate masks are ranked against
  your include/exclude points instead of always taking the highest IoU blob.
- **Five model families** — SAM 2.1 Tiny → Large, plus **EfficientTAM** and
  **MatAnyone**, with an optional **SAM 3.1 Object Multiplex** adapter.
- **Stable auto-selection** — the hardware profiler picks a tier with real
  headroom (see [Models & hardware](#models--hardware)); selection is
  deterministic, so a busy desktop never silently downgrades your model.
- **Content presets** — **Anime / AMV**, **Human** and **Object**, each tuning
  the decoder for its edge character.
- **Face-part prompts** — one-click mouth / eye isolation for AMV emphasis.
- **Scene-cut detection** — propagation never crosses a cut, so a matte from
  shot A can't leak into shot B.

### 🪄 Matte

- **Learned video matting** — MatAnyone / MatAnyone 2 build true soft alpha
  with hair, motion blur and semi-transparent coverage, tracked through motion.
- **Guided-filter edge cleanup** bounded to a narrow contour band, so interiors
  stay byte-identical and thin structures are protected by a distance transform.
- **Optical-flow temporal smoothing** — Farneback flow aligns neighbouring
  masks and majority-votes only near the contour, killing flicker without
  smearing detail.
- **Per-shot profiles** for model, edge smoothness, temporal smoothing and
  detail preservation.

### 🧽 Remove

- **AI object removal** — a distilled video-inpainting model paints a masked
  subject out and rebuilds the background, processed in cross-faded windows.
- **Auto window sizing** from available VRAM, with low-VRAM tiled decode and
  graceful out-of-memory guidance instead of a raw traceback.
- **Subtitle & logo cleanup** — LaMa (TorchScript/CUDA, ONNX fallback, or
  built-in donor/Telea repair) with tracking bands and seed regions. Subtitle
  and logo detectors are independent pipelines and never feed each other.

### ✏️ Refine

- **Brush (add / erase)** on any frame — `Shift` to erase, `[` / `]` to size.
  A stroke doubles as a SAM 2 prompt, so the model learns intent, not just pixels.
- **Undo / redo** with a full brush stroke as a single undo step.
- **Zoom / pan** and **Overlay / Cutout / Raw** views for edge inspection.

### 🔍 Inspect & QC

- **Matte QC Inspector** (`Ctrl+Shift+M`) — source frame beside the current
  matte with colour-coded evidence for hair loss, background spill, enclosed
  holes and edge flicker, plus raw-vs-current scores.
- **QC Coach** — a local, deterministic, offline assistant that turns plain
  complaints ("the hair is too smooth") into concrete profile settings, with
  explicit polarity so "too smooth" and "too sharp" never collapse together.
- **Performance Monitor** (`Ctrl+Shift+G`) — live GPU utilisation, VRAM,
  temperature, render-time estimate and model-fit safety.
- **Version history** (`Ctrl+Shift+H`) — checksummed automatic and manual
  checkpoints you can compare and restore.

### 📦 Export & deliver

- **PNG sequence (RGBA)**, **ProRes 4444 (MOV, alpha)**, **QuickTime Animation
  (MOV, alpha)** and **H.264 (MP4)** with chroma-key backgrounds.
- **Soft mattes flow straight through** — export derives a coverage ramp from
  the signed distance to the contour and un-premultiplies edge pixels, so hair
  keeps its own colour instead of blending into the backdrop.
- **Edge softness & spill controls** applied at render time on copies only —
  tweak and re-export without re-running segmentation.
- **Batch render queue** for persistent multi-project jobs.

---

## The workflow

```
Import ─▶ Segment ─▶ Propagate ─▶ Refine ─▶ Inspect ─▶ Remove ─▶ Export
              ▲___________________________________|
                       correct & re-propagate
```

1. **Open a clip** — scene cuts are detected automatically.
2. **Prompt the subject** — left-click include, right-click exclude, then
   **Propagate to End**. Propagation stops at every cut.
3. **Matte it** — run learned matting for a soft alpha; tune gamma and grow.
4. **Remove what you don't want** — AI object removal paints masked subjects
   out; Clean Up erases subtitles and logos.
5. **Polish** — despill, colour-cast fix, temporal smoothing, edge softness.
6. **Inspect** — run the Matte QC Inspector before committing to a render.
7. **Export** — `Ctrl+E`, or queue several projects for an overnight batch.

Every step is reversible: re-segment, switch models, change a profile, or roll
back to a checkpoint at any time.

---

## Supported formats

| Stage | Formats |
|---|---|
| **Import (video)** | Anything OpenCV/FFmpeg can decode — MP4, MOV, MKV, WebM, AVI, M4V, WMV, FLV, MPEG, TS … |
| **Import (still)** | PNG, JPG/JPEG, BMP, TIFF, WebP image sequences |
| **Project** | `.bbproj` (JSON + embedded PNG masks) |
| **Export (video)** | ProRes 4444, QuickTime Animation, H.264 MP4 |
| **Export (stills)** | PNG sequence with straight or premultiplied alpha |
| **Audio** | Original audio muxed into video exports (Pro) |

---

## Models & hardware

Better Brush ships multiple model families and picks a sensible one for your
card. You can always override the choice per project or per shot.

| Family | Tiers | Notes |
|---|---|---|
| **SAM 2.1** | Tiny · Small · Base+ · Large | Meta's Segment Anything 2 — the default and the quality leader |
| **EfficientTAM** | TI · S | Lightweight "track anything" fallback for small / integrated GPUs |
| **MatAnyone** | 1 · 2 | Learned video matting for soft human/humanoid alpha |
| **SAM 3.1** | Object Multiplex | Optional, gated; needs the `sam3` extra and a capable CUDA GPU |
| **Custom** | your checkpoint | A model you fine-tuned yourself (see below) |

**How auto-selection works.** The profiler maps your card's **total VRAM**
(a stable hardware property — *not* the fluctuating free-VRAM reading) to the
largest tier that still leaves ~35&nbsp;% headroom, so a modest GPU is never
run at its limit and never silently downgraded mid-session:

| Tier | Comfortable total VRAM |
|---|---|
| EfficientTAM TI | ≥ 1.1 GB |
| EfficientTAM S | ≥ 1.8 GB |
| SAM 2.1 Tiny | ≥ 2.8 GB |
| SAM 2.1 Small | ≥ 5 GB |
| SAM 2.1 Base+ | ≥ 8.5 GB |
| SAM 2.1 Large | ≥ 13.5 GB |
| SAM 3.1 | ≥ 15 GB |

On cards under ~10&nbsp;GB, the video predictor's memory bank and frame buffer
are offloaded to CPU to make propagation fit; on compute capability ≥&nbsp;8 the
session runs under bfloat16 autocast.

> **GPU inference needs the CUDA build of PyTorch.** `pip install torch` on
> Windows yields a CPU-only wheel. Install from
> `https://download.pytorch.org/whl/cu128` (or your CUDA version) instead.

---

## Install

### From source (recommended)

```bash
git clone https://github.com/SnegOwo/Better-Brush.git
cd Better-Brush

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# CUDA-enabled PyTorch first (pick the right cuXXX index for your driver):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python -m src.main
```

The first segmentation downloads a SAM 2.1 checkpoint (~150&nbsp;MB) into
`checkpoints/`. Matting, object-removal and fine-tuning weights download on
first use. Prebuilt Windows bundles ship FFmpeg inside.

### Optional extras

```bash
pip install "scenedetect[opencv]>=0.7.0"   # sharper scene-cut detection
pip install "mediapipe>=0.10.0"            # face-landmark part prompts
pip install "onnxruntime>=1.17"            # LaMa ONNX inpainting fallback
pip install -e ".[sam3]"                    # optional SAM 3.1 adapter
```

| Extra | What it unlocks |
|---|---|
| `scenedetect` | PySceneDetect scene-cutting with a built-in fallback |
| `mediapipe` | One-click mouth / eye part isolation |
| `onnxruntime` | LaMa AI inpainting on CPU |
| `.[sam3]` | Optional SAM 3.1 Object Multiplex model |

---

## Quick start

1. **Open a clip** — `Ctrl+O`, or drag a file onto the window.
2. **Click the subject** — one include point is usually enough on a keyframe.
3. **Propagate** — press **Propagate to End**. Watch the overlay solve.
4. **Refine** — `B` for the brush; paint corrections where it drifts.
5. **Inspect** — `Ctrl+Shift+M`, run the QC Inspector, apply a suggested fix.
6. **Export** — `Ctrl+E`, choose a format, and render.

Prefer a guided tour? Open **Help → Quick Tutorial** inside the app.

---

## Keyboard shortcuts

| Action | Shortcut |
|---|---|
| Open video | `Ctrl+O` |
| Media library | `Ctrl+L` |
| Save project | `Ctrl+S` · **Save As** `Ctrl+Shift+S` |
| Undo / redo | `Ctrl+Z` · `Ctrl+Y` or `Ctrl+Shift+Z` |
| Play / pause | `Space` or `P` |
| Previous / next frame | `←` / `→` |
| Brush tool | `B` · **erase** hold `Shift` · size `[` / `]` |
| Set timeline in / out / clear | `I` / `O` / `X` |
| Matte QC Inspector | `Ctrl+Shift+M` |
| Shot Profiles | `Ctrl+Shift+P` |
| Performance Monitor | `Ctrl+Shift+G` |
| Version History | `Ctrl+Shift+H` |
| Create checkpoint | `Ctrl+Alt+S` |
| Export | `Ctrl+E` |
| Batch Render Queue | `Ctrl+Shift+Q` |

---

## Fine-tune your own anime SAM 2

Better Brush is the **data engine** for a custom model. The loop is Meta's own
"model in the loop": the app proposes masks, you correct the ones it gets wrong,
and those corrections become the training set.

```
1. Open a clip and propagate to a matte you're happy with.
2. File → Save Training Data…        → frames + corrected masks on disk
3. Repeat over 5–20 clips.
4. Train with tools/sam2_training/anime_amv.yaml
5. Drop the checkpoint into checkpoints/custom/ and pick it in AI Model.
```

- **[Fine-tuning guide](docs/FINE-TUNING.md)** — the full recipe, data advice and
  per-GPU settings.
- **[Colab kit](tools/sam2_training/colab/README.md)** — train on a **free Colab
  T4** (or L4/A100) with no local GPU: a notebook, a one-shot driver script and
  a Colab-tuned YAML.

| GPU | Resolution | Notes |
|---|---|---|
| Colab free **T4** 16 GB | 384 | `float16`, 2 frames — fits, slow |
| Colab Pro **L4 / A100** | 512–768 | `bfloat16`, 4 frames |
| RTX 3060 12 GB | 512 | comfortable, ~4–6 it/s |
| RTX 5060 Laptop 8 GB | 384–448 | fits; keep `num_frames` 2–4 |
| 24 GB desktop | 768–1024 | raise resolution, then epochs |

---

## Batch render queue

`Ctrl+Shift+Q` opens a persistent job queue:

- add the current project or browse for other `.bbproj` files;
- save and reuse export presets;
- start, pause, resume, cancel, retry or remove jobs;
- progress and errors stored per job, with retry returning a job to the **front**;
- a job interrupted by a restart returns to **queued** instead of being stranded
  in **running**.

The queue renders the masks already stored in a project; it does not re-run
segmentation. Each job owns its own reader, masks and exporter, so jobs never
interfere with the live session.

---

## Project files

A `.bbproj` is a JSON project holding the video reference, base64/PNG-encoded
masks, prompt points, keyframes, per-object records, scene cuts and processing
profiles.

- Writes are **atomic** (`os.replace` over a same-directory temp file), so a
  reader never sees a half-written project.
- The schema is **backward compatible** with v1/v2/v3; newer fields are additive.
- Shot profiles resolve **object/shot override → object default → project
  default**.

---

## Editions: Lite & Pro

| | **Lite** — free | **Pro** — €5 one-time |
|---|:---:|:---:|
| Point-prompt segmentation | ✅ | ✅ |
| Brush correction & propagation | ✅ | ✅ |
| Learned matting & object removal | ✅ | ✅ |
| PNG / ProRes / QuickTime / MP4 export | ✅ up to 1080p | ✅ up to 8K |
| Multi-object scenes | ❌ (1 object) | ✅ |
| Resize beyond 1920×1080 | ❌ | ✅ |
| Original audio muxing | ❌ | ✅ |
| Library recoding (MP4 ↔ MOV ↔ WebM) | ❌ | ✅ |
| Watermark on video exports | small corner mark | none removed |

The free Lite tier keeps the *real* rendering value — full SAM 2 masking,
propagation and every alpha-capable codec up to 1080p — so a free user can still
produce professional results. Pro unlocks multi-object editing, 4K+, original
audio, recoding and watermark-free output.

Pro keys are issued **offline** (HMAC-signed) and activated in-app under
**Help → Activate Pro…**. No account, no server, no telemetry.

[Get Pro →](https://sneg.dev/BetterBrush)

---

## Architecture

```
QApplication
  └─ MainWindow / QThread workers
      ├─ VideoReader + FrameBuffer        (decode + bounded LRU caches)
      ├─ SAM2Manager                      (image + video predictors)
      ├─ Processing pipeline              (refine, temporal, plates, QC)
      ├─ ProjectManager                   (.bbproj persistence)
      └─ Exporter / RenderQueue           (FFmpeg + PNG, persistent jobs)
```

| Layer | Main files | Responsibility |
|---|---|---|
| Bootstrap | `src/main.py`, `src/core/app.py` | Qt setup, splash, warm-up |
| Application | `src/ui/main_window.py` | Menus, panels, canvas/timeline wiring |
| Domain model | `src/core/scene.py`, `project.py`, `profiles.py` | Objects, masks, keyframes, profiles |
| Model runtime | `src/models/sam2_manager.py` | Model loading, prompting, propagation |
| Video | `src/processing/video_reader.py`, `frame_buffer.py` | Decode, metadata, caches |
| Image processing | `mask_refine.py`, `temporal_smooth.py`, `plates.py` | Edges, flicker, inpainting |
| QC | `matte_qc.py`, `qc_coach.py` | Issue evidence, local coaching |
| Export | `src/export/exporter.py`, `render_queue.py` | FFmpeg/PNG output, persistent jobs |
| Hardware | `src/hardware/detector.py`, `profiler.py`, `monitor.py` | Capability detection, telemetry |

Workers live in `src/processing/inference_worker.py`, never touch widgets
directly, report through Qt signals, and cancel cooperatively via a
`threading.Event`. The full architecture, algorithms and persistence rules are
in the [Developer Guide](docs/DEVELOPER-GUIDE.md).

---

## Documentation

| Guide | Contents |
|---|---|
| [User Guide](docs/USER-GUIDE.md) | The complete editor-facing workflow |
| [Complete Feature Reference](docs/COMPLETE-FEATURES.md) | Every feature and internal system, in depth |
| [Developer Guide](docs/DEVELOPER-GUIDE.md) | Architecture, algorithms, persistence, build |
| [Fine-tuning](docs/FINE-TUNING.md) | Train your own anime/AMV SAM 2 model on one GPU |
| [Colab Kit](tools/sam2_training/colab/README.md) | Fine-tune in the browser, no local GPU needed |
| [Audit Report](docs/AUDIT-REPORT.md) | Recent bug audit and fixes across the pipeline |
| [Roadmap Ideas](docs/ROADMAP-IDEAS.md) | Next-level backlog |

---

## Development

```bash
python -m pytest tests/ -v         # 359 tests, no GPU required
python -m ruff check src/          # lint

pyinstaller --clean better_brush.spec                              # developer build
pyinstaller --clean --noconfirm --distpath dist_public better_brush.spec   # public release
```

- Tests run headless: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/ -q` (Windows).
- The GUI also installs as the `better-brush` command (`python -m src.main`).
- Vendored model packages and their licenses: [src/vendor/README.md](src/vendor/README.md).
- See [CONTRIBUTING-style guardrails](docs/COMPLETE-FEATURES.md#25-technical-guardrails)
  before changing worker, persistence or detection code.

---

## Troubleshooting

| Symptom | Try |
|---|---|
| Mask is too large | Add an exclude point just outside the subject, then re-segment |
| Hair disappears | Lower edge smoothness, raise detail preservation |
| Edge flickers | Enable temporal smoothing; use a two-frame window |
| Exported edges look jagged | Set edge softness to 2 or higher |
| Spill cleaner does nothing | Check the mode isn't **Off** |
| Cleaner eats part of the subject | Lower Tolerance, or **Protect** a box over that detail |
| Fix colour cast did nothing | It needs a small **Area** box over the backdrop, not the whole frame |
| Faint logo remains | Draw a slightly larger region, then **Mask to End** |
| Export sits on "Preparing export" | It is decoding every masked frame before rendering |
| GPU warning / OOM | Close other GPU apps; lower resolution or pick a smaller model |
| Project won't open | Keep the `.bbproj` beside its video, or use **Locate Video** |
| Long session feels risky | Create a named checkpoint first (`Ctrl+Alt+S`) |

More symptoms are covered in the [Complete Feature Reference](docs/COMPLETE-FEATURES.md#23-troubleshooting).

---

## FAQ

**Does my footage leave my machine?**
No. Inference, QC and repair all run locally. No account, upload or cloud
dependency.

**Do I need a GPU?**
A dedicated NVIDIA GPU is strongly recommended for interactive propagation. The
Performance Monitor shows VRAM/thermal headroom so you can size a shot to your
card. EfficientTAM keeps integrated/older GPUs usable.

**What if the matte drifts mid-shot?**
Add a corrective brush stroke on the drifting frame and re-propagate. Corrections
are stored as a new keyframe and stay local.

**Are results reproducible?**
Yes. The same footage, prompts and settings produce the same matte — important
for revisions and team handoffs.

**Can I train it on my own footage?**
Yes — that's the point of the [fine-tuning loop](#fine-tune-your-own-anime-sam-2).

---

## License

Business Source License 1.1 (BUSL-1.1) — see [LICENSE](LICENSE).
Personal, evaluation and educational use is free; commercial production use
requires a license. The license converts to Apache License 2.0 on 2030-09-11.

Third-party components (SAM 2, EfficientTAM, MatAnyone, LaMa, FFmpeg, Qt …)
retain their own licenses; see [src/vendor/README.md](src/vendor/README.md).

---

<div align="center">

**© 2026 Sneg — Sneg HQ, a company of Lugunica** · [sneg.dev](https://sneg.dev) · [Get Pro](https://sneg.dev/BetterBrush)

<sub>Built for editors who would rather roto in minutes than in weekends.</sub>

</div>
