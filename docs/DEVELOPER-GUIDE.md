<div align="center">

# Better Brush Developer Guide

<span style="color:#c084fc">Architecture, algorithms, persistence, and production techniques.</span>

</div>

This document is for maintainers and contributors. For the editor-facing workflow, start with the [User Guide](USER-GUIDE.md).

---

## 1. System shape

Better Brush is a PySide6 desktop application with a deliberately local-first pipeline:

```text
QApplication
  └─ MainWindow / QThread workers
      ├─ VideoReader + FrameBuffer
      ├─ SAM2Manager (image + video predictors)
      ├─ Processing pipeline (mask refinement, temporal smoothing, plates, QC)
      ├─ ProjectManager (.bbproj persistence)
      └─ Exporter (PNG / FFmpeg) and RenderQueue
```

The main thread owns widgets and user state. Long operations run in `QThread` workers and return results through Qt signals. Optional AI engines are loaded lazily and always have deterministic OpenCV fallbacks.

| Layer | Main files | Responsibility |
|---|---|---|
| Bootstrap | `src/main.py`, `src/core/app.py` | Qt setup, splash, background dependency warm-up, window handoff |
| Application state | `src/ui/main_window.py` | Menus, panels, canvas/timeline wiring, worker orchestration |
| Domain model | `src/core/scene.py`, `src/core/project.py`, `src/core/profiles.py` | Objects, masks, keyframes, profiles, project schema |
| Model runtime | `src/models/sam2_manager.py` | SAM 2 loading, prompts, propagation, checkpoints |
| Video | `src/processing/video_reader.py`, `frame_buffer.py` | Decode, metadata, bounded frame caches |
| Image processing | `mask_refine.py`, `temporal_smooth.py`, `plates.py` | Edge cleanup, flicker cleanup, subtitle/logo plates, inpainting |
| QC | `matte_qc.py`, `qc_coach.py` | Issue evidence, comparison, local coaching, tip database |
| Export | `src/export/exporter.py`, `render_queue.py` | FFmpeg/PNG output, persistent jobs, presets, retry/pause |
| Hardware | `src/hardware/detector.py`, `profiler.py`, `monitor.py` | Capability detection, conservative model choice, live telemetry |

---

## 2. Developer paragraph: the technique behind the product

<span style="color:#c084fc">Better Brush separates irreversible-looking AI output from reversible project state.</span> SAM 2 produces the semantic mask, but the editor retains the prompt points, keyframes, object identity, per-shot profiles, cleanup regions, and versioned snapshots. Refinement is a bounded, anime-aware operation: a guided filter proposes an edge-aware change, a thin-feature protection zone prevents hair and line art from being erased, and a catastrophic-area guard returns the original mask when a proposal would collapse or explode the subject. Temporal cleanup uses optical flow to align neighbours and votes only near a contour, so stable interiors are not blurred. Subtitle and logo cleanup are deliberately separate pipelines: captions use temporal percentiles and glyph evidence, while watermarks use static local-contrast evidence and inpainting. This separation is what makes the editor fast enough for iteration and safe enough for a production session: the AI proposes, QC explains, profiles tune, and history restores.

---

## 3. Bootstrap and threading

### Startup sequence

1. `src.core.app.run()` configures logging and creates `QApplication`.
2. A lightweight animated splash is shown.
3. Torch, SAM 2, OpenCV, and exporter modules load on a background thread.
4. `MainWindow` is created after the heavy imports are warm.
5. A `.bbproj` passed on the command line opens directly; otherwise the welcome launcher opens.

### Worker rules

Workers live in `src/processing/inference_worker.py`. They follow these rules:

- never mutate widgets from `run()`;
- emit progress and completion through signals;
- use cooperative `threading.Event` cancellation;
- own temporary readers/engines when work is independent;
- return an `Exception` object through the completion signal instead of raising across the Qt boundary;
- **emit completion on every exit path.** Keep the `try/except` around the whole body with no early `return` inside it, so the failure branch always reaches the same signal. A worker that logs and returns silently leaves the caller waiting forever;
- **restore window state before branching on the result.** `MainWindow` completion handlers call `hide_message()` and `setEnabled(True)` *first*, then check `isinstance(result, Exception)`. A handler that returns early on the error path before restoring leaves the window permanently disabled.

`MainWindow` disables the relevant controls while a worker runs, but the global undo/project state is only mutated in the completion handler.

One Qt trap this codebase has already hit: `QWidget.setEnabled(False)` **cascades to child widgets**, so a progress dialog parented to the main window has its buttons force-disabled too. Any dialog that must stay interactive while the window is disabled has to be re-enabled explicitly after the window is disabled.

---

## 4. Video and caching

`VideoReader` exposes frame-indexed RGB access with OpenCV metadata. `FrameBuffer` is an LRU cache bounded by both frame count and megabytes. `CleanFrameStore` is a separate bounded cache for pre-baked cleanup frames so scrubbing does not re-run inpainting on the GUI thread.

Do not share a `VideoReader` between concurrent workers without adding a lock or a reader-per-job policy. The batch queue intentionally creates a reader per job.

---

## 5. SAM 2 model policy

`SAM2Profiler` maps hardware to a conservative `SAM2Recommendation`. The safety policy keeps a fraction of free VRAM instead of selecting the largest tier that technically fits. Recommendations include:

- model tier and Hydra config;
- checkpoint URL/path;
- device/backend;
- mixed-precision flag;
- content profile parameters;
- IoU and stability thresholds;
- in-memory frame-bank size.

A per-shot profile can force a model. `_ensure_sam_manager` compares the requested model/device with the loaded manager and unloads/recreates it when the profile changes. Do not swap a manager while a worker owns it.

### Optional SAM 3.1

`ModelSize.SAM3_1` maps to Meta's gated `facebook/sam3.1` checkpoint and the separate `sam3` repository. `src/models/sam3_adapter.py` lazily imports the package and adapts `build_sam3_image_model(enable_inst_interactivity=True)` to the existing SAM 2 image-predictor contract. `build_sam3_multiplex_video_predictor` is adapted to the existing session/prompt/propagation methods with normalized point tensors. Checkpoint downloads use `huggingface_hub` and `HF_TOKEN`; the CUDA/VRAM floor is deliberately higher than every SAM 2 tier.

SAM 3.1 is never imported at module import time and never becomes the automatic choice on an unknown or unsupported backend. The SAM 2 path remains the default for CPU, DirectML, and machines without the optional dependency/access token.

Optional NVML telemetry is used by the monitor for live utilization, VRAM, temperature, and pressure. It is not required for inference.

---

## 6. Mask processing

### Guided refinement

`src/processing/mask_refine.py` combines a guided filter with a narrow boundary band:

1. estimate a local guided-filter response;
2. restrict changes to a contour band;
3. protect thin structures using a distance transform;
4. respect `detail_preservation`;
5. reject catastrophic area collapse or explosion.

The default behavior is tuned for anime line art. A higher detail-preservation value widens the protected thin-feature zone.

### Temporal smoothing

`src/processing/temporal_smooth.py` uses Farneback optical flow to warp neighbouring masks into the current frame. Majority voting is applied only near the edge, and thin features are protected. `PostProcessWorker` can resolve a profile per frame, which keeps temporal windows from crossing shot boundaries.

That per-frame resolver returns a **positional 5-tuple** — `(edge_smoothness, temporal_smoothing, detail_preservation, temporal_window, mask_inset)`. The arity is validated in `PostProcessWorker.__init__` via `_normalize_profile_resolver`, which accepts a legacy 4-tuple (padding `mask_inset` with `0`) and rejects anything else with a message naming the expected shape. This matters because a wrong arity is otherwise invisible until `groups.items()` is destructured — after the whole clip has been smoothed, deep inside a worker thread where the error is only logged. Validation happens on the caller's *first* invocation rather than at construction, so building a worker with an idle resolver stays free of side effects.

### Edge finishing

`src/processing/edge_finish.py` runs at **export time only**, on copies. Nothing it does is written back to the cached masks, which is what keeps the controls adjustable and reversible.

1. `soft_coverage(mask, feather)` turns a binary matte into a coverage ramp using the signed distance to the contour: `np.where(mask, inside - 0.5, -(outside - 0.5))`, then `clip(0.5 + signed / feather)`. The `0.5` offset places the 50% crossing on the contour, because `distanceTransform` measures to the nearest opposite *pixel centre*.
2. `decontaminate_edges` un-premultiplies the ramp. A partially covered pixel satisfies `observed = alpha * subject + (1 - alpha) * background`, so the subject colour is recovered by subtracting the median background (estimated from the fully transparent region) and dividing by coverage. Fully transparent and fully opaque pixels are untouched, so the subject interior is never altered.

Two traps worth knowing:

- **The sign must be chosen per side.** `inside - outside` is wrong and silently erases thin structures to zero coverage, because `outside` is large for interior pixels. `np.where(mask, inside, -outside)` is correct.
- **Decontamination only works where there is foreground signal.** Given a pixel that is *pure* background, the maths correctly returns the background. It recovers a subject colour from a genuine `alpha*subject + (1-alpha)*background` blend, which is what an anti-aliased edge is.

The exporter's `_compose_rgba` accepts the resulting coverage. With a solid background it blends toward the key colour so a chroma-key edge is anti-aliased too, instead of stepping.

### Colour spill

`src/processing/despill.py` confines work to a **work area**, so a colour match can never run across the whole frame:

- no region → the morphological edge band, `dilate(mask) - erode(mask)`;
- region in `clean` mode → the user's box, replacing the band;
- region in `protect` mode → everything *except* the box.

`spill_match` scores on **circular hue** rather than raw RGB distance, so blue and dark purple are not confused, and gates on saturation so a near-gray pixel scores near zero regardless of the target. It uses a **plateau** — full score across the inner half of the tolerance window, then a ramp — because a linear falloff means a pixel must sit almost exactly on the target hue, and `alpha_key` (which drops a pixel only when the match beats 50%) then removes almost nothing. Measured on a fringe: a slightly-off target hue went from ~0.0 to 0.81 at tolerance 30.

`apply_shot_transform` is the single entry point every export path uses, so a rendered result always matches the preview. It applies the cast correction first (once the global tint is gone the fringe is much closer to plain background colour, so despill has far less to do), then inset, then the spill pass. `ExportWorker` and the batch render queue both call it — duplicating the transform in each is how they would silently drift apart.

`neutralize_cast` is grey-world against a reference area, with per-channel gains clamped to `[0.6, 1.7]` so a near-black channel cannot blow the correction up. Its weakness is inherent: it treats whatever it is handed as neutral, so a reference covering the whole frame (or a matte covering nearly everything) normalises to the scene average and does almost nothing. The UI asks for a small box for this reason.

### Why not flatten regions for analysis

A recurring suggestion is to average each subject region (hair, skin, costume) into a flat colour so the spill logic has less to cope with. Do not. On the reference anime clip a 9x9 mean moves pixels by only 3.8/255 and the local standard deviation inside a 9x9 window is 6.1 — the regions are already flat, so it buys nothing. Worse, SAM segments on learned edges, texture and semantics, not colour, so handing it a flattened image removes the cues it runs on. Averaging is useful only on an *estimate* of the contaminant, which is why the eyedropper medians a 5x5 patch instead of sampling one pixel.

---

## 7. Subtitle and logo cleanup

### Sampling

`sample_frames` selects deterministic, staggered samples across a clip. Staggering avoids phase-locking to an overlay that appears on alternating frames.

### Subtitle path

The subtitle path is intentionally separate from the logo path:

1. detect a caption band using row energy, temporal residuals, and glyph-shaped components;
2. build a low-percentile temporal plate when the caption flickers;
3. switch to per-frame inpainting when the caption is always on;
4. track moving/resizing caption bands;
5. gate repairs with stable caption-ink evidence.

Do not route a user-requested logo region through the subtitle detector.

### Logo/watermark path

The logo path now uses two complementary signals:

- **strong evidence:** high value or high saturation;
- **faint white evidence:** positive luminance deviation from a local spatial background.

The median/static plate path applies a low static-difference threshold and a small dilation so faint anti-aliased marks do not survive as a ghost. The per-frame guided path uses the same local-contrast rule plus a median reference mask, so a semi-transparent logo gets complete coverage even when one frame has low contrast. The repair is feathered into the frame and can use LaMa when available, with donor-row/OpenCV repair as the safe fallback.

Detection regions are padded slightly to include anti-aliased edges. The subtitle band is passed as an exclusion when both targets are active.

### Inpainting backends

| Backend | When used | Notes |
|---|---|---|
| LaMa TorchScript/CUDA | CUDA plus bundled model | Native-resolution repair, shape-quantized to reduce warm-up cost |
| LaMa ONNX | CPU fallback | Fixed 512×512 input, quality/speed trade-off |
| Donor rows + Telea | Always available | Reproducible texture-preserving repair and regeneration seeds |

The first slow AI attempt can disable the AI engine for the rest of a session. This keeps frame-by-frame exports responsive.

---

## 8. QC and coaching

### Matte QC

`src/processing/matte_qc.py` produces issue evidence for:

- hair/fine-detail loss;
- background spill;
- enclosed holes;
- edge flicker relative to a previous mask.

The report is shape-safe, uses boolean evidence masks, and never raises for an empty or mismatched mask. `MatteComparison` reports after-minus-before deltas so “improved” means lower severity.

### QC Coach

`src/processing/qc_coach.py` is deliberately deterministic and offline:

1. frame statistics estimate coverage and edge detail;
2. the QC report prioritizes issue-specific tips;
3. typed complaints are matched against built-in and custom tips;
4. answers and recent questions persist in the user settings directory;
5. the inspector can emit a settings suggestion for the current shot.

The matcher has explicit polarity for smooth vs sharp complaints. Do not collapse “sharp” and “smooth” into one keyword bag: they require opposite settings.

---

## 9. Project schema and profiles

A `.bbproj` is JSON containing base64 PNG masks. The schema remains backward-compatible with the existing v1/v2/v3 formats; newer fields are additive.

Important fields:

- top-level video reference and metadata;
- prompt points and keyframes;
- per-object records;
- scene cuts;
- default processing profile;
- per-object/per-shot profile records.

`ProjectManager.save` writes to a same-directory temporary file and uses `os.replace`, preventing readers from observing a partially written project. Mask PNGs are deterministic and decoded back to `bool` arrays.

Shot profiles resolve in this order:

1. object/shot override;
2. object default;
3. project default.

Resolution is **whole-profile, never field-level**: each tier replaces the entire `ShotProfile` rather than merging into it. That means a new field is automatically backward-compatible — old records simply carry its default — and there is no partial-override hazard. It also means there is no fallback from an `(object_id, shot_id)` record to the same shot's `object_id=None` record; resolution is per-object lane only. New per-shot code must call `get_for_frame`, which performs the three-tier lookup, not `get_for_shot`, which does not accept an object default.

`ShotProfile.to_dict` is the **only** field list in the persistence path. `src/core/project.py` contains no field names, so every new field must be added to `to_dict`, `from_dict` and `__post_init__` and nothing else. Watch two traps:

- `copy()` round-trips through `to_dict`/`from_dict`, so a field missing from `to_dict` is silently lost on every copy — and `ShotProfileMap` copies on every read and write.
- `ShotProfileDialog.profile()` builds a **fresh** `ShotProfile` from its own widgets. Any field it does not populate is silently reset to the default when the user presses Apply. It reads the colour and edge fields back from the embedded `SpillPanel` for exactly this reason.

The colour/edge fields are `spill_color`, `spill_mode`, `spill_tolerance`, `spill_radius`, `spill_region`, `spill_region_mode`, `cast_neutralize`, `cast_strength` and `edge_softness`. They belong on `ShotProfile` in `src/core/profiles.py`, **not** in `src/hardware/profiler.py`: that module is a frozen hardware recommendation that `profiles.py` already imports from, so putting them there would invert the dependency and risk a cycle, and would bypass `.bbproj` persistence entirely since serialization is only reached through `ShotProfile.to_dict`.

Do not store transient hardware telemetry in the project file. It changes per machine and would make projects noisy.

---

## 10. History, autosave, and queue persistence

### History

`src/core/history.py` stores ordinary `.bbproj` snapshots plus JSON sidecars. Sidecars contain version, label, timestamp, size, and SHA-256. List/load operations verify the checksum and optionally parse the project. Retention prunes the oldest versions.

The main window schedules a debounced checkpoint after dirty edits and also exposes manual checkpoints. A restore opens a verified snapshot and then marks the session dirty so the current project path remains authoritative.

### Render queue

`src/export/render_queue.py` is framework-independent. `RenderQueue` persists presets and jobs atomically, and `execute_render_job` lazily imports project/video/exporter dependencies. `RenderControl.checkpoint()` provides pause/cancel cooperation between frames.

Queue jobs do not share the GUI's SAM manager or frame buffer. A job owns its project read, video reader, masks, frames, and exporter.

---

## 11. Testing and quality gates

Run:

```bash
python -m ruff check src/ tests/
$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/ -q
python -m compileall -q src tests
python tools/check_imports.py
```

There is no configured static type checker in this repository. Keep new public functions annotated and add focused tests for pure logic and headless GUI construction.

Testing conventions:

- synthetic NumPy frames for processing behavior;
- fake NVML objects for telemetry;
- `tmp_path` for history/queue persistence;
- `QT_QPA_PLATFORM=offscreen` for QApplication tests;
- no network or real GPU requirement for unit tests.

---

## 12. Packaging

The Windows bundle is produced with PyInstaller:

```bash
pyinstaller --clean better_brush.spec
pyinstaller --clean --noconfirm --distpath BetterbrushDistV3 better_brush.spec
```

Optional models and FFmpeg are resolved through the constants and detector modules. New modules imported from `MainWindow` or a package `__init__` are included automatically; if a module is only discovered dynamically, add it to the spec explicitly.

---

## 13. Safe extension points

Good additions:

- a new processing function with pure-array tests;
- a new export preset field with `to_dict`/`from_dict` coverage;
- a new QC tip with an explicit polarity and setting recommendation;
- a new project field with a default and backward-compatible load path;
- a new Inspect action with a keyboard shortcut and tutorial step.

Risky additions:

- sharing a reader/manager across workers;
- mutating widgets from a worker;
- changing subtitle detection while “fixing” logos;
- bumping the project schema without a migration/default path;
- writing a large frame cache without a memory bound.

---

<div align="center">

<span style="color:#555560">End of developer guide</span>

</div>
