# BetterBrush — Complete Feature & Internals Reference

<span style="color:#4a9eff">Every feature, every workflow, every internal system.</span>

---

## Table of contents

1. [Product overview](#1-product-overview)
2. [The core workflow](#2-the-core-workflow)
3. [Prompting: points and brush](#3-prompting-points-and-brush)
4. [Segmentation](#4-segmentation)
5. [Propagation](#5-propagation)
6. [Brush correction](#6-brush-correction)
7. [Post-processing and shot profiles](#7-post-processing-and-shot-profiles)
8. [Edge finishing and colour spill](#8-edge-finishing-and-colour-spill)
9. [Subtitle and logo cleanup](#9-subtitle-and-logo-cleanup)
10. [Matte QC Inspector](#10-matte-qc-inspector)
11. [QC Coach](#11-qc-coach)
12. [Performance Monitor](#12-performance-monitor)
13. [Export](#13-export)
14. [Batch Render Queue](#14-batch-render-queue)
15. [Version history and recovery](#15-version-history-and-recovery)
16. [Multi-object scenes](#16-multi-object-scenes)
17. [Models and hardware](#17-models-and-hardware)
18. [Project files (.bbproj)](#18-project-files-bbproj)
19. [Editions and pricing](#19-editions-and-pricing)
20. [Fine-tuning: train your own anime SAM 2](#20-fine-tuning-train-your-own-anime-sam-2)
21. [Keyboard shortcuts](#21-keyboard-shortcuts)
22. [Preview modes](#22-preview-modes)
23. [Troubleshooting](#23-troubleshooting)
24. [Architecture overview](#24-architecture-overview)
25. [Technical guardrails](#25-technical-guardrails)

---

## 1. Product overview

BetterBrush is a **local-first AI rotoscoping tool** for video editors, purpose-built for AMV and anime workflows. It takes a raw video clip and produces a clean alpha matte — a per-pixel mask that separates the subject from the background — using the SAM 2 family of segmentation models.

The core promise: **rotoscoping, without the roto work.** Instead of manually drawing masks frame by frame, you click the subject once, and the AI propagates the matte across the entire clip. You correct where it drifts with a brush, and export a production-ready alpha.

### What makes it different

- **100% local** — footage never leaves the machine. All inference runs on-device.
- **Non-destructive** — the source video is never modified. Masks are stored separately in a project file.
- **Inspectable** — every automatic decision can be reviewed, corrected, and reverted.
- **Production-oriented** — ProRes 4444, PNG sequences, QuickTime Animation, H.264, up to 8K.

---

## 2. The core workflow

The entire BetterBrush workflow follows five verbs:

```
Segment → Propagate → Correct → Export
```

1. **Segment** — click the subject. A binary matte appears in a fraction of a second.
2. **Propagate** — the matte tracks the subject across the whole clip.
3. **Correct** — brush over frames where the tracking drifts.
4. **Inspect** — open the QC Inspector to verify edge quality.
5. **Export** — render the final alpha in your chosen format.

Each step is reversible. You can re-segment, change the model, adjust profiles, and restore from a checkpoint at any time.

---

## 3. Prompting: points and brush

### Point tool (default)

The primary way to tell the model what to segment.

- **Left-click** — add an **include point** (shown as a green dot). The model should include this area in the matte.
- **Right-click** — add an **exclude point** (shown as a red dot). The model should exclude this area.
- Press **Segment Frame** to generate the matte.

A single include point is often enough. Add exclude points for background areas that are being wrongly included. The model respects these prompts when selecting among candidate masks.

### Brush tool

Press `B` to switch to the Brush tool.

- **Paint** — add matte pixels by dragging the cursor.
- **Shift + paint** — erase matte pixels.
- **`[` / `]`** — decrease / increase brush size.
- **Eraser checkbox** — alternative to holding Shift.

The brush works on the current frame and also serves as a prompt for a SAM 2 refinement pass — painting over a missing area tells the model to include it in the next segmentation.

A brush stroke creates **one undo step** when the stroke ends, so you can revert an entire correction with a single Ctrl+Z.

---

## 4. Segmentation

Segmentation is the process of converting a single frame's point prompts into a binary matte.

### How it works

1. The frame is sent to the selected SAM 2 model.
2. The model generates one or more candidate masks based on the point prompts.
3. A **prompt-aware mask selector** picks the candidate that best respects the include/exclude points.
4. The selected mask is stored and displayed as a binary matte (every pixel is either fully opaque or fully transparent).

### When the mask is wrong

- Add an **exclude point** on the background that is being included, then re-segment.
- Add an **include point** inside the missing part of the subject.
- Use the **Brush** for a small local correction rather than redoing the whole shot.
- Try a different content preset (Anime/AMV, Human, Object).

### Content presets

The content preset changes how the SAM 2 decoder evaluates candidate masks:

| Preset | Tuned for |
|---|---|
| **Anime / AMV** | Crisp line art, hard edges, flat cel shading |
| **Human** | Faces, hair, skin detail |
| **Object** | General-purpose, safe starting point |

---

## 5. Propagation

Propagation carries a keyframe matte forward (or backward) through the clip.

### How it works

- **◀ Prop** and **Prop ▶** — step the mask one frame in the chosen direction.
- **Propagate to End** — automatically track from the current keyframe to the end of the clip.
- The model uses its **video predictor** to maintain object identity across frames, remembering what it has seen.

### Keyframes and timeline

- **Keyframe diamonds** on the timeline show where an explicit mask exists.
- Between keyframes, the model interpolates.
- Adding a second keyframe is useful when the subject changes shape quickly — it gives the model a new reference point.

### Scene cuts

- **Detect Cuts** finds shot boundaries in the clip.
- Propagation stops at a cut — it never crosses a scene boundary.
- A cut also gives the shot-profile system a clear processing boundary, preventing flicker from leaking between edits.
- You can add cuts manually even when the shot doesn't change — this is useful for organising processing.

### Timeline export range

- **Shift-drag** the timeline to create an export range.
- **`I` / `O`** — set timeline in/out points.
- **`X`** — clear the range.

---

## 6. Brush correction

When propagation drifts — the matte slips off the subject, or includes background — you correct it with the brush.

### How it works

1. Navigate to the frame where the mask is wrong.
2. Switch to the Brush tool (`B`).
3. Paint over the area that should be in the matte (BRUSH +) or out (ERASE).
4. The correction is stored as a new keyframe.
5. Propagation continues from the corrected frame.

### Brush as SAM 2 prompt

The brush is not just a pixel paint tool. When you paint, the stroke is also used as a **prompt for a SAM 2 refinement pass**. This means the model understands the semantic intent of your correction — "this area is part of the subject" — rather than just blindly copying pixels.

### Undo

- **Ctrl+Z** — undo the last brush stroke or point change.
- **Ctrl+Y** / **Ctrl+Shift+Z** — redo.
- A brush stroke is one undo step when the stroke ends, so an entire correction can be reverted at once.

---

## 7. Post-processing and shot profiles

After propagation, two related operations clean up the result:

### Edge cleanup

Uses a **guided filter** to smooth the matte boundary while preserving edges. The process:

1. Estimate a local guided-filter response.
2. Restrict changes to a narrow contour band around the mask edge.
3. Protect thin structures (hair, line art) using a distance transform.
4. Respect the `detail_preservation` setting.
5. Reject catastrophic proposals (if a change would collapse or explode the subject, the original mask is kept).

### Temporal smoothing

Uses **Farneback optical flow** to align neighbouring masks with the current frame, then applies majority voting only near the contour. Stable interiors are not blurred.

### Global settings

The **Edge smoothness** slider is a starting point for the whole project. It can be changed at any time — release the slider or press **Apply** to re-run the current shot's cleanup on existing masks.

### Shot profiles

For mixed footage (anime + live action + graphics), open **Inspect → Shot Profiles** (`Ctrl+Shift+P`) to set values per object/shot:

| Setting | Effect |
|---|---|
| **Model** | Auto or a fixed SAM 2 tier |
| **Edge smoothness** | 0 = raw; higher values clean the boundary more |
| **Temporal smoothing** | Reduces frame-to-frame flicker |
| **Detail preservation** | Protects hair and fine line art from over-smoothing |
| **Edge softness** | Anti-aliases the exported matte (see §8) |
| **Spill cleaner** | Removes background colour from the matte edge (see §8) |

Profiles persist inside the project file. Resolution is **whole-profile, never field-level** — each tier replaces the entire ShotProfile rather than merging into it.

---

## 8. Edge finishing and colour spill

Two separate problems, both handled at **export time only** (on copies, never on saved masks).

### Edge softness — jagged edges

An exported matte is normally binary: every pixel is fully opaque or fully gone. Fine hair and motion-blur edges look like a staircase.

**Edge softness** fixes this by building a **coverage ramp** from the distance to the matte boundary:

1. Compute the signed distance to the contour: `np.where(mask, inside - 0.5, -(outside - 0.5))`.
2. Clip to `[0.5 - feather/2, 0.5 + feather/2]` — the 50% crossing sits exactly on the contour.
3. **Decontaminate edges**: a partially covered pixel satisfies `observed = alpha * subject + (1 - alpha) * background`. The subject colour is recovered by subtracting the median background and dividing by coverage.

| Value | Effect |
|---|---|
| **0** | Hard matte — useful for a matte-only workflow |
| **2** | **Default.** Right for most work |
| **3–4** | Very soft hair, heavy motion blur |

Only affects exported video. Turning it off restores the original masks exactly.

### Spill cleaner — coloured fringe

A subject shot against a saturated background picks up a fringe of that colour along its silhouette.

1. Set the mode: **Off**, **Despill** (keep coverage, remove hue) or **Alpha key** (fade the fringe out of the matte).
2. Press **Pick**, then click the fringe on the preview.
3. Raise **Tolerance** until the fringe goes. Start low.
4. **Edge band radius** sets how far either side of the matte edge counts as fringe.

The spill matcher scores on **circular hue** (not raw RGB distance) and gates on saturation, so blue and dark purple are not confused.

### Work area controls

- **Set Area** — drag a box to clean only inside it (e.g., blue sky on the right while blue eyes sit on the left).
- **Protect** — drag a box to shield it and clean everywhere else (e.g., blue hair under a blue sky).
- Drawing a box with either button replaces the other.

### Fix colour cast

For strongly tinted light (a violet sky, a green screen) that tints the subject itself:

- Tick **Fix colour cast** and draw a **small** Area box over the backdrop.
- The cast is measured (grey-world against the reference area) and corrected before cleaning.
- Only enable for unwanted light — it will neutralise an intentional backdrop.

---

## 9. Subtitle and logo cleanup

The Clean Up section has two independent targets:

### Remove Subtitles

- Detects a caption band using row energy, temporal residuals, and glyph-shaped components.
- Builds a low-perientile temporal plate when the caption flickers.
- Switches to per-frame inpainting when the caption is always on.
- Tracks moving/resizing caption bands.
- **Never uses the logo detector.**

### Remove Logo

- Detects saturated marks and faint low-opacity white watermarks.
- Uses local contrast and temporal stability.
- Expands the detected region slightly to include anti-aliased edges.
- Repairs the mask with a feathered inpainting.

### Controls

| Control | Effect |
|---|---|
| **Remove Subtitles** | Runs the subtitle pipeline only |
| **Remove Logo** | Runs the logo pipeline only |
| **Remove Both** | Runs both pipelines independently |
| **Automatic detection** | Let the app find overlays |
| **Draw a rectangle** | For faint/moving logos the heuristic cannot classify confidently |
| **Mask to End** | Pre-cleans the whole clip for smooth scrubbing |
| **Regenerate** | Tries another deterministic repair variation |
| **Clear Cleanup** | Removes all active cleanup regions |
| **Check-up** | Shows which regions are armed and which inpainting engine is active |

### Inpainting backends

| Backend | When used |
|---|---|
| **LaMa TorchScript/CUDA** | CUDA plus bundled model — native-resolution repair |
| **LaMa ONNX** | CPU fallback — fixed 512x512 input |
| **Donor rows + Telea** | Always available — reproducible, texture-preserving repair |

---

## 10. Matte QC Inspector

Open with **Inspect → Matte QC Inspector** (`Ctrl+Shift+M`).

### What it shows

- **Source frame** on the left.
- **Current matte + QC overlay** on the right.
- Colour evidence for hair/detail loss, background spill, enclosed holes, and edge flicker.
- Raw-versus-current score values and changed-pixel count.

### Mirrored navigation

- **Link views** is enabled by default — zooming or dragging one pane mirrors the other.
- Turn it off to inspect two areas independently.
- **Reset view** returns both panes to 100%.

### QC evidence types

| Evidence | What it detects |
|---|---|
| **Hair/detail loss** | Fine structures that the matte is missing |
| **Background spill** | Background colour bleeding into the matte edge |
| **Enclosed holes** | Gaps inside the subject that should be filled |
| **Edge flicker** | Frame-to-frame instability at the matte boundary |

The report is shape-safe, uses boolean evidence masks, and never raises for an empty or mismatched mask. A `MatteComparison` reports after-minus-before deltas so "improved" means lower severity.

---

## 11. QC Coach

The QC Coach is a **local, deterministic, offline** analysis tool built into the Matte QC Inspector.

### How it works

1. Frame statistics estimate coverage and edge detail.
2. The QC report prioritises issue-specific tips.
3. Typed complaints are matched against built-in and custom tips.
4. Answers and recent questions persist in the user settings directory.
5. The inspector can emit a settings suggestion for the current shot.

### Using it

1. Select a common complaint or type your own (e.g., "the hair is too smooth").
2. Press **Ask** for a direct explanation and setting recommendation.
3. Press **Save as custom tip** to store that answer locally.
4. Press **Apply suggestion to current shot** to apply the recommended profile values.

The matcher has **explicit polarity** for smooth vs. sharp complaints — they require opposite settings and are never collapsed into one keyword bag.

---

## 12. Performance Monitor

Open with **Inspect → Performance Monitor** (`Ctrl+Shift+G`).

Shows live GPU telemetry when the platform exposes it:

- GPU utilization
- Used and total VRAM
- Temperature and thermal pressure
- Estimated render time
- Model-fit safety with a reserve margin

A high temperature or insufficient free VRAM can downgrade a model from safe to caution/unsafe. The monitor is optional — close it if you are exporting on a laptop and need the GPU for another task.

---

## 13. Export

Press **Ctrl+E**.

### Formats

| Format | Use |
|---|---|
| **PNG Sequence (RGBA)** | Frame-by-frame alpha delivery to another compositor |
| **ProRes 4444 (MOV)** | Production alpha video |
| **QuickTime Animation (MOV)** | Large alpha files with predictable editing |
| **H.264 (MP4)** | Review and chroma-key workflows; no alpha |

### Export options

- **Combined masks** — all objects in one file.
- **One object** — export a single object's masks.
- **Separate files per object** — individual files for each object.

### Range options

- **All Masked Frames** — every frame that has a mask.
- **Timeline Selection** — the shift-dragged range.
- **Manual frame range** — specify start and end.

### What happens at export

1. The exporter decodes every masked frame.
2. **Edge softness and spill** are applied at render time (on copies, not saved masks).
3. Cleanup regions are applied to the source frames first.
4. The result is written to the selected destination.

Because edge finishing is applied at render time, you can keep adjusting controls and re-export without re-running any segmentation. Turning them off restores the original masks exactly.

### Size estimate

The export dialog estimates size before starting. A long clip takes a moment on first run — the window sits on "Preparing export" while it decodes every masked frame.

---

## 14. Batch Render Queue

Open with **Export → Batch Render Queue** (`Ctrl+Shift+Q`).

### Features

- Add the current saved project or browse for other `.bbproj` files.
- Save a named preset from the current export settings.
- Set a final destination.
- Start, pause, resume, cancel, retry, or remove jobs.
- Progress and errors are stored per job.
- A job interrupted by a restart returns to **queued** instead of being stranded in **running**.

### Important notes

- A project must be saved before it can be queued.
- The queue renders the masks already stored in the project; it does not re-run segmentation automatically.
- Each job owns its own project read, video reader, masks, frames, and exporter — no sharing with the GUI's SAM manager or frame buffer.

---

## 15. Version history and recovery

Open with **Inspect → Version History** (`Ctrl+Shift+H`).

### How it works

- **Automatic checkpoints** are created after edits when a project is dirty.
- **Manual checkpoints** can be named.
- Checkpoints are **checksummed and verified** before restore.
- Retention keeps a bounded number of versions.
- The dialog compares the selected mask with the adjacent checkpoint.
- Restore asks for confirmation and keeps the current project path.

### Sidecars

Each checkpoint has a JSON sidecar containing version, label, timestamp, size, and SHA-256. List/load operations verify the checksum and optionally parse the project.

### When to create checkpoints

Use checkpoints before:
- Experimenting with a model change
- Applying aggressive smoothing
- Drawing a large cleanup region
- Any risky operation

---

## 16. Multi-object scenes

BetterBrush supports tracking multiple subjects in the same clip.

### How it works

- Each object has its own mask, keyframes, and profile.
- Objects are stored per-object in the project file.
- The exporter can combine masks, export one object, or write separate files per object.
- The QC Inspector can compare per-object results.

### Visual differentiation

In multi-object mode, each object gets a coloured dot on the timeline and a unique rim glow colour in the canvas, so you can tell them apart at a glance.

---

## 17. Models and hardware

### Model families

| Family | Description |
|---|---|
| **SAM 2.1** | Meta's Segment Anything Model 2 — the default. Tiny through Large tiers. |
| **EfficientTAM** | Efficient Tencent AMG model — faster alternative. |
| **MatAnyone** | Specialised for human/humanoid segmentation. |
| **SAM 3.1** (optional) | Meta's gated `facebook/sam3.1` checkpoint for capable CUDA machines. |

### Model selection

- **Auto (recommended)** — the profiler chooses a model that leaves safety headroom instead of filling all available VRAM.
- You can force a specific tier when you understand the memory cost.
- A per-shot profile can force a model.

### Hardware profiler

The `SAM2Profiler` maps hardware to a conservative `SAM2Recommendation`:

- model tier and Hydra config
- checkpoint URL/path
- device/backend
- mixed-precision flag
- content profile parameters
- IoU and stability thresholds
- in-memory frame-bank size

The safety policy keeps a fraction of free VRAM. Recommendations are never "the largest tier that technically fits."

### Optional SAM 3.1

- Install the `sam3` extra.
- Request access to the gated `facebook/sam3.1` checkpoint.
- Authenticate with `hf auth login`.
- CUDA/VRAM floor is deliberately higher than every SAM 2 tier.
- Never imported at module import time; never the automatic choice on an unknown or unsupported backend.

---

## 18. Project files (.bbproj)

A `.bbproj` is a JSON file containing:

- Top-level video reference and metadata
- Base64-encoded PNG masks (decoded back to `bool` arrays)
- Prompt points and keyframes
- Per-object records
- Scene cuts
- Default processing profile
- Per-object/per-shot profile records

### Safety

- `ProjectManager.save` writes to a same-directory temporary file and uses `os.replace`, preventing readers from observing a partially written project.
- Mask PNGs are deterministic.
- The schema is backward-compatible with existing v1/v2/v3 formats; newer fields are additive.

### Resolution order

Shot profiles resolve in this order:

1. Object/shot override
2. Object default
3. Project default

---

## 19. Editions and pricing

| Capability | Lite | Pro |
|---|---|---|
| Point prompts and brush | Yes | Yes |
| PNG / ProRes / H.264 | Yes | Yes |
| Multi-object scenes | — | Yes |
| Original audio | — | Yes |
| Watermark-free export | — | Yes |
| Higher-resolution export | — | Yes |

- **Free to start** — the Lite edition covers the core workflow.
- **Pro is a one-time payment** (€5) — no subscription.

---

## 20. Fine-tuning: train your own anime SAM 2

BetterBrush ships everything you need to train your own anime-tuned SAM 2 checkpoint on one consumer GPU.

### The loop

```
1. Open a clip in BetterBrush
2. Point / brush / propagate to a matte you're happy with
3. File > Save Training Data...   -> frames + corrected masks on disk
4. Repeat 1-3 over 5-20 clips
5. Train with tools/sam2_training/anime_amv.yaml
6. Drop the checkpoint into checkpoints/custom/ and select it in AI Model
```

### What to export

Only frames that actually carry a mask are exported. **Export the corrections, not the easy wins** — a mask you accepted unchanged teaches the model what it already knows; the frames where you painted over a mistake are where the value is.

### Hardware guide

| GPU | Resolution | Notes |
|---|---|---|
| RTX 3060 12 GB | 512 | Comfortable, ~4-6 it/s |
| RTX 5060 Laptop 8 GB | 384-448 | Fits; keep num_frames 2-4 |
| Colab free T4 16 GB | 384 | Works, slow, needs float16 |
| 24 GB desktop | 768-1024 | Raise resolution, then epochs |

---

## 21. Keyboard shortcuts

### File

| Action | Shortcut |
|---|---|
| New Project | `Ctrl+Shift+N` |
| Open Project | `Ctrl+Shift+O` |
| Save Project | `Ctrl+S` |
| Save Project As | `Ctrl+Shift+S` |
| Open Video | `Ctrl+O` |
| Media Library | `Ctrl+L` |

### Inspect

| Action | Shortcut |
|---|---|
| Matte QC Inspector | `Ctrl+Shift+M` or `Ctrl+M` |
| Performance Monitor | `Ctrl+Shift+G` |
| Shot Profiles | `Ctrl+Shift+P` |
| Version History | `Ctrl+Shift+H` |
| Create Checkpoint | `Ctrl+Alt+S` |

### Export and Edit

| Action | Shortcut |
|---|---|
| Export | `Ctrl+E` |
| Batch Render Queue | `Ctrl+Shift+Q` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` or `Ctrl+Shift+Z` |

### Canvas

| Key | Action |
|---|---|
| `B` | Switch between Point and Brush tools |
| `Space` or `P` | Play/pause |
| `Left` / `Right` | Previous/next frame |
| Mouse wheel | Zoom the canvas |
| `Shift` + brush | Erase instead of add |
| `[` / `]` | Smaller/larger brush |
| `Delete` | Remove the selected point |
| `I` / `O` / `X` | Set timeline in/out points or clear the range |

---

## 22. Preview modes

The **View** menu above the timeline switches the canvas between:

| Mode | What you see |
|---|---|
| **Overlay** | The current mask is tinted over the frame |
| **Cutout** | The subject on a checkerboard — best for seeing edge quality |
| **Raw** | The source frame without the mask layer |

---

## 23. Troubleshooting

| Symptom | Try |
|---|---|
| Mask is too large | Add an exclude point just outside the subject, then re-segment |
| Hair disappears | Lower edge smoothness and raise detail preservation |
| Edge flickers | Enable temporal smoothing, then use a two-frame window |
| Background fringe remains | Use Cutout view, lower smoothness, and add an exclude point |
| Exported edges look jagged | Set edge softness to 2 or higher |
| Spill cleaner seems to do nothing | Check the mode is not Off |
| The cleaner eats part of the subject | Lower Tolerance, or Protect a box over that detail |
| Fix colour cast did nothing | It needs a small Area box over the backdrop, not the whole frame |
| Colour is gone from a region | A Protect box may be active, or an Area box may be too large |
| A faint logo remains | Draw a slightly larger region, then Mask to End |
| Export is slow | Check the Performance Monitor, reduce resolution, or use a smaller model |
| Export window sits on "Preparing export" | It is decoding every masked frame before rendering |
| GPU warning | Close other GPU-heavy apps and let VRAM/thermal headroom recover |
| Project will not open | Keep the .bbproj beside its video, or use Locate Video |
| A long session is risky | Create a named checkpoint before continuing |

---

## 24. Architecture overview

```
QApplication
  └─ MainWindow / QThread workers
      ├─ VideoReader + FrameBuffer
      ├─ SAM2Manager (image + video predictors)
      ├─ Processing pipeline (mask refinement, temporal smoothing, plates, QC)
      ├─ ProjectManager (.bbproj persistence)
      └─ Exporter (FFmpeg/PNG) and RenderQueue
```

### Layer responsibilities

| Layer | Main files | Responsibility |
|---|---|---|
| Bootstrap | `src/main.py`, `src/core/app.py` | Qt setup, splash, dependency warm-up |
| Application state | `src/ui/main_window.py` | Menus, panels, canvas/timeline wiring |
| Domain model | `src/core/scene.py`, `project.py`, `profiles.py` | Objects, masks, keyframes, profiles |
| Model runtime | `src/models/sam2_manager.py` | SAM 2 loading, prompts, propagation |
| Video | `src/processing/video_reader.py`, `frame_buffer.py` | Decode, metadata, bounded caches |
| Image processing | `mask_refine.py`, `temporal_smooth.py`, `plates.py` | Edge cleanup, flicker, inpainting |
| QC | `matte_qc.py`, `qc_coach.py` | Issue evidence, local coaching |
| Export | `src/export/exporter.py`, `render_queue.py` | FFmpeg/PNG output, persistent jobs |
| Hardware | `src/hardware/detector.py`, `profiler.py`, `monitor.py` | Capability detection, telemetry |

### Threading model

- Workers live in `src/processing/inference_worker.py`.
- They never mutate widgets from `run()`.
- Progress and completion are emitted through Qt signals.
- Cancellation uses cooperative `threading.Event`.
- Long-running workers own temporary readers/engines.

### Video caching

- `VideoReader` exposes frame-indexed RGB access with OpenCV metadata.
- `FrameBuffer` is an LRU cache bounded by both frame count and megabytes.
- `CleanFrameStore` is a separate bounded cache for pre-baked cleanup frames.
- No reader is shared between concurrent workers without a lock.

---

## 25. Technical guardrails

These rules keep the codebase safe and maintainable:

- **Never share a reader/manager across workers** without adding a lock.
- **Never mutate widgets from a worker** — emit signals instead.
- **Never change subtitle detection** while "fixing" logos.
- **Never bump the project schema** without a migration/default path.
- **Never write a large frame cache** without a memory bound.
- **Always emit completion on every exit path** — a worker that returns silently leaves the caller waiting forever.
- **Always restore window state before branching on the result** — a handler that returns early on the error path leaves the window permanently disabled.
- **`QWidget.setEnabled(False)` cascades to child widgets** — dialogs that must stay interactive need explicit re-enabling.

---

<div align="center"><span style="color:#555560">End of complete feature reference</span></div>
