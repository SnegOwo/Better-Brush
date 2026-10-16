# Anime rotoscope toolbox - candidate techniques

> status: proposed

This document is a toolbox, not a roadmap of committed work. Each entry is a
technique that could make Better Brush's anime/AMV segmentation better, judged
on four axes: what defect it fixes, what it costs (speed and engineering), which
pieces the repository already has, and the smallest version worth building. It
is grounded in the real modules: every entry names the file it would live in and
the existing helper it reuses. The document starts by stating the gap that
dominates all of it - the desktop app and the headless engine do not run the
same post-processing - then works through cel holds, line priors, alpha snapping,
temporal smoothing, edge decontamination, hair, multi-character identity,
scene-cut re-seeding, interactive tools, and cheaper model alternatives. It ends
with a prioritised shortlist of five things to build next, in order.

Recommendations are "do" (worth building, low risk), "try" (worth a spike,
outcome uncertain), or "skip" (not worth it, with the reason). Effort is in
engineer-days and implementation size is S (< 1 day), M (2-5 days), L (> 1 week).

---

## 1 · The gap that comes first: the app and the engine disagree

This is not a technique, it is the precondition for several of them.

The desktop app runs a post-processing pass that includes cel hold, line lock and
optical-flow temporal smoothing (documented in `docs/ANIME-AMV.md`, applied by
the post-process worker). The After Effects engine does not. Its pipeline
(`src/engine/pipeline.py:run_job`) goes: load model -> extract -> `_analyze` ->
`_propagate` -> `_close_holes` (`mask_refine.fill_mask_holes`) -> optional
`_matte` (MatAnyone 2) -> export with `edge_finish.soft_coverage`.

What is therefore missing from every AE-panel result today, compared with the
app:

| Feature | Module that already exists | Called by the engine? |
| --- | --- | --- |
| Cel hold / matte threading | `src/processing/cel_hold.py` (`apply_cel_hold`, `hold_runs`) | No |
| Line-art snapping | `src/processing/line_prior.py` (`snap_to_anime_lines`) | No |
| Optical-flow temporal smoothing | `src/processing/temporal_smooth.py` (`temporal_smooth_masks`) | No |
| Edge decontamination | `src/processing/edge_finish.py` (`finish_edges`, `decontaminate_edges`) | Yes, since round 7 |
| Scene-cut awareness | `src/processing/scene_detector.py` (`detect_scene_cuts`) | No (the engine samples 3 frames in `_analyze`) |
| Spill cleaning | `src/processing/despill.py` (`despill_frame`, `despill_alpha`) | No |
| Anime QC | `src/processing/anime_qc.py` (`analyze_anime_matte`) | No (`_quality_report` only computes coverage and a mean-abs-diff flicker) |

Round 7 narrowed this gap: the engine now decontaminates edges, and it snaps the
matte to the mask inside flat source regions (`MattingSettings.flat_snap`). Cel
hold, line lock and optical-flow smoothing are still app-only, and they are the
entries below that matter most.

So the highest-value toolbox entry is not a new algorithm at all: it is wiring
the existing anime post-processing into `pipeline.run_job` in the same order the
app uses. That reframes several "new feature" ideas below as "finish an
integration".

Recommendation: **do** this first, as a single entry (it is the substrate for
entries 2, 4, 5 and 9). Effort: 4-6 days. Size: M.

---

## 2 · Cel-hold detection and matte threading

**Fixes:** boiling - the matte contour twitching on frames that are a held
drawing. In an AMV this is the most visible defect there is.

**What we have:** `src/processing/cel_hold.py` is complete and tested
(`tests/test_cel_hold.py`): `frame_signature`, `frames_match`, `find_holds`,
`apply_cel_hold`, `hold_runs`, `held_frame_count`, `_DEFAULT_THRESHOLD = 2.0`.
`apply_cel_hold(masks, frames, threshold)` copies the previous mask verbatim
across a hold, so boiling is zero by construction on those frames. The app
already runs it for the Anime preset.

**The gap:** the engine never calls it, and the engine already has every input
it needs - the extracted frames are in `work/frames/*.jpg` and the masks are in
memory during `run_job`. Wiring it is a five-line insertion after `_propagate`:

```python
from src.processing.cel_hold import apply_cel_hold
frames = {i: _read_rgb(frames_dir / f"{i:05d}.jpg") for i in range(info.count)}
held = apply_cel_hold(masks, frames)
masks = held.masks
```

A real implementation should not re-read every JPEG (memory); sample the frames
or read them lazily, and cap the number held in RAM.

**Smallest worth building:** call `apply_cel_hold` after propagation and before
`_close_holes`, gated on `request.preset == "anime"`. Report
`held.copied_count` in `JobResult.qc` so the panel can say "42 frames threaded
across 11 held cels".

**Bigger version:** feed `hold_runs` into propagation so held frames are skipped
entirely and their masks are copied, saving inference time (the module docstring
promises this). That needs `SAM2Manager.propagate_in_video` to accept skips, or
a wrapper that propagates only the non-held frames.

**Cost/risk/recommendation:** no new model, no speed cost (it saves time), low
risk (the module degrades to unchanged when frames differ). **Do.** Effort:
1-2 days for the wiring, 3-4 more for the skip-propagation optimisation. Size: S,
then M.

---

## 3 · Line-art prior: seed and repair

**Fixes:** edge wobble (the correct anime edge is the pencil line) and
background-line inclusion (the mask grows along a fence or a background
character).

**What we have:** `src/processing/line_prior.py`:
`extract_lines(frame, low, high, dilate)`, `boundary_band`, `boundary_line_support`,
`background_line_penalty(mask, lines, points, reach)`, `snap_to_lines`,
`snap_to_anime_lines(mask, frame, radius, allow_shrink)`. `_SNAP_SUPPORT_FLOOR = 0.15`
means a non-line-art frame is returned unchanged, so it is safe on live action.
It is used during app segmentation (candidate penalty) and post-process (edge
snapping).

**The gap:** the engine uses neither. `auto_subject.find_subject` scores
candidates without `background_line_penalty`, and `pipeline._propagate` never
snaps the contour.

**Smallest worth building (two independent pieces):**

1. **Repair:** after `_close_holes` and before matting/export, snap each tracked
   frame's contour to the line art with `snap_to_anime_lines(mask, frame)`,
   gated on `preset == "anime"`. This is the visible edge win.
2. **Seed:** in `auto_subject.evaluate`, subtract a small term proportional to
   `background_line_penalty(candidate.mask, lines, points)` from the candidate
   score, so a mask that swallowed background line art loses to a clean one.
   This needs the prompt points, which `find_subject` has, and the extracted
   lines for the frame, which it can compute once.

**Cost/risk:** `extract_lines` is a Canny per frame, cheap next to SAM 2;
`snap_to_lines` is morphological, cheap. Risk is moderate: over-snapping can
shrink a soft subject, but `_SNAP_SUPPORT_FLOOR` and `allow_shrink` bound it.
Recommendation: **do** the repair, **try** the seed penalty (measure candidate
rank changes on the test clip before trusting it). Effort: repair 2-3 days, seed
3-4 days. Size: M.

---

## 4 · Flat-region alpha snapping

**Fixes:** anime colour regions are flat, but learned mattes and coverage ramps
leave interiors at 0.6-0.9 alpha, which reads as faint translucency inside a cel
and survives into the composite.

**What we have:** `edge_finish.soft_coverage(mask, feather)` already produces a
clean ramp centred on the contour, and `matting._solid_core` /
`MattingSettings.clamp_solid_interior` already force a mask's deep interior
opaque. There is no explicit "snap anything more than N px inside the contour to
1.0" pass in the engine.

**Smallest worth building:** a function in `src/processing/edge_finish.py`,
`snap_interior(alpha, mask, radius)`, that sets `alpha = 1.0` where
`cv2.erode(mask, radius)` is true and blends a short band back to the ramp. This
is close to `_solid_core`, so reuse its general shape rather than writing a new
one. Apply it in the engine's `_alpha_for` / export path for anime only.

**Why it is not already free:** `_solid_core` exists only inside the matting
clamp, and only when matting runs. Without matting the engine ships a pure
`soft_coverage` ramp and interiors are fine; the defect appears when matting is
on and under-reads a region. So the win is real but narrow.

**Cost/risk:** near-zero cost, low risk (it can only raise interior alpha, and
the mask is the animator's contract). Recommendation: **do**, folded into the
matting post-process rather than as a standalone pass. Effort: 1 day. Size: S.

---

## 5 · Temporal smoothing, in the mask domain and the alpha domain

**Fixes:** residual flicker where cel hold does not apply - real motion where
the contour still jitters, and soft-matte edges that breathe.

**What we have:** `src/processing/temporal_smooth.py:temporal_smooth_masks(frames,
masks, window, on_progress, cancel_event)`. It warps neighbouring masks into the
current frame via Farneback flow, majority-votes only in the edge band, and
protects thin features (`_thin_features`) so hair is not smoothed away. It is
mask-domain: it operates on bool masks.

**The gap:** the engine does not call it. And it has no alpha-domain twin, which
matters because after MatAnyone 2 the useful signal is a float matte, not a
bool mask.

**Smallest worth building:**

1. Run `temporal_smooth_masks` in the engine on the tracked masks, gated on
   anime, after `_close_holes` and before matting.
2. Add an alpha-domain variant: the same warping and weighting but on
   `alphas[index]` in float, with thinning protection replaced by "do not
   smooth where the alpha gradient is steep". This is a new function,
   `temporal_smooth_alpha`, in the same module, deliberately sharing
   `_compute_flow` / `_warp_mask`.

**Cost/risk:** flow is the cost here, roughly one Farneback pair per frame per
neighbour; at 1080p with `window=1` this is seconds per 100 frames, acceptable.
Risk: ghosting if flow is wrong on a cut - but scene cuts are detected (entry 8)
and smoothing must reset at a cut. Recommendation: **do** the mask-domain call;
**try** the alpha domain (it is the honest fix for soft-matte boiling, but it
needs the cut boundary right first). Effort: mask 1 day, alpha 3-4 days. Size:
S / M.

---

## 6 · Edge decontamination and despill

**Fixes:** a hair tip or line-art edge carrying the background's colour at full
strength - the classic anime "halo" - and tinted shots.

**What we have:** `edge_finish.finish_edges(frame, mask, feather)` runs
`soft_coverage` then `decontaminate_edges`, which estimates the background from
the transparent region and un-premultiplies partial pixels. `src/processing/despill.py`
adds `neutralize_cast`, `spill_match`, `despill_frame`, `despill_alpha`,
`apply_shot_transform` for a user-chosen colour, and the app exposes all of it
(Colour Spill Cleaner, cast fix, Area/Protect boxes).

**The gap:** the engine only calls `soft_coverage`. Exports (`_write_rgba_png`,
`_streaming_mov`) never decontaminate, so AE-panel results wear the background
colour on every soft edge even though the app would have cleaned it.

**Smallest worth building:** in `pipeline._rgba` / `render_frame`, when a
matte or feather is in play, compose via `finish_edges(frame, mask, feather)`
instead of `_alpha_for` + a raw RGB copy. Keep the returned decontaminated frame.
For anime specifically, `estimate_background` is reliable because the background
is flat, so this is a strong win.

**Bigger version:** expose the despill controls over the engine wire (a
`spill` object in `JobRequest`) so the panel can offer what the app offers. That
is a contract change, not a quality fix, and it needs panel UI; defer.

**Cost/risk:** a per-frame median over the transparent region plus arithmetic;
negligible next to SAM 2. Risk: under-estimatested background on a shot where
the subject fills the frame (`estimate_background` returns None below 96 px of
outside) - already handled by returning the frame unchanged. Recommendation:
**do**. Effort: 2 days. Size: S.

---

## 7 · Hair-strand handling

**Fixes:** three distinct hair defects - a strand thinner than the mask
resolution disappearing, an enclosed dark-hair region being punched out, and a
soft matte under-reading fine coverage.

**What we have, piece by piece:**

| Defect | Existing helper | Where |
| --- | --- | --- |
| Enclosed hair/webbing holes | `mask_refine.fill_mask_holes(mask, max_hole_area)` | Engine calls it via `_fill_holes` at 1% of the frame |
| Thin structures erased by morphology | `mask_refine._guided_refine` `thin_features` protection; `temporal_smooth._thin_features` | App refinement and smoothing |
| Strand softness under-read | `matting._clamp_alpha_to_mask(solid_interior=True)`, `_solid_core`, `MattingSettings.grow` | Engine matting already uses `clamp_solid_interior=True` |
| Strand edge quality | `line_prior.snap_to_lines` grows onto the line exterior | Entry 3 |

**The gap:** strands are handled by three different modules that never
coordinate. `fill_mask_holes` runs before matting; `snap_to_lines` runs nowhere
in the engine; `grow` is never set by the engine (`MattingSettings` default
`grow=0`).

**Smallest worth building:** a single hair pass in the engine:

1. Snap to line art (entry 3) to recover strand tips on the pencil line.
2. Fill enclosed holes at 1% (already done).
3. In matting, set a small positive `grow` (start 1-2 px) so the matte is
   allowed to reach slightly past the mask where hair really ends, and keep
   `clamp_solid_interior=True`.

**Bigger version:** a dedicated hair-band matte - run MatAnyone 2 ONLY in a
dilated band around the contour and keep the mask authoritative elsewhere. This
is closer to how production mattes are built, but it is a rework of `_propagate`
and its prompt/anchor loop. Try only if the small version is insufficient.

**Cost/risk:** all reuse; risk is a 1-2 px grow eating into background, bounded
by `mask_clamp_pad` (default 6). Recommendation: **do** the small version;
**try** the hair-band matte later. Effort: 2-3 days. Size: S.

---

## 8 · Multi-character splitting and identity across a cut

**Fixes:** two characters in frame, and the same character reappearing after a
cut (the engine currently refuses both).

**What we have:** the app has multiple objects (`main_window._objects`,
`_combined_mask_at`, `test_multi_object.py`). The engine deliberately refuses:
`auto_subject` flags rivals (`MULTI_SCORE_MARGIN`, `MULTI_MIN_AREA_RATIO`,
`MULTI_MAX_IOU`, `_PART_OF_SUBJECT`) and `pipeline._analyze` raises
`ERR_MULTIPLE_SUBJECTS`, handing off to the full app. `SAM2Manager` notes it is
single-supervisor: "every object is tracked as `obj_id=1` inside one state".

**The gap:** a one-button panel that only ever cuts one character is a design
choice, not a bug (see `docs/AFTER-EFFECTS-EXTENSION-PLAN.md` section 14,
decision 4). Two honest options:

1. **Keep one object per job, add identity:** allow the panel to run the same
   character across a cut by re-seeding from the previous scene's mask, and
   report "character lost after the cut" when the mask collapses. This is
   scene re-seeding (entry 9) plus a confidence report, not new tracking.
2. **Add a second object:** extend `JobRequest` with an object list and run
   `SAM2Manager` twice, once per object, each with its own fresh video state
   (the class already warns that states must not be reused). Multiplies runtime
   by the object count and needs panel UI for "which one".

**Cost/risk:** option 1 is small and fits the product stance. Option 2 is a
contract and UX change with a linear speed cost. Recommendation: **do** option
1 (as part of entry 9), **skip** option 2 for now - the accepted hand-off to the
full app is the correct answer for multi-character scenes. Effort: option 1 is
folded into entry 9. Size: S.

---

## 9 · Scene-cut detection and per-scene re-seeding

**Fixes:** the tracker drifting or losing the subject across a hard cut, and a
key frame from scene A being used for scene B.

**What we have:** `src/processing/scene_detector.detect_scene_cuts(video_path,
threshold=27.0, downscale_width=32, min_distance=2, ...)` with a PySceneDetect
path (`_detect_scenedetect`) and a native histogram fallback (`_detect_native`).
The app uses it (`self._scene_cuts`) and already re-seeds matting at the first
frame of every shot (`main_window._run_matting`: `cut_starts`). The engine
samples three frames in `_analyze` and never looks at cuts.

**Smallest worth building:** in `pipeline.run_job`, after `_extract_range`, run
`detect_scene_cuts` on the source (or on the extracted frames), split the range
into scenes, and re-run `_analyze` + `_propagate` per scene, concatenating masks.
A cheaper first step: detect cuts, and if the key frame's scene is not the
longest scene, re-seed on the longest scene instead - this alone prevents the
worst "key frame in the wrong shot" failures.

**Second piece - "keep the character across the cut":** when the mask coverage
drops below a floor for several frames after a cut, try re-seeding from the last
good pre-cut mask (box prompt via `predict_frame_from_mask`, which already
accepts `grow=True`); if it fails, report `ERR_NO_CHARACTER` for that scene
rather than shipping a wrong track.

**Cost/risk:** detection is cheap; per-scene re-seeding multiplies the expensive
`_analyze` step by the scene count (each `_analyze` is ~7 s in the measured
baseline). Risk: an over-sensitive detector on a fast AMV cut creating many small
scenes. Bound it with `min_cut_distance` and a minimum scene length. Recommend:
**try** - the payoff is real but the scene detector must be tuned for AMV
editing, which is exactly the content with the most cuts. Effort: 4-6 days.
Size: M.

---

## 10 · Interactive tools

The panel today (`after-effects/js/panel.js`, `js/host.jsx`, `src/engine/server.py`)
offers: frame scrubber (`GET /frame?dir=&index=`), add/remove brush
(`correction_add` / `correction_remove` base64 PNG masks), `correction_frame`,
`resume_dir` re-tracking, and `bbOpenFullApp`. The full app offers more. Below,
"new wire" means a new field in `JobRequest` (`src/engine/schemas.py`) - that is
a versioned contract, so batch changes.

| Tool | What it fixes | Where it would live | Cost | Recommendation |
| --- | --- | --- | --- | --- |
| Point add/remove prompts | A mask that misses an object a single click could fix; the current brush is heavier | New `prompt_points` on `JobRequest` -> `SAM2Manager.add_points_to_video`; panel canvas already exists | M | **do** |
| Box prompt | "The subject is roughly here" without painting | New `prompt_box` -> `predict_frame_from_mask` already builds a box; panel is a drag gesture | S | **do**, with points |
| Negative prompts | Removing a swallowed background region without a full erase stroke | Same `prompt_points` with `label=0`; `PointPrompt` already carries the label | S | **do**, with points |
| Per-frame keyframe masks + interpolation | Correcting several frames cheaply; today only one correction frame is authoritative | `resume_dir` already stores `masks.npz`; add a `keyframes` list and interpolate before propagate (morphological warp, or `temporal_smooth._warp_mask`) | M | **try** |
| "Propagate from this frame" | Fixing a drift that starts mid-clip without re-running the whole shot | `correction_frame` already exists; expose a button that sets it and re-tracks forward only (`propagate_in_video` from `start_frame_idx`) | S | **do** |
| Onion-skin previous/next mask | Seeing whether a fix is stable | Client-side: fetch `/frame` at `index-1` / `index+1` (or add an `alpha=1` variant) and composite | S | **try** |
| Mask-difference / QC overlay | Showing what changed and where it is wrong | Reuse `matte_qc.render_mask_overlay` / `overlay_mask` server-side; new `GET /overlay` route, or client-side diff of two `/frame` PNGs | M | **do** (QC overlay is the cheapest trust builder) |
| "Auto-fix boiling" button | Threading held cels without a re-run | A `POST /job` mode that reads `masks.npz` via `pipeline._mask_at`, applies `cel_hold.apply_cel_hold`, re-saves via `_save_masks`, re-exports - no SAM 2 involved | S-M | **do** |

Notes on the two smallest, highest-trust items:

- **Auto-fix boiling** is attractive because the masks and the frames are
  already on disk in the result directory (`work/masks.npz`, `work/frames/`
  when the clip is under `_KEEP_FRAMES_LIMIT = 1800`). A new
  `JobRequest.boiling_fix` flag can run entirely in OpenCV with no model, so it
  is fast and safe. It is the "make this shot not boil" button an AMV editor
  actually wants.
- **The QC overlay** should show, per frame, the two anime metrics the model
  cannot see: contour change on a held pair (`anime_qc.contour_change`) and
  background-line inclusion (`anime_qc.background_line_ratio`). `suggest_keyframes`
  already picks the frames a fixer should look at; the overlay makes them
  clickable.

Effort across the table: roughly 3-4 weeks if all of it is built; the
recommended subset (points/box/negative, propagate-from-frame, auto-fix boiling,
QC overlay) is about 2 weeks. Size: M-L.

---

## 11 · Cheaper and alternative models

| Option | What it is | Status in the repo | Verdict |
| --- | --- | --- | --- |
| **EfficientTAM S / Ti** | Lightweight track-anything, ViT encoder + efficient memory | Vendored (`src/vendor/efficienttam`), `ModelSize.EFFICIENTTAM_{S,TI}`, `_ETAM_DISABLE_COMPILE`, HF weights | Already the weak-GPU fallback. **Try** it explicitly on anime to quantify the quality gap; if a fine-tune could close it, it becomes the AMD/CPU story too |
| **SAM 2.1 Tiny / Small / Base+ / Large** | The same family at four sizes | `ModelSize`, `_QUALITY_TO_MODEL`, engine tiers | Ship Base+ for Best; Tiny is a documented draft tier. **Do** nothing new |
| **MatAnyone vs MatAnyone 2** | The two learned video matting networks | Both vendored, `MATTING_MODELS = ("matanyone", "matanyone2")`, `MattingSettings.model` | Engine hard-codes `MatAnyoneRunner("matanyone2")` in `pipeline._matte`. **Try** the switch on anime edges; MatAnyone 1 is known to collapse on short clips (the top-k guard exists for it) |
| **GroundingDINO / open-vocabulary detector** | Boxes a subject from a text prompt ("the girl") | Not in the repo; no dependency | **Skip** for now. It solves subject selection, which `auto_subject` mostly handles, and adds a large model. Revisit only if "no character found" is a common real complaint |
| **Anime face detector** | Anchoring the subject on the face | Already present: `face_parts.detect_landmarks` (mediapipe -> Haar -> skin-centroid), used by `auto_subject.detect_face_center` | Already doing the job. **Do** keep improving the skin fallback, which is what actually fires on anime |
| **Tiny "is this an anime frame" classifier** | Switch defaults (cel hold, line lock, matting settings) automatically | Not present | **Try**. A few-MB ONNX classifier (or even a cheap heuristic: edge density + flat-region fraction) could set `preset=anime` per scene. Low risk, but the preset is one dropdown today, so the win is modest |
| **SAM 3.1** | Meta's successor, multiplexed prompts | `ModelSize.SAM3_1`, `src/models/sam3_adapter.py`, `[sam3]` extra, gated HF weights, CUDA-only | **Try** on anime when a 15 GB CUDA card is available; it is CUDA-only (`load_image_predictor` raises otherwise), so it is not a fallback |
| **Distilled / quantised SAM 2** | A smaller equivalent | Not present (roadmap idea #20) | **Skip** until a fine-tune exists; the fine-tune is the higher-value path |

A concrete cheap experiment worth running before any of the heavier work: run
the same 20-frame anime clip through Tiny, Small, Base+ and EfficientTAM S with
matting on and off, and record coverage, boundary F-score (doc
`SAM2-ANIME-FINETUNE-PLAN.md` section 5.1) and boiling. That table decides
whether "cheaper model" ideas are worth pursuing at all.

---

## 12 · Prioritised shortlist: the five things to build next

In order, each chosen because it is grounded in existing code, has a clear
quality win, and has a measurable acceptance test.

1. **Wire the anime post-processing into the engine** (entry 1). Cel hold, line
   snap, temporal smoothing and `finish_edges` decontamination, in the app's
   order, gated on `preset == "anime"`. This is the single largest quality jump
   for AE-panel users and it is integration, not research.
   *Acceptance:* a fixed anime clip run through `python -m src.engine` shows
   lower `qc.flicker`, non-zero `held` count in `JobResult.qc`, and `anime_qc.line_support`
   above the pre-change baseline. *Effort:* 4-6 days.

2. **Auto-fix-boiling button in the panel** (entry 10). A model-free
   `JobRequest.boiling_fix` mode over the stored `masks.npz`.
   *Acceptance:* pressing it on a held-cel shot threads the runs and
   `qc.flicker` drops; no model load, run completes in seconds. *Effort:* 2-3
   days.

3. **Point / box / negative prompts over the engine wire** (entry 10). Reuses
   `PointPrompt`, `add_points_to_video` and `predict_frame_from_mask`; the panel
   canvas already exists for the brush.
   *Acceptance:* a single click adds a missed hair clump and a single
   right-click removes a swallowed background patch, each in one re-track.
   *Effort:* 5-8 days including panel UI and schema tests.

4. **Line-lock repair in the engine** (entry 3, part 1). `snap_to_anime_lines`
   after hole filling, anime only.
   *Acceptance:* `anime_qc.line_support` rises and background-line ratio does
   not; the contour visual sits on the pencil line on a held frame. *Effort:*
   2-3 days.

5. **QC overlay with suggested keyframes** (entry 10). Show boiling and
   background-line evidence per frame, reusing `anime_qc` and
   `matte_qc.render_mask_overlay`, and make `suggest_keyframes` clickable.
   *Acceptance:* the frames `anime_qc.suggest_keyframes` names are the frames
   the overlay flags, and clicking one jumps the scrubber there. *Effort:* 3-4
   days.

Entries 5, 7, 9 and 11 are worth doing after this list, but they either depend
on it (scene re-seeding depends on the post-processing wiring) or need a
measurement first (alpha-domain smoothing, model alternatives).

---

## 13 · What we are not doing

- **We are not replacing SAM 2 with a purpose-built anime segmenter.** The
  fine-tune plan (`docs/SAM2-ANIME-FINETUNE-PLAN.md`) is the route; a new
  network is a different project.
- **We are not adding open-vocabulary detection (GroundingDINO et al.).** It
  solves a problem the product solves by handing off to the full app.
- **We are not making a half-capable brush UI inside the 340 px CEP panel.**
  The plan from `AFTER-EFFECTS-EXTENSION-PLAN.md` (decision 4) stands: the panel
  does the small fixes, the full app does the rest.
- **We are not changing the mask/matte contract.** The tracked mask stays the
  animator's authoritative input; matting may only soften its edge.
- **We are not applying anime-only post-processing to live action.** Every
  entry gates on `preset == "anime"` or on a self-degrading helper
  (`_SNAP_SUPPORT_FLOOR`, `estimate_background` returning None).
- **We are not rewriting `temporal_smooth` from mask domain to alpha domain in
  one go.** The alpha variant is a new function that shares flow helpers; the
  mask path stays.

---

## 14 · Open questions / how to verify

1. **What order does the app actually use for post-processing?** The engine
   should match it. Read the post-process worker in
   `src/processing/inference_worker.py` and `main_window` and record the exact
   sequence before wiring anything.
2. **How expensive is `detect_scene_cuts` on a 25-minute AMV source?** Measure;
   if it is slow, restrict it to a downscaled pass or a sample.
3. **Does `snap_to_lines` improve or hurt a soft-shaded subject?** It is a
   no-op below `_SNAP_SUPPORT_FLOOR = 0.15`, but a partially line-art shot could
   still be affected. A/B on the fixed clip set.
4. **Does the alpha-domain smoothing preserve real motion?** The mask version
   protects thin features; the alpha version has no such guard yet. Validate on
   a fast-pan clip.
5. **Is `MatAnyoneRunner("matanyone2")` hard-coded in the engine?** Yes, in
   `pipeline._matte`; adding a `Request.model`-style choice is a small contract
   change. Confirm before assuming the panel can pick.
6. **What does the AE panel's `quality` segment actually send?** It sends
   `tiny | fast | balanced | best`; any toolbox that changes per-scene behaviour
   must not silently contradict the tier the panel names.
7. **How much do these post-processing passes slow a job?** The measured engine
   baseline (`AFTER-EFFECTS-EXTENSION-PLAN.md` section 18.2) is ~54 s for 120
   frames; re-measure after the wiring so "better" does not mean "much slower".
8. **Does `masks.npz` survive long enough for the auto-fix button?** The engine
   drops extracted frames above 1800 but keeps the packed masks; confirm the
   result directory is not cleaned up while AE still links to it.
