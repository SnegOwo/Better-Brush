# Better Brush — Audit & Bug-Fix Report

**Scope:** the rotoscoping / clean-up / render pipeline (`src/`), with a focus
on the reported intermittent *"sometimes it just cannot detect the character"*
failure and on anime / AMV quality.

**Method:** full read of the processing, hardware, model and export modules;
reproduction of the model-selection flakiness on the target machine
(RTX 5060 Laptop 8 GB, 8151 MiB VRAM, compute capability 12.0); the full test
suite (`python -m pytest tests/ -q`) as the regression gate. Every fix below is
covered by a test or by an explicitly added regression test.

**Result:** root cause of the intermittent failure found and fixed; 17 defects
addressed across 9 modules; test suite green (**359 passing**, up from 350).

---

## 1 · The headline bug: intermittent "cannot detect the character"

### Root cause — model tier chosen from *live free* VRAM

`src/hardware/profiler.py::_pick_model` picked the SAM 2 tier from
`gpu.available_vram_mb` (free VRAM **at the moment the recommendation was
built**) multiplied by a safety factor:

```python
effective = int(gpu.available_vram_mb * self._VRAM_SAFETY_FACTOR)
```

Free VRAM is not a property of the machine — it moves continuously (other
windows, the browser, the loaded model itself, driver caches). On the 5060
laptop it was observed swinging between **3881 MB and 4788 MB** within minutes.
Because the EfficientTAM family has a much lower threshold than SAM 2.1 Tiny:

| tier | threshold (MB) |
|---|---|
| `EFFICIENTTAM_TI` | 1 100 |
| `EFFICIENTTAM_S` | 1 800 |
| `SAM2_1 TINY` | 2 800 |

a swing of a few hundred MB of free VRAM silently flipped the app between
**SAM 2.1 Tiny** and the far weaker **EfficientTAM** models. EfficientTAM is a
plain-ViT "track anything" model; on anime line art it misses thin strokes and
hair — exactly the reported symptom, and it happened only *sometimes* because it
depended on the machine's instantaneous free memory.

### Fix — deterministic tier from **total** VRAM

`_pick_model` now chooses from the card's **total** VRAM, which is a stable
hardware property:

* `effective = int(gpu.total_vram_mb * _VRAM_SAFETY_FACTOR)` (same 0.65
  headroom — the conservatism is unchanged).
* SAM 2 is only considered when its package is importable **and** the card has
  at least `_SAM2_MIN_TOTAL_MB` (3 500 MB) of *total* VRAM.
* EfficientTAM is now strictly a fallback for GPUs that genuinely cannot host
  SAM 2.1 Tiny (or when SAM 2 is not installed) — never a silent downgrade a
  slightly busy GPU can trigger.

Two new fields on `SAM2Recommendation`, `offload_state_to_cpu` /
`offload_video_to_cpu`, are set for cards under `_OFFLOAD_BELOW_MB` (10 GB) and
consumed by the model manager (§2).

**Regression guard:** `tests/test_profiler_policy.py::test_model_is_deterministic_regardless_of_free_vram`
asserts that eight different free-VRAM values on the same 12 GB card all map to
the **same** tier. The old policy tests were rewritten to the total-VRAM policy.

**Before → after on the user's machine (8151 MB total):** a stable **SAM 2.1
Small** (or Tiny) selection at 448–512 px; no more EfficientTAM flip-flop.

---

## 2 · Model loading & precision (`src/models/sam2_manager.py`)

| # | Defect | Fix |
|---|---|---|
| 2.1 | Autocast context used a stale/wrong precision flag | `_speed_context` now keys off `rec.mixed_precision`, which the profiler only sets on compute capability ≥ 8 (bfloat16). On older cards nothing is wrapped in a half-precision context. |
| 2.2 | Video predictor built without the profiling offload flags | Added `_video_offload_kwargs()` + `_build_video_predictor()`; propagates `offload_state_to_cpu` / `offload_video_to_cpu` from the recommendation. |
| 2.3 | EfficientTAM build could re-enable torch.compile and crash / hang | `_build_video_predictor` preserves `hydra_overrides_extra=_ETAM_DISABLE_COMPILE` for EfficientTAM and degrades gracefully on a `TypeError` (older signature) instead of raising. |

## 3 · Out-of-memory handling (`src/processing/inference_worker.py`)

`InferenceWorker.run` and `PropagationWorker.run` previously surfaced a raw
`CUDA out of memory` traceback. They now catch CUDA OOM and raise a clear,
actionable message (lower resolution / window size, or pick a smaller model),
while letting genuine errors propagate unchanged.

---

## 4 · Segmentation & mask quality

### 4.1 Mask selection (`src/processing/mask_selection.py`)
`_positive_spread_penalty` scored a candidate's "spread" without a floor, so a
degenerate or tiny envelope could dominate the ranking. The envelope span is now
floored at `0.2 × min(h, w)`, so the penalty reflects real spatial spread.

### 4.2 Temporal smoothing (`src/processing/temporal_smooth.py`)
* **Flow direction** was computed against the wrong frame; the warp now always
  uses `_compute_flow(cur_gray, neighbor_gray)` with the matching
  `_warp_mask`, so the optical-flow prior pushes the mask the correct way.
* **Cancellation dropped frames:** the result accumulator was rebuilt instead of
  seeded from `dict(masks)`, so cancelling mid-smoothing discarded every
  already-computed frame. It now starts as `dict(masks)` and only overwrites the
  frames it finishes.

---

## 5 · Export pipeline

| # | Defect | File | Fix |
|---|---|---|---|
| 5.1 | Video frames written with **absolute project index** (`frm_000042.png`). ffmpeg's image2 demuxer starts at 0 and stops at the first gap, so any export whose first masked frame wasn't 0 — or that had a hole — produced a truncated/empty video. | `src/export/exporter.py` | Frames are written with **sequential** names `frm_{i:06d}.png` keyed off the enumerate index. |
| 5.2 | **`ValueError: current cannot exceed total`** in "separate" object mode. Each object contributes its own mask set, so `sum(len(masks))` exceeds the distinct-frame total fed to `RenderProgress`. | `src/export/render_queue.py` | "Loading masks" now reports the true mask count against itself. |
| 5.3 | **`retry()` moved a job to the *end*** of the queue (dict pop + re-add) despite the docstring promising the front. | `src/export/render_queue.py` | The job is re-inserted at the front (`{job.id: job, **remaining}`). |
| 5.4 | **FFmpeg process leak on timeout.** `communicate(timeout=3600)` raised but left the child running (burning CPU/GPU, locking the output file). | `src/export/exporter.py` | Both the encode and the audio-mux paths now `kill()` + reap the process on `TimeoutExpired`; the (optional) audio mux falls through to a silent export as documented. |

Regression tests added for 5.2 and 5.3 in `tests/test_render_queue.py`.

---

## 6 · Clean-up / inpainting (`src/processing/plates.py`)

| # | Defect | Fix |
|---|---|---|
| 6.1 | **Tracked caption chased sideways on vertical drift.** The density-weighted centroid computed the row distribution into `cx` and the column distribution into `cy` (`dens.sum(axis=1)` is the *y* distribution). | The row/column weights are assigned to the correct axes (`cx` from columns, `cy` from rows). |
| 6.2 | **`_best_band` returned an un-clamped band.** Callers index the score arrays with `band[1] : band[1] + band[3]`; a band hugging the top edge produced a **negative** start index, so NumPy wrapped to the end of the array and scored the wrong rows — promoting the wrong band. | `_best_band` now clamps its result to the frame before returning (mirrors `_clamp`). |
| 6.3 | **Static/logo mask bottom extension mixed coordinate origins.** It used the fixed `plate.region` y with the *tracked* crop offset, so once a tracked caption drifted the extension sat above the glyphs and the outline below them was never covered. | Uses the effective cropped region (`ry`/`rh`) with the crop offset. |
| 6.4 | **`clean_video_frames` silently processed nothing** when `CAP_PROP_FRAME_COUNT` came back 0 (some containers/streams): `total = 0` → `end = 0` → early `return 0`. | A reported 0 is treated as "unknown" and the bake streams to EOF (`None` = no upper bound). |
| 6.5 | **One transient LaMa error disabled AI inpainting for the whole session** (`_LAMA_SLOW = True` in the `except`). | A consecutive-**error streak** (`_LAMA_FAIL_RUN` / `_LAMA_FAIL_STREAK = 3`) is required before retiring the engine; a single hiccup falls back for that frame only. |

---

## 7 · Diffuse/removal & despill

### 7.1 Removal composite (`src/processing/removal.py`)
Three defects in `_composite`:

* **Boolean matte scaled to 0/1.** `mask.astype(np.uint8)` yields 0/1, so the
  blend alpha was ~0.004 (1/255) and the composite did almost nothing whenever
  no feather pass re-expanded the range. Now scaled to 0/255.
* **`mask_grow == 0 and feather == 0` returned the *whole* reconstructed
  frame**, ignoring the mask entirely and wiping everything outside the subject.
  Short-circuit removed; the binary matte now gates the composite.
* **`cv2.normalize` collapsed a constant matte to all-zero** (a matte covering
  the whole region silently disabled the removal). Normalisation is skipped when
  `grown.max() == grown.min()`.

Covered by `tests/test_removal_composite.py`.

### 7.2 Despill (`src/processing/despill.py`)
`_sample_area` returned an **empty `(0, 0)` area** when a reference *region* was
given but no matte was present (`shape is None` fell through). The cast was then
never corrected even though the user had drawn an Area box. A region is now
sampled directly from the frame shape, independent of the matte.

### 7.3 Matting (`src/processing/matting.py`)
A failed segmentation-mask lookup silently degraded the clamp to the matte's own
0.5 threshold (letting background into the matte) and logged at `debug`. It now
logs the **first** failure at `warning` (once per run) so a genuinely broken
getter is visible, while still never killing the run.

---

## 8 · Verified-but-unchanged (deliberately)

* **`_VRAM_THRESHOLDS` and `_VRAM_SAFETY_FACTOR`** — the conservative headroom
  that prevents overheating/black-screens is correct; only the *capacity input*
  changed (free → total).
* **`RoPEAttention.feat_sizes`** — lowering the training resolution does **not**
  require editing `feat_sizes`; SAM 2 recomputes the rotary embeddings when the
  token count differs (`if self.freqs_cis.shape[0] != q.shape[-2]`). Square
  inputs (the recipe's `square: true`) are the real constraint.
* **`choose_prompt_mask_index`** ranking, despill hue model, and the
  EfficientTAM fallback thresholds were reviewed and left as-is.

---

## 9 · Anime / AMV tuning summary

* Default anime preset: `pred_iou_thresh 0.82`, `num_mask_tokens 3`,
  `max_frames_in_memory 64` (long clips, crisp lines).
* Stable model selection removes the random EfficientTAM downgrade that hurt
  thin line art.
* Fixed temporal-smoothing flow direction and the tracker centroid improve mask
  stability across pans and vertical caption drift.
* For a permanent quality jump on your own footage, fine-tune — see
  [`FINE-TUNING.md`](FINE-TUNING.md) and the new
  [`tools/sam2_training/colab/`](../tools/sam2_training/colab/README.md) Colab
  kit.

---

## 10 · Verification

```
$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/ -q
# 359 passed, 1 warning (at the time of this audit; the suite now has 524 tests)
```

New / updated regression tests:

* `tests/test_profiler_policy.py` — total-VRAM policy, determinism, offload flags.
* `tests/test_render_queue.py` — separate-mode progress; retry-to-front.
* `tests/test_removal_composite.py` — bool-matte scaling, no-grow/no-feather,
  constant-matte normalise guard, empty matte.
* `tests/test_despill.py` — region without a matte.
