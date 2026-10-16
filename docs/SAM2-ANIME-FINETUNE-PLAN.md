# SAM 2.1 anime/AMV fine-tuning - plan

> status: proposed

This document plans how to make Better Brush's segmentation actually good on
anime and AMV footage by fine-tuning SAM 2.1, and how to route the result into
the app through the mechanism the app already has rather than inventing a
parallel one. That mechanism is concrete today: a fine-tuned checkpoint lives in
`checkpoints/custom/` behind a `sam2_custom.json` sidecar, is addressable as
`ModelSize.CUSTOM`, is resolved by `CheckpointManager.custom_model()`, and is
preferred by the engine's `best` quality tier. The plan covers what fails on
anime and why, where training data legally comes from, how to auto-label it with
SAM 2.1 Large plus the app's own brush, the actual training recipe (what is
frozen, LoRA versus full, the loss, the VRAM budget for an 8 GB laptop card),
the metrics that matter for rotoscoping, and a three-stage path from "a
200-frame eval set and a baseline number" to a shipped custom checkpoint. The
existing tools are not thrown away: `tools/sam2_training/anime_amv.yaml`,
`tools/sam2_training/colab/sam2_colab_finetune.py`,
`tools/sam2_training/install_lugnica.py` and `src/processing/training_data.py`
are the rails, and the plan extends them.

Effort is in engineer-days. Where a number is guessed it says so.

---

## 1 · The rails that already exist

Read this before proposing anything, because the app is further along than the
plan might assume.

| Piece | Where | What it does |
| --- | --- | --- |
| Data exporter | `src/processing/training_data.py` `export_clip` | Writes DAVIS-style `JPEGImages/<clip>/<frame>.jpg`, `Annotations/<clip>/<obj>/<frame>.png`, `ImageSets/<split>.txt`; only frames with a mask; appends to the split file |
| Export UI | File > Save Training Data | Calls the exporter over the loaded clip's masks |
| Training recipe (local) | `tools/sam2_training/anime_amv.yaml` | Base+ config, `freeze_image_encoder: True`, resolution 512, `num_frames: 4`, batch 1, lr 2e-5, 20 epochs, `MultiStepMultiMasksAndIous` loss |
| Training driver (Colab/any GPU) | `tools/sam2_training/colab/sam2_colab_finetune.py` | `_apply_base()` rewrites the Hiera shape per `--base`, `checkpoint_url()`, `write_config()`, auto AMP dtype (`float16` on T4, `bfloat16` otherwise) |
| Colab recipe | `tools/sam2_training/colab/anime_amv_colab.yaml` | 384 px, `num_frames: 2`, float16 |
| Installer | `tools/sam2_training/install_lugnica.py` | `install(src, base, label, name, dest_dir)`, `DEFAULT_NAME`, `SIDECAR_NAME`, `CONFIG_FOR_BASE` |
| Sidecar reader | `src/models/checkpoint_manager.py` | `custom_dir()` = `checkpoints/custom/`, `custom_model()` -> `CustomModel(checkpoint, config, label)`, `_CUSTOM_SIDECAR = "sam2_custom.json"` |
| Custom tier | `src/hardware/profiler.py` | `ModelSize.CUSTOM` (threshold 8500 MB), `_refs_for(CUSTOM)` resolves the sidecar |
| Hydra mapping | `src/models/sam2_manager.py` | `_HYDRA_CONFIGS[CUSTOM] = "custom"`; `_config_name(CUSTOM)` matches the sidecar's config **basename** against `sam2.1_hiera_{t,s,b+,l}.yaml` and raises otherwise |
| Engine selection | `src/engine/model_setup.py` | `_QUALITY_TO_MODEL["best"] = "base_plus"`, but `resolve_model_name("best")` returns `"custom"` when `custom_model_available()` |
| Capability report | `src/engine/model_setup.describe()` | Returns `custom_model`, `custom_label`, `quality_models`, `matting`, `device`, `backend` |
| Panel copy | `after-effects/README.md`, `js/panel.js` | "Best" means "your fine-tuned checkpoint if one is installed, Base+ otherwise"; the panel names the resolved checkpoint |
| Tests | `tests/test_lugnica_kit.py` | Asserts `_apply_base` rewrites the trunk shape per base and that every `CONFIG_FOR_BASE` value appears as `configs/sam2.1/<config>` in `sam2_manager.py` |
| Docs | `docs/FINE-TUNING.md`, `docs/LUGNICA-SAM.md` | The current end-to-end walkthrough, per-GPU settings, legal note |

Three constraints fall out of the rails and shape everything below:

1. **The sidecar's `config` must be one of the four stock SAM 2.1 Hiera configs.**
   `sam2_manager._config_name` matches by basename and raises
   `"Unknown base config ..."` otherwise. A custom checkpoint is therefore a
   *same-architecture* set of weights, not a new network.
2. **The training data format is positive-only.** `training_data.export_clip`
   writes one PNG per object; `PNGRawDataset(is_palette=False)` reads it as
   positive ground truth. SAM 2's sampler (`prob_to_sample_from_gt_for_train`,
   `num_correction_pt_per_frame`) creates the negative/correction signal at
   train time. There is no "hard negative mask" channel, and adding one means
   changing the SAM 2 trainer, not Better Brush.
3. **The data engine is the product.** The value of the fine-tune is the
   corrections a human made in the app; a dataset downloaded from a research
   repository is both legally and practically off the table (see
   `docs/FINE-TUNING.md` section "Legal note").

---

## 2 · What actually fails on anime today

Each row is a specific, reproducible failure, tied to the code that shows it and
to the training-data recipe that addresses it. "Recipe" is what a labelled frame
must contain for the model to learn the fix.

| Failure | Why it happens | Evidence in the repo | Training-data recipe |
| --- | --- | --- | --- |
| **Edge floats off the pencil line** | SAM 2 treats a line as an ordinary low-contrast edge; the matte lands a pixel inside or outside it | `line_prior.boundary_line_support`, `anime_qc.line_support` | Frames whose subject boundary is a crisp black line, corrected to the line. Include thick and thin line styles |
| **Background line art swallowed** | Character and background share the same black strokes (fences, desks, background characters); the candidate that grows along them scores well | `line_prior.background_line_penalty`, `mask_selection.trim_prompt_background` | Frames with busy line-art backgrounds where the human has NOT selected the background. This is the "hard negative" content, delivered as ordinary frames, not as a negative channel |
| **Hair with dark interiors** | Dark hair and shadow are locally indistinguishable from a dark background; the mask punches enclosed holes | `mask_refine.fill_mask_holes` is a post-fix; the model still produces the holes | Close-ups with multi-tone hair, corrected so the enclosed dark region is inside the mask. Do not rely on hole filling to teach |
| **Thin props** (sword blade, ribbon, antenna, glasses bridge) | A 1-2 px structure is a coin flip for a mask decoder at 1024 px encoding | `mask_refine` `thin_features` preserve logic exists precisely for this | Frames with a thin rigid prop; the GT keeps it connected and unbroken |
| **Hard edges, no motion blur** | Most matting training data (and MatAnyone) is live-action with a real coverage ramp; anime edges are 0/1 | `edge_finish.soft_coverage` exists because the binary edge reads as a staircase | GT stays binary. Do NOT train soft alpha into SAM 2; softness is a post-process |
| **Semi-transparent FX** (glow, smoke, glass, water) | The model cannot decide whether a translucent region is subject or background, so it flakes frame to frame | `matte_qc` spill/partial-coverage evidence | Frames with FX near the subject; label only the character, treat the FX as background. Consistency matters more than coverage |
| **Painted-on highlights / cel shading** | Flat interior shading is sometimes read as a separate object and split off | `auto_subject._plausible_whole`, `_prefer_whole` exist to reunite parts | Frames where the highlight is enclosed; GT has it inside the mask |
| **Subject shares a colour with the background** | A blue-haired character against a blue sky defeats colour-based cues | `CHANGELOG.md` "Color Spill Cleaner" and "Area and Protect boxes" were added for exactly this | Frames of that shot, labelled carefully. This is a per-shot problem the model learns slowly; expect to need many examples or accept per-shot manual work |

The post-processing stack (`line_prior`, `cel_hold`, `mask_refine`, `edge_finish`,
`despill`, `temporal_smooth`) fixes several of these *after* the model has
spoken, and Doc 3 (`docs/ANIME-ROTOSCOPE-TOOLBOX.md`) plans the rest. The
fine-tune's job is the part no post-process can recover: which pixels belong to
the character at all.

---

## 3 · Data

### 3.1 Where the frames come from (legally)

The existing legal position in `docs/FINE-TUNING.md` is correct and this plan
keeps it: build the dataset from the user's own corrections on footage they are
allowed to use. Concretely, acceptable sources:

1. **The user's own clips**, which for an AMV editor means the sources they
   already edit with. The dataset is never redistributed; it is local training
   input.
2. **Public-domain animation.** In Japan, films whose copyright has expired are
   usable, for example the 1910s-1930s shorts (Namakura Gatana, 1917; Ugokie
   Kori no Tatehiki, 1931) and pre-1953 works under the old Japanese term
   (verify the current Japanese copyright term before relying on this; it
   changed in 2018 and again with the TPP-related extension). These are thin on
   modern line-art style, so they help the model less than the variety implies.
3. **CC-licensed animation** where the licence permits derivative works, from
   archives such as the Internet Archive's moving-image collections or CC-film
   lists. Check per-title; "CC" is not one licence.

Do **not** train on third-party anime instance-segmentation datasets. They are
typically research-licensed and built from copyrighted artwork, and shipping a
model derived from them is exactly the risk `docs/FINE-TUNING.md` warns about.

An honest note: for a modern anime style, the best data is the user's own
corrected frames. Public-domain sources are a legal baseline, not a quality
win.

### 3.2 How to auto-label: SAM 2.1 Large plus the brush

The labelling pipeline reuses the app and the engine; there is no new labeler to
write.

1. **Propose.** Run SAM 2.1 Large (`ModelSize.LARGE`, hydra
   `configs/sam2.1/sam2.1_hiera_l.yaml`, checkpoint `sam2.1_hiera_large.pt`)
   through `SAM2Manager` on each candidate clip. Large is worth the slowness at
   label time; the shipped model can be smaller.
2. **Track to pseudo-labels.** Use `init_video` + `add_points_to_video` (or
   `add_mask_to_video`) + `propagate_in_video`. The engine already does this in
   `pipeline._propagate`, and the desktop app does the same through
   `inference_worker`. An engine job with `output_dir` set keeps `seed.png`,
   `seed.json` and `masks.npz` beside the result, which is a ready-made
   pseudo-label store for the whole clip.
3. **Review and correct.** The whole premise of the product is that the human
   fixes what is wrong. Export only the corrected clips with
   `training_data.export_clip`. The `docs/FINE-TUNING.md` rule stands: export
   the frames you *fixed*, not the ones the model got right.
4. **Hard negatives as content, not as a channel.** When the model swallowed a
   background region, the corrected frame (with that region removed from the
   mask) is the negative signal. Optionally record *why* a clip was corrected
   (from `anime_qc.analyze_anime_matte` messages, e.g. "background line art
   included") in a sidecar so the dataset can be stratified by failure type.
5. **Quality gate.** Reject a pseudo-label frame where the propagated mask is
   empty, covers > 80% of the frame, or touches the border on more than three
   sides (`auto_subject._BACKGROUND_EDGE` and `_area_pref` already encode these
   bounds; reuse them rather than re-deriving). A dataset built from bad
   pseudo-labels teaches the model to reproduce them.

### 3.3 A mask dataset from matte data

The app can produce soft mattes (`src/processing/matting.py`, MatAnyone 2) and a
coverage ramp (`edge_finish.soft_coverage`) that are strictly better than a
binary mask at the edge. SAM 2's trainer as configured consumes *binary* GT
(`PNGRawDataset(is_palette=False)`, `loss_mask` BCE plus `loss_dice`). So:

- **Do not feed soft alpha as GT.** It will be read as a binary mask and
  thresholded implicitly by the loader, and the coverage information is lost
  anyway.
- **Do feed the *binarised* matte as GT** when the matte is a cleaner silhouette
  than the tracked mask. Rule of thumb: threshold the MatAnyone alpha at 0.5,
  then run `mask_refine.fill_mask_holes` on it, and use that as the label. This
  is the one useful thing matte data does for the segmentation dataset.
- **Keep the matte itself** only if we later add a soft-GT loss (stage 3 idea
  below). It is not a stage 1-2 requirement.

Also useful: derive a *held-frame* duplicate set from the matte data. If two
consecutive frames are the same cel (`cel_hold.frames_match`), the mask must be
identical. A training sample that pairs a frame with a hold of itself, labelled
with the same mask, directly teaches temporal stability. See section 4.4.

---

## 4 · The recipe

The shipped recipe is `tools/sam2_training/anime_amv.yaml` plus its Colab twin
`tools/sam2_training/colab/anime_amv_colab.yaml`, generated by
`sam2_colab_finetune.py`. This section states what to keep, what to change, and
what is genuinely new work.

### 4.1 Checkpoint and freeze schedule

- **Base:** `base_plus` by default, matching `docs/LUGNICA-SAM.md` and the
  `CONFIG_FOR_BASE` map in `install_lugnica.py`. `small` for a lighter shipped
  file, `tiny` for pipeline testing only.
- **Frozen:** the Hiera image encoder, via `freeze_image_encoder: True`
  (line 204 of the yaml). This is the single biggest memory saver and the part
  anime quality depends on least.
- **Trained:** mask decoder, memory attention, memory encoder, prompt encoder,
  object pointers. That is where edge quality lives.
- **Consequence for the sidecar:** the trained checkpoint must load into the
  stock architecture, because `sam2_manager._config_name(CUSTOM)` resolves a
  stock Hydra config and `build_sam2` builds the plain model. Any recipe that
  changes module shapes (extra heads, changed `num_maskmem`) breaks the app
  path. If a shape change is ever needed, the sidecar needs a new field and
  `_config_name` a new mapping; treat that as a separate, larger change.

### 4.2 LoRA versus full fine-tune

| | Full decoder fine-tune (current) | LoRA on the decoder |
| --- | --- | --- |
| Mechanism | Train the decoder/attention weights directly | Freeze base weights, train low-rank adapters |
| Fits the app today | Yes: output is a same-shape `.pt` | Only if adapters are **merged** into the base weights before install |
| Memory | Higher (optimizer state over decoder params) | Lower, but the frozen base still sits in VRAM |
| Quality at our data scale (300-1000 frames) | Good; what `LUGNICA-SAM.md` recommends | Comparable or slightly behind; easier to overfit small sets |
| Risk | More VRAM, slower per step | `peft` is not a dependency; merge step must be correct |

**Recommendation:** keep the full decoder fine-tune as the default and as the
thing that ships. Treat LoRA as a stage-2 *experiment* whose only acceptable
deliverable is a merged checkpoint that is byte-for-byte loadable by the existing
`build_sam2` path. Do not add `peft` to the app's dependencies; it belongs to the
training environment (`tools/sam2_training/`), never to `src/`.

### 4.3 Loss

The shipped loss is `training.loss_fns.MultiStepMultiMasksAndIous` with
`weight_dict: {loss_mask: 20, loss_dice: 1, loss_iou: 1, loss_class: 1}`
(BCE-style mask loss plus dice plus IoU prediction). For anime rotoscoping, the
quality metric that matters is the boundary, and the shipped loss weights the
interior as heavily as the edge.

Plan: add a **boundary-aware** variant in the *training* repo we clone, not in
Better Brush:

- A signed-distance-weighted BCE: weight = 1 + k * exp(-d / sigma) where d is the
  distance to the GT contour. This is a small edit to a custom loss class in the
  cloned `sam2/training/loss_fns.py` or a subclass in
  `tools/sam2_training/`.
- Keep `loss_mask` and `loss_dice`; add the boundary term with a small weight
  (start at 0.5-1.0) so it refines the edge without destabilising training.
- Focal is already effectively covered by the sampler's hard-example mining
  (`num_correction_pt_per_frame: 7`, `prob_to_sample_from_gt_for_train: 0.1`);
  do not add a second focal term until the boundary term shows it is needed.

This is genuinely new work and its effect is uncertain; stage 3 tests it against
the stage-1 baseline.

### 4.4 Resolution and augmentation

The yaml already has: `RandomHorizontalFlip`, `RandomAffine(degrees=10, shear=5)`,
`RandomResizeAPI(sizes=512, square=true)`, `ColorJitter(brightness=0.1,
contrast=0.03, saturation=0.03, hue=null)`. Add, in order of value:

1. **Cel-hold duplication.** For each clip, detect holds with
   `cel_hold.hold_runs` and, with probability p (start 0.3), replace a frame's
   sample with another frame from the same run and the identical mask. This
   teaches that a held drawing must produce the same matte, which is the single
   most visible anime defect. Cheap to implement in the dataset wrapper.
2. **Flat-colour jitter.** Anime interiors are flat; a small per-channel gain
   applied to flat regions (or a stronger `ColorJitter.saturation` than 0.03)
   improves robustness to remasters and encodes. Keep `hue: null` off the table
   only after testing; a little hue jitter is safe for flat art.
3. **Line-art dropout / thinning.** Randomly drop or thin the extracted line art
   (`line_prior.extract_lines` gives the mask) so the model does not become
   dependent on a line always being present. Use sparingly.
4. **Never** add motion blur or film grain augmentation to anime. It teaches the
   wrong edge model and fights the "hard edges" reality.
5. **Keep `square: true`.** Changing the input aspect requires touching SAM 2's
   `feat_sizes`/RoPE bookkeeping; `docs/FINE-TUNING.md` notes RoPE recomputes,
   but square inputs are the safe, tested path.

### 4.5 VRAM budget on an 8 GB laptop card (RTX 5060 Laptop)

Starting point from `docs/FINE-TUNING.md` and the yaml, which are already tuned
for this card:

| Setting | 8 GB value | Why |
| --- | --- | --- |
| `scratch.resolution` | 384 (448 only if it fits) | Peak activation scales with tokens squared |
| `scratch.num_frames` | 2 | The temporal memory is the second-biggest consumer |
| `scratch.train_batch_size` | 1 | Never raise on 8 GB |
| `scratch.max_num_objects` | 2 | Enough for two-character corrections |
| `freeze_image_encoder` | `True` | Mandatory |
| `optim.amp.amp_dtype` | `bfloat16` | 5060 (Ada/Blackwell) has bf16; the Colab driver already picks this automatically |
| `use_act_ckpt_iterative_pt_sampling` | `True` only if it OOMs | Gradient checkpointing trades time for memory |
| `distributed.backend` | `gloo` | Windows PyTorch ships no NCCL (already in the yaml) |

Offload that is *not* available at train time: the video predictor's
`offload_state_to_cpu` belongs to `SAM2Manager`, not the trainer. If the trainer
OOMs, the levers are resolution, then frames, then activation checkpointing, in
that order.

### 4.6 A hard constraint to state up front

The custom checkpoint is loaded by `build_sam2` with `device=<resolved device>`.
On an AMD/CPU-only machine that is slower or unsupported (see
`docs/AMD-WINDOWS-SUPPORT-PLAN.md`). A fine-tune improves quality, not
throughput; on CPU it makes every run *slower* because the model is the same
size and CPU is the bottleneck. Say so in the release notes.

---

## 5 · Evaluation

### 5.1 Metrics that matter for rotoscoping

A generic IoU is necessary but not sufficient. Score, per frame and aggregated:

| Metric | What it catches | Where it comes from |
| --- | --- | --- |
| Mask IoU | Overall correctness | New, trivial to compute |
| Boundary F-score (F-measure within a 2-3 px band of the GT contour) | The edge, which is the whole point | New; reuse `line_prior.boundary_band` for the band |
| Boundary IoU (Cheng et al.) | Edge quality without interior bias | New, small |
| Temporal boiling | Contour twitch on held cels | `anime_qc.boiling_score`, `contour_change` |
| Line support | Fraction of the contour on the pencil line | `anime_qc.line_support` with `line_prior.boundary_line_support` |
| Background-line ratio | How much background line art the mask swallowed | `anime_qc.background_line_ratio` |
| Suggested keyframes | Frames a fixer must look at | `anime_qc.suggest_keyframes` |
| Human A/B | The only metric that decides shipping | Manual, on a fixed clip list, blind order |

Temporal boiling is the one to lead with. A model that wins on IoU but boils on
held cels is worse for an AMV editor than the stock model.

### 5.2 The eval harness

Add `tools/sam2_training/eval_anime_sam.py` (a script, matching the existing
`tools/sam2_training/` convention, not a `tests/` file, because it needs models
and a GPU):

- Input: a dataset root built by `training_data.export_clip`, with a `val` split
  (`ImageSets/val.txt`) that no training run has seen.
- Load two `SAM2Manager`s: stock and custom (the custom one is installed via the
  normal sidecar, so `ModelSize.CUSTOM` just works).
- For each held-out clip: run image-prediction scoring (deterministic, no
  propagation randomness) and, separately, a full propagate run through
  `pipeline.run_job` to measure boiling on a real sequence.
- Emit a JSON + Markdown table per checkpoint with the metrics in 5.1, plus the
  mean over clips and a per-failure-type breakdown if the dataset sidecar
  records failure types (section 3.2 step 4).
- The `tests/` suite gains only the cheap parts: a unit test that the metric
  functions agree with hand-built masks (extend `tests/test_anime_qc.py` and add
  `tests/test_eval_metrics.py`), so `pytest tests/ -q` stays GPU-free.

### 5.3 Release path into the app

1. `python tools/sam2_training/install_lugnica.py <ckpt.pt> --base base_plus
   --label "Lugnica SAM v2"` copies the checkpoint into
   `checkpoints/custom/` and writes `sam2_custom.json`.
2. The engine's `best` tier picks it up (`resolve_model_name("best")`), and the
   desktop app's AI Model dropdown lists it.
3. The panel's "Best" option now loads it; `describe()` reports
   `custom_model: true`, `custom_label: "Lugnica SAM v2"`, and
   `quality_models.best == "custom"`, so the panel names it.
4. Add a version/capability report: extend `describe()` with
   `custom_eval` (metrics from 5.2, stored in the sidecar or a sibling JSON) and
   `custom_base` (which base architecture). The panel can then say "Lugnica SAM
   v2 (Base+, boundary F 0.91)" rather than just the label. This is optional and
   should be additive so old sidecars keep working.

---

## 6 · Staged plan

### Stage 1 - a 200-frame eval set and a baseline number

The point of stage 1 is to turn "it feels worse on some frames" into a number.
No training.

- Build a held-out set: 200 frames from 8-12 clips the model has not been
  trained on, exported through `training_data.export_clip` into a `val` split,
  with GT corrected by hand where the stock model is wrong.
- Write `tools/sam2_training/eval_anime_sam.py` and score the stock SAM 2.1
  Base+ (and Large, for an upper bound).
- Record the baseline in a new table in this document.
- Acceptance: the script reproduces the baseline on a re-run (deterministic
  image scoring), and the numbers are in the doc.
- Effort: 3-5 days (mostly labelling the 200 frames).

### Stage 2 - a LoRA on the decoder

- Implement LoRA in the training environment only; train on a real correction
  set (target 300-1000 frames, 5-20 clips, per `docs/LUGNICA-SAM.md`).
- Merge the adapters into a stock Base+ checkpoint and install it via
  `install_lugnica.py`.
- Acceptance: the merged checkpoint loads through `build_sam2` with the stock
  `sam2.1_hiera_b+.yaml` (i.e. `_config_name(CUSTOM)` returns the stock config,
  no `"Unknown base config"`), and beats the stage-1 baseline on boundary
  F-score and boiling, with no regression on IoU.
- Effort: 5-10 days, plus GPU time.

### Stage 3 - a full custom checkpoint in the sidecar format

- Move from LoRA to the full decoder fine-tune with the boundary-aware loss and
  the cel-hold augmentation, using the Colab driver or a local run.
- Ship it as "Lugnica SAM v2": sidecar + `install_lugnica.py` + the "Best" tier
  + the eval report in `describe()`.
- Acceptance: the new checkpoint is installed with one command on a clean
  machine; the panel's "Best" names it; the eval harness shows it beats the
  stock model on the stage-1 set; and a human A/B on a fixed clip list prefers
  it. Rollback is a sidecar edit (documented behaviour).
- Effort: 10-15 days including training time and review.

Rough total: 3-5 weeks of engineering plus GPU/Colab time.

---

## 7 · What we are not doing

- **We are not changing the sidecar format** in stages 1-2. A custom checkpoint
  remains a same-architecture weight set behind `sam2_custom.json`. Shape
  changes are a separate, larger change.
- **We are not training soft/alpha GT into SAM 2.** Softness is the job of
  `edge_finish` and MatAnyone 2; SAM 2 stays a binary segmenter.
- **We are not downloading or redistributing anime mask datasets.** The dataset
  is the user's own corrections.
- **We are not adding `peft`, `timm` or any training dependency to the app.**
  Training dependencies belong in the training environment.
- **We are not promising a per-resolution or per-studio model.** One custom
  checkpoint that is better on average is the goal; the app already has per-shot
  profiles for the rest.
- **We are not auto-shipping a fine-tune to every user.** It stays a
  user-installed, opt-in "Best" tier, consistent with the offline, no-telemetry
  stance.

---

## 8 · Open questions / how to verify

1. **How many corrected frames are actually enough?** `docs/LUGNICA-SAM.md`
   says 300-1000 from 5-20 clips. Verify with the stage-1 harness on two
   datasets of different sizes before committing to a labelling budget.
2. **Does the boundary-aware loss help or hurt at this data scale?** It is
   untested here. Compare boundary F-score with and without it on the stage-1
   set; if it is neutral, drop it and keep the shipped loss.
3. **Is LoRA worth the merge complexity?** Measure stage-2 LoRA against a full
   decoder fine-tune on the same data. If full wins, skip LoRA entirely.
4. **Does cel-hold duplication overfit temporally?** It could teach the model
   that consecutive frames are always identical. Evaluate on a clip with fast
   motion to check.
5. **Does the custom model break the engine's non-anime presets?** A fine-tune
   is available under "Best" regardless of preset. Test on a live-action clip;
   if it regresses, only offer the custom tier when `preset == "anime"`
   (`resolve_model_name` is the single place to change).
6. **Which base ships?** Base+ is the documented sweet spot, but verify against
   Small on the eval set; a smaller checkpoint that is equally good is a better
   download for an 8 GB user.
7. **Is the eval set representative?** 200 frames from a handful of shows can
   be gamed. Keep a second, untouched "test" split and only look at it at stage
   3 release.
