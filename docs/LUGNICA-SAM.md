# Build "Lugnica SAM"

A short, opinionated walkthrough for turning your own corrections into a
fine-tuned anime/AMV **SAM 2.1** checkpoint that shows up in Better Brush under
the name **Lugnica SAM**. The deep reference is
[`FINE-TUNING.md`](FINE-TUNING.md); this page is the "just tell me what to do"
version.

> **The idea:** Better Brush proposes masks → you fix the ones it gets wrong →
> those corrections are the training data → SAM 2's official trainer learns
> from them. That's the same "model in the loop" loop Meta used to build SAM 2.

---

## TL;DR

1. Export corrections from Better Brush over **5–20 clips** (**File → Save
   Training Data…**).
2. Train in **Colab** (no local GPU needed): run the notebook, or one command —
   `python sam2_colab_finetune.py --dataset <dir> --base base_plus`.
3. Download the checkpoint, then on your PC:
   `python tools/sam2_training/install_lugnica.py <checkpoint.pt> --base base_plus`.
4. Restart Better Brush and pick **Lugnica SAM** under **AI Model**.

---

## Which base should I build on?

Lugnica SAM is whatever SAM 2.1 size you fine-tune. All four work with the same
recipe; only speed and peak quality change.

| `--base` | Backbone | Speed | Anime quality | Recommended for |
|---|---|:--:|:--:|---|
| `tiny` | Hiera-T (0.15 GB) | ⚡⚡⚡ | fair | fastest previews only |
| `small` | Hiera-S (0.2 GB) | ⚡⚡ | good | light/laptop, quick re-runs |
| **`base_plus`** | Hiera-B+ (0.3 GB) | ⚡ | **best** | **the default sweet spot** |
| `large` | Hiera-L (0.9 GB) | 🐢 | best+ | 24 GB cards, max quality |

**Use `base_plus`.** It's the quality/speed sweet spot, fits an 8 GB laptop at
384 px, and is the recipe that ships tuned. Drop to `small` only if you want a
smaller file and snappier inference. Point-and-click segmentation quality is
dominated by your *data*, not by going up to `large`.

```bash
# any base:
python sam2_colab_finetune.py --dataset /content/dataset --base small
```

The matching base checkpoint is downloaded automatically and the config's
architecture is rewritten to match, so the trained weights always line up with
the base they started from.

---

## My GPU or Colab?

| | Your RTX 5060 Laptop (8 GB) | Google Colab |
|---|---|---|
| Cost | free | free tier (T4) or Pro (L4/A100) |
| Speed at 384 px | faster per step | slower per step on free T4 |
| Uses your machine | **yes** — you can't use it while training | no — runs in the browser |
| Setup | clone/install SAM 2 locally | zero — the notebook does it |
| Best for | short runs, iteration | **leaving it running unattended** |

**Recommended: Colab.** Two reasons: you keep your laptop, and a free T4 has
16 GB so it never OOMs like your 8 GB card can. Use the laptop only when you
want to iterate quickly on a tiny dataset.

- Free Colab runs `resolution 384`, `num_frames 2`, `float16` (the T4 has no
  bfloat16). The driver detects this automatically.
- Colab Pro (L4/A100) → pass `--resolution 512` (up to 768), `--num-frames 4`,
  and it switches to `bfloat16`.

---

## Do I need more training material?

Yes — but **more correction, not more footage**. The model already knows how to
segment things; it needs to learn where *you* disagree with it.

- **Target ~300–1000 corrected frames from 5–20 different clips/shows.**
- Spread the variety: cel vs soft shading, close-up vs wide, hair/ribbons/
  glasses/hands, fast motion.
- Export the frames you **fixed**, not the ones it already nailed — those teach
  nothing.
- 5,000 frames from one clip is *worse* than 500 from ten: it overfits one art
  style fast.

You don't need to buy or scrape anything. Building it from your own corrections
also keeps the licensing clean (see the legal note in
[`FINE-TUNING.md`](FINE-TUNING.md#legal-note)).

---

## The simple step-by-step

### 1 · Build the dataset (in Better Brush, on your PC)

1. Open a clip, point/brush/propagate to a matte you're happy with.
2. **File → Save Training Data…** → write into one `dataset/` folder.
3. Repeat over 5–20 clips. Everything appends to the same folder.

You get the layout SAM 2's trainer already expects:

```
dataset/
  JPEGImages/<clip>/00042.jpg
  Annotations/<clip>/0/00042.png
  ImageSets/train.txt
```

### 2 · Train (Colab — recommended)

1. Zip/upload `dataset/` to Google Drive.
2. Open [`BetterBrush_SAM2_Finetune.ipynb`](../tools/sam2_training/colab/BetterBrush_SAM2_Finetune.ipynb)
   in Colab (`Runtime → Change runtime type → GPU`).
3. Run the cells. The driver clones SAM 2, installs it, downloads the base
   checkpoint, writes the config and trains.

One-command equivalent (run *inside* the notebook, or on any Linux GPU box):

```bash
python sam2_colab_finetune.py \
    --dataset /content/drive/MyDrive/BetterBrush/dataset \
    --base base_plus --resolution 384 --num-frames 2
```

### 2b · …or train on your own GPU

```bash
python tools/sam2_training/colab/sam2_colab_finetune.py \
    --dataset C:\path\to\dataset --base base_plus --resolution 384 --num-frames 2
```

Same script; it just isn't Colab-specific.

### 3 · Grab the checkpoint

Training writes into `sam2_logs/<run>/checkpoints/`. Download the newest `.pt`
(from Colab: copy it to Drive). The driver prints the path at the end.

### 4 · Install it as "Lugnica SAM"

On the machine that runs Better Brush — **one command**:

```bash
python tools/sam2_training/install_lugnica.py path/to/checkpoint.pt --base base_plus
```

That copies the weights into `checkpoints/custom/lugnica_sam_v1.pt` and writes
the `sam2_custom.json` sidecar with `"label": "Lugnica SAM"`. Want a different
name or a lighter variant?

```bash
python tools/sam2_training/install_lugnica.py ckpt.pt --base small \
    --label "Lugnica SAM (small)" --name lugnica_sam_small_v1.pt
```

`--base` **must** match what you trained on, or the architecture won't line up.

### 5 · Select it

Restart Better Brush, open **AI Model**, and choose **Lugnica SAM**. It behaves
like any built-in tier — including per-shot selection from the Shot Profile
dialog. To roll back, point the sidecar at the previous checkpoint.

---

## Can I automate it?

Yes — the Colab path is already fully automated, including the boring parts
(clone, `pip install`, checkpoint download, config patching, AMP dtype, GPU
detection, launching `train.py`).

- **Notebook**: upload the dataset once, hit *Run all*, walk away.
- **Script**: `sam2_colab_finetune.py` runs the exact same pipeline, so you can
  script it or re-run with a different `--base`.
- **Dry run**: add `--no-train` to prepare everything and stop before training —
  handy to sanity-check the generated config.
- **Re-train**: export more corrections into the same `dataset/`, re-run; the
  folder grows and the model improves. Keep the sidecar name if you just want to
  overwrite "Lugnica SAM".

The only manual steps are uploading the dataset and installing the result — both
are intentional (they cross your PC ↔ Colab boundary).

---

## Getting a good result (the 20% that matters)

- **Resolution before epochs.** If you have headroom, raise `--resolution`
  (384 → 448 → 512) before adding epochs; anime line art loves pixels.
- **Learning rate.** `base_lr: 2e-5` with a frozen encoder is the safe default.
  Masks drifting over frames → drop to `1e-5`. Barely changing → try `5e-5`.
- **Epochs.** Watch `loss_mask`; ~20 epochs on a small set is usually enough.
  More mostly memorises your clips.
- **Validate on a clip you never exported.** Segment it with Lugnica SAM and
  compare against before. That's your real number.
- **Re-export is half of it.** The fastest win is often just exporting the
  handful of shots where the stock model still fails.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | lower `--resolution` first, then `--num-frames` |
| Colab killed / idle | keep the tab open; save the checkpoint to Drive as you go |
| Loss won't move | more/varied data, or `5e-5` LR |
| Masks drift across frames | lower LR to `1e-5`; export negative/fix frames |
| Model doesn't appear in-app | restart Better Brush; check `checkpoints/custom/sam2_custom.json` |
| "Unknown base config" on load | `--base` at install doesn't match what you trained |

## Files

| Path | What it is |
|---|---|
| `tools/sam2_training/colab/BetterBrush_SAM2_Finetune.ipynb` | The easy path — open in Colab and run |
| `tools/sam2_training/colab/sam2_colab_finetune.py` | The driver (`--base`, `--resolution`, `--no-train`, …) |
| `tools/sam2_training/colab/anime_amv_colab.yaml` | Colab-tuned training recipe |
| `tools/sam2_training/install_lugnica.py` | Installs a checkpoint as "Lugnica SAM" |
| [`FINE-TUNING.md`](FINE-TUNING.md) | Full reference: per-GPU settings, LR, legal note |
| [`tools/sam2_training/colab/README.md`](../tools/sam2_training/colab/README.md) | Colab-specific details and troubleshooting |
