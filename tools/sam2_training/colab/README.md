# Fine-tune SAM 2.1 for anime / AMV on Google Colab

Train a **SAM 2.1 Base+** checkpoint on the masks you corrected in Better Brush,
entirely in the browser — no local GPU needed. This folder is the Colab edition
of the recipe in [`../../docs/FINE-TUNING.md`](../../../docs/FINE-TUNING.md).

```
colab/
  BetterBrush_SAM2_Finetune.ipynb   # the easy path: open in Colab, run cells
  sam2_colab_finetune.py            # the driver: clone / install / config / train
  anime_amv_colab.yaml              # Colab-tuned training recipe
  README.md                         # this file
```

> The workflow is Meta's "model in the loop": Better Brush proposes masks, you
> fix the ones it gets wrong, those corrections become the training data.

---

## 0 · Build the dataset (in Better Brush, on your PC)

1. Open a clip, point / brush / propagate to a matte you're happy with.
2. **File → Save Training Data…** → write a `dataset/` folder.
3. Repeat over **5–20 clips**. Export the clips you had to *correct* — the easy
   wins teach the model nothing.

The folder looks exactly like this:

```
dataset/
  JPEGImages/<clip>/00042.jpg      # source frames
  Annotations/<clip>/0/00042.png   # object 1's corrected mask
  Annotations/<clip>/1/00042.jpg   # object 2's corrected mask (if any)
  ImageSets/train.txt              # one <clip> name per line
```

Aim for **~300–1000 corrected frames** with varied art style, shot type and
subject. All clips append to the same `train.txt`, so the dataset can grow
across sessions.

---

## 1 · Upload it to Colab

Two easy options — either is fine:

* **Google Drive (recommended):** upload `dataset/` to
  `My Drive/BetterBrush/dataset/`, then mount Drive in the notebook.
* **Zip upload:** zip the `dataset/` folder, run the notebook, and unzip it to
  `/content/dataset`.

## 2 · Open the notebook

1. Go to <https://colab.research.google.com> → **Upload** →
   `BetterBrush_SAM2_Finetune.ipynb`.
2. **Runtime → Change runtime type → Hardware accelerator → GPU.**
   * **T4 (free)** — 16 GB, no bfloat16 → the notebook uses `float16`,
     `resolution 384`, `num_frames 2`.
   * **L4 / A100 (Colab Pro)** — bfloat16 → try `resolution 512–768`,
     `num_frames 4`.
3. Run the cells top to bottom. When asked, upload
   `anime_amv_colab.yaml` **and** `sam2_colab_finetune.py` (both are in this
   folder).

The first cell checks the GPU, then the notebook clones SAM 2, installs the
training extras, downloads only the Base+ checkpoint, writes the config and
starts training.

### Prefer the command line?

`sam2_colab_finetune.py` does the same thing in one shot:

```bash
python sam2_colab_finetune.py \
    --dataset /content/drive/MyDrive/BetterBrush/dataset \
    --resolution 384 --num-frames 2
```

Useful flags: `--no-train` (prepare only), `--epochs N`,
`--amp-dtype bfloat16|float16` (auto-detected otherwise),
`--workdir /content/sam2`. Import it in a cell instead —
`import sam2_colab_finetune as bb` — to call `bb.train(repo)` and friends
step by step.

---

## 3 · Watch it learn (optional)

In a second cell / terminal:

```bash
tensorboard --logdir /content/sam2/sam2_logs --port 6006
```

`loss_mask` should fall and then flatten. **20 epochs** is usually enough on a
small dataset; more mostly memorises your clips.

## 4 · Bring the checkpoint back to Better Brush

The notebook copies the newest checkpoint to
`My Drive/BetterBrush/sam2_checkpoints/anime_amv_v1.pt`. Then, on your PC:

```text
<Better-Brush>/checkpoints/custom/anime_amv_v1.pt       <- the trained file
<Better-Brush>/checkpoints/custom/sam2_custom.json      <- the sidecar
```

`sam2_custom.json`:

```json
{
  "checkpoint": "anime_amv_v1.pt",
  "config": "sam2.1_hiera_b+.yaml",
  "label": "Anime AMV v1"
}
```

Restart Better Brush → **Anime AMV v1** appears in the **AI Model** dropdown
(and per shot in **Inspect → Shot Profiles**). Keep the stock checkpoint around;
rename the sidecar to switch back.

---

## What fits where

| GPU | `--resolution` | `--num-frames` | notes |
|---|---|---|---|
| Colab free **T4** 16 GB | 384 | 2 | works; `float16` (no bf16). Slow. |
| Colab Pro **L4** 24 GB | 512 | 4 | `bfloat16`; good balance |
| Colab Pro **A100** 40 GB | 768 | 4–8 | fastest; raise resolution then epochs |
| Local RTX 5060 Laptop 8 GB | 384–448 | 2–4 | see the main FINE-TUNING guide |
| Local RTX 3060 12 GB | 512 | 4 | comfortable |

If you still OOM: lower `--resolution` first, then `--num-frames`, then set
`use_act_ckpt_iterative_pt_sampling: true` in the config.

## Troubleshooting

* **`No module named 'sam2'`** — the `pip install -e .[dev]` cell did not finish
  (or the runtime restarted). Re-run cell 1; the install is idempotent.
* **Session disconnected / timed out** — Colab free kills idle sessions and caps
  runtime. Keep the tab open, and save the checkpoint to Drive often (the last
  cell does it for you). For long runs, prefer Colab Pro.
* **`CUDA out of memory`** — reduce `--resolution` / `--num-frames`. The dataset
  is also small enough that `--epochs` does not affect memory.
* **The model got *worse* (masks drift across frames)** — the learning rate is a
  touch high for your data; edit `base_lr` in `anime_amv_colab.yaml` to `1e-5`.
* **`image_size` / shape errors** — keep the notebook's default `[64, 64]`
  `feat_sizes` in the config; SAM 2 recomputes the rotary embeddings for the
  actual resolution at runtime, so lowering `resolution` is safe on its own.
* **Train on a held-out clip** — put one clip in a separate
  `ImageSets/val.txt` and compare before/after on frames the model never saw.

## Legal note

The SAM 2 training code is Apache-2.0, and a checkpoint you fine-tune yourself
is yours to license however you like — as is a dataset built from your own
corrections. Do not train on third-party anime mask datasets you download:
they are usually research-licensed and built from copyrighted artwork.
