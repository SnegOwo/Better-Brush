# Fine-tuning SAM 2 for anime / AMV

Better Brush ships everything you need to train your own anime-tuned SAM 2
checkpoint on **one consumer GPU**. The workflow is the same "model in the
loop" idea Meta used to build SAM 2 itself: Better Brush makes the masks, you
fix them, Better Brush exports them, the official SAM 2 trainer learns from
your corrections.

> **Why not just use a sharper prompt?** The Anime/AMV content preset only
> tunes the decoder thresholds of the stock model. A fine-tune changes the
> weights themselves — it learns what an anime character *is* (flat cel
> shading, hair strands, line art) instead of being told to accept messier
> candidates.

## The loop

```
 1. open a clip in Better Brush
 2. point / brush / propagate to a matte you're happy with
 3. File > Save Training Data...   -> frames + corrected masks on disk
 4. repeat 1-3 over 5-20 clips (more variety = better model)
 5. train with tools/sam2_training/anime_amv.yaml
 6. drop the checkpoint into checkpoints/custom/ and select it in AI Model
```

### 1-3 — Build the dataset in Better Brush

**File > Save Training Data…** writes a folder in the layout the official
trainer already understands:

```
dataset/
  JPEGImages/<clip>/00042.jpg        # the source frame
  Annotations/<clip>/0/00042.png     # object 1's corrected mask
  Annotations/<clip>/1/00042.jpg     # object 2's corrected mask
  ImageSets/train.txt                # one <clip> name per line
```

Only frames that actually carry a mask are exported, so the positive signal
is not diluted by hundreds of blank frames. Every clip you export gets
appended to `train.txt`, so one dataset folder can grow across sessions.

**What is worth exporting:** clips you had to *correct*. A mask you accepted
unchanged teaches the model what it already knows; the frames where you
painted over a mistake are where the value is. Export the corrections, not
the easy wins.

### 5 — Train

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[dev]"
mkdir -p checkpoints && cd checkpoints
./download_ckpts.sh          # or just the base_plus checkpoint
cd ..
cp /path/to/Better-Brush/tools/sam2_training/anime_amv.yaml \
   configs/sam2.1_training/anime_amv.yaml
```

Edit the three paths at the top of the config:

```yaml
dataset:
  img_folder: /path/to/dataset/JPEGImages
  gt_folder:  /path/to/dataset/Annotations
  file_list_txt: /path/to/dataset/ImageSets/train.txt
```

Then:

```bash
python training/train.py -c configs/sam2.1_training/anime_amv.yaml \
    --use-cluster 0 --num-gpus 1
```

TensorBoard (optional): `tensorboard --logdir sam2_logs/<run>/tensorboard`.

### 6 — Use the result

```bash
mkdir -p <Better-Brush>/checkpoints/custom
cp sam2_logs/.../checkpoints/checkpoint.pt \
   <Better-Brush>/checkpoints/custom/anime_amv_v1.pt
cat > <Better-Brush>/checkpoints/custom/sam2_custom.json <<'JSON'
{
  "checkpoint": "anime_amv_v1.pt",
  "config": "sam2.1_hiera_b+.yaml",
  "label": "Anime AMV v1"
}
JSON
```

Restart Better Brush — the model appears in the **AI Model** dropdown under
that label and can be selected per shot from the Shot Profile dialog like any
built-in tier. Keep the old checkpoint around; rename the sidecar to switch
back.

## Google Colab (train without a local GPU)

You do not need a local GPU: the same recipe runs on a **free Colab T4** (or an
L4/A100 on Colab Pro) from the browser. Everything is under
[`tools/sam2_training/colab/`](../tools/sam2_training/colab/README.md):

| File | Purpose |
|---|---|
| `BetterBrush_SAM2_Finetune.ipynb` | Open in Colab → upload your dataset → run the cells |
| `sam2_colab_finetune.py` | Driver: clone SAM 2, install, write the config, train |
| `anime_amv_colab.yaml` | Colab-tuned recipe (384 px, 2 frames, `float16` on T4) |

Quick start:

1. In Better Brush, **File → Save Training Data…** over 5–20 clips, then upload
   the `dataset/` folder to Google Drive.
2. Open the notebook in Colab
   (`Runtime → Change runtime type → GPU`) and upload `anime_amv_colab.yaml` +
   `sam2_colab_finetune.py` when prompted.
3. Run the cells. The driver **auto-detects the GPU**: `float16` on a T4 (no
   bfloat16), `bfloat16` on an L4/A100. On the free T4 it trains at
   `resolution 384`, `num_frames 2`; on an L4/A100 try `512–768` / `4`.
4. The last cell copies the checkpoint to Drive; install it exactly as in
   [step 6](#6--use-the-result) above.

Command-line equivalent:

```bash
python sam2_colab_finetune.py --dataset /content/drive/MyDrive/BetterBrush/dataset \
    --resolution 384 --num-frames 2
```

> **Lowering `resolution` does not require touching `feat_sizes`.** SAM 2's
> `RoPEAttention` recomputes its rotary embeddings whenever the token count
> differs (`if self.freqs_cis.shape[0] != q.shape[-2]: ...`), so the `[64, 64]`
> in the config is only a default. Keeping square inputs (the recipe's
> `RandomResizeAPI(square: true)`) is what matters.

Colab tips: keep the tab open (free sessions are killed when idle), save the
checkpoint to Drive as you go, and lower `--resolution` first if you hit a
`CUDA out of memory`. See the Colab README for a fuller troubleshooting list.

## Hardware: what fits where

The official MOSE recipe assumes 8×A100 80 GB. What makes it fit on one card:

| Setting | Effect |
|---|---|
| `freeze_image_encoder: True` | The Hiera trunk stays frozen — the single biggest memory saver, and the part anime quality depends on least |
| `scratch.resolution: 512` | Drop to **384** on an 8 GB laptop |
| `scratch.num_frames: 4` | Short clips still train the temporal memory; 2 if you OOM |
| `scratch.train_batch_size: 1` | Keep at 1 |
| `optim.amp.amp_dtype` | `bfloat16` on 3060/5060/Colab-L4; **`float16` on a Colab free T4** (no bf16) |

Rough guide (batch 1, 512 px, 4 frames, frozen encoder):

| GPU | resolution | notes |
|---|---|---|
| RTX 3060 12 GB | 512 | comfortable, ~4-6 it/s |
| RTX 5060 Laptop 8 GB | 384-448 | fits; keep `num_frames: 2-4` |
| Colab free T4 16 GB | 384 | works, but slow and needs `amp_dtype: float16` |
| 24 GB desktop | 768-1024 | raise resolution, then epochs |

If it still OOMs: halve `resolution` first, then `num_frames`, then set
`use_act_ckpt_iterative_pt_sampling: True`.

## Getting a good result

**Data beats hyper-parameters.** ~300-1000 corrected frames from 5-20
different shows beats 5,000 frames from a single clip, because the model
overfits to one art style fast. Vary:

- art style (cel, thick line, soft shading, 3D-rendered anime)
- shot type (close-up, wide, crowd)
- subject (hair, glasses, ribbons, hands, fast motion)

**Learning rate:** `base_lr: 2.0e-5` is a safe default when the encoder is
frozen. If the model forgets how to track (masks drifting across frames),
lower it to `1e-5`. If it barely improves, raise to `5e-5`.

**Epochs:** watch `loss_mask`. 20 epochs on a small dataset is usually
enough; more mostly memorises your clips.

**Validate on a held-out clip** — put one clip in a separate
`ImageSets/val.txt`, segment it with Better Brush without exporting, and
compare before/after on frames the model never saw.

## Legal note

The *training code* is Apache-2.0, and a model you fine-tune yourself is
yours to license however you like — but the dataset is yours too, which is
the point of building it from your own corrections. Do not train on
third-party mask datasets you download (anime instance-segmentation sets are
typically research-licensed and built from copyrighted artwork); the
model-in-the-loop route keeps the provenance clean.
