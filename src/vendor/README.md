# Vendored model packages

Better Brush ships a few upstream neural-network packages under `src/vendor/`
so the app runs offline without pip-installing upstream repositories. Each
package keeps its original license; the provenance and the (minimal) changes
made for this codebase are listed below.

## `efficienttam/` — EfficientTAM (Meta AI)

- **Upstream:** https://github.com/yformer/EfficientTAM
- **Checkpoints:** https://huggingface.co/yunyangx/efficient-track-anything
- **License:** Apache-2.0 (see `LICENSE.apache-2.0.txt`)
- **Used for:** the lightweight segmentation models (EfficientTAM S / Ti)
  recommended on weak GPUs that cannot host SAM 2.1 Tiny.
- **Changes:** renamed the package (`efficient_track_anything` →
  `src.vendor.efficienttam`) so it cannot collide with a pip install; configs
  are now loaded from disk with OmegaConf instead of Hydra's process-global
  search path (SAM 2 registers its own, and the two would shadow each other);
  the vendored `__init__.py` no longer touches global Hydra state; the
  example/benchmark files were dropped. The `torch.compile` fast path in the
  upstream configs is disabled by Better Brush (it needs a working C++
  toolchain and misbehaves inside the frozen Windows bundle).

## `minimax/` — MiniMax-Remover transformer

- **Upstream:** https://github.com/zibojia/MiniMax-Remover
- **Weights:** https://huggingface.co/zibojia/minimax-remover (CC-BY-NC-4.0)
- **Used for:** the AI object-removal (inpainting) engine.
- **Changes:** only the architecture file is vendored (the weights cannot
  load without the exact module layout); the class was renamed
  `Transformer3DModel` → `MinimaxRemoverTransformer3DModel` because the
  upstream name now collides with the Wan transformer shipped by diffusers.
  The training loop, pipeline glue and the app integration are Better Brush
  code (`src/processing/removal.py`).

## `matanyone/` — MatAnyone, and `matanyone2/` — MatAnyone 2

- **Upstream:** https://github.com/pq-yang/MatAnyone and
  https://github.com/pq-yang/MatAnyone2 (S-Lab, Nanyang Technological
  University / SenseTime Research)
- **Weights:** https://huggingface.co/PeiqingYang/MatAnyone and
  https://huggingface.co/PeiqingYang/MatAnyone2
- **License:** NTU S-Lab License 1.0 (see `LICENSE.ntu-slab.txt`)
- **Used for:** the learned video-matting (soft alpha) models.
- **Changes:** renamed the packages (`matanyone` → `src.vendor.matanyone`,
  `matanyone2` → `src.vendor.matanyone2`); the training-only dataset/loaders
  and `train.py` were dropped. Everything else (the network, the memory
  manager, the inference core) is upstream. The runner, the prompt/anchor
  loop, the gamma/grow post-processing and the app integration are Better
  Brush code (`src/processing/matting.py`); the runner installs a top-k clamp
  in the memory manager because the upstream `top_k=30` over fewer stored
  entries (small frames, short clips) raises instead of degrading.
