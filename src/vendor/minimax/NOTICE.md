# NOTICE — MiniMax-Remover architecture module

This directory contains a single, adapted architecture module used by Better
Brush's AI object-removal stage. It is **not** a copy of the upstream repository.

## Provenance

- **Upstream project:** *MiniMax-Remover* — Bojia Zi, Weixuan Peng, Xianbiao Qi,
  Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong.
  - Code: https://github.com/zibojia/MiniMax-Remover
    (upstream source file `transformer_minimax_remover.py`)
  - Weights: https://huggingface.co/zibojia/minimax-remover
  - Paper: https://arxiv.org/abs/2505.24873
- The upstream repository publishes **no license file**, so its source remains
  "all rights reserved" by its authors. This module is therefore included only as
  a minimal adaptation required for interoperability with the separately licensed
  weights, with attribution retained. No other upstream file is reproduced.
- The upstream source file carries no copyright or license header of its own; the
  attribution comment at the top of `__init__.py` was added by Better Brush.

## Modifications made by Better Brush

- Class renamed `Transformer3DModel` → `MinimaxRemoverTransformer3DModel`. The
  upstream file reused diffusers' historical class name, which clashes with the
  Wan transformer shipped by `diffusers` (hence the rename).
- Vendored under `src/vendor/minimax/` so the package resolves without the
  upstream repository on `sys.path`.

## Weights

The model weights are **not** distributed with Better Brush. They are downloaded
from Hugging Face the first time the object-removal feature is used, and are
licensed Creative Commons Attribution-NonCommercial 4.0 International
(CC-BY-NC-4.0) — non-commercial use only.

See the repository's `THIRD-PARTY-NOTICES.md` and `LICENSE` for the full picture.
