# Third-Party Notices

Better Brush bundles and depends on the third-party components listed below.
Each remains the property of its respective owners and is used under its own
license. This file is provided to satisfy the attribution and notice
requirements of those licenses. Where a component's full license text is
shipped with the software, the path is given.

**Better Brush's own source code is licensed under FSL-1.1-MIT** (see
`LICENSE`) — free to use, modify and share, including for monetised video
content. Some of the bundled model weights below are licensed for
non-commercial use only; **those third-party terms take precedence for the
weights**, and are unaffected by the code license. Commercial licensing of the
weights requires permission from their respective authors.

---

## Model weights bundled with the application

### SAM 2 / SAM 2.1 — Meta Platforms, Inc.
- **Files:** `checkpoints/sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt`,
  `sam2.1_hiera_base_plus.pt`
- **License:** Apache License 2.0
- **Source:** https://github.com/facebookresearch/sam2

### anime_amv_v1.pt — custom fine-tune (Sneg)
- **Files:** `checkpoints/custom/anime_amv_v1.pt`
- **Provenance:** a fine-tune of the SAM 2.1 base checkpoint above, produced by
  Sneg for anime/AMV footage. Because it is a derivative of an Apache-2.0 work,
  it is distributed under the same Apache-2.0 terms as the base weights.

### EfficientTAM — Meta AI
- **Files:** `checkpoints/efficienttam_s_512x512.pt`, `src/vendor/efficienttam/`
- **License:** Apache License 2.0 (see `src/vendor/efficienttam/LICENSE.apache-2.0.txt`)
- **Source:** https://github.com/yformer/EfficientTAM

### MatAnyone / MatAnyone 2 — S-Lab, Nanyang Technological University / SenseTime Research
- **Files:** architecture code at `src/vendor/matanyone/` and
  `src/vendor/matanyone2/`; MatAnyone 2 weights at `checkpoints/matanyone2/`.
  (MatAnyone 1 weights are not bundled — they download on first use.)
- **License:** NTU S-Lab License 1.0 — **non-commercial use only**
  (see `src/vendor/matanyone2/LICENSE.ntu-slab.txt`). Copyright 2023 S-Lab.
  Commercial use requires permission from the authors.
- **Source:** https://github.com/pq-yang/MatAnyone and
  https://github.com/pq-yang/MatAnyone2

### MiniMax-Remover — Bojia Zi et al.
- **Files:** one adapted architecture module at `src/vendor/minimax/` (provenance
  and the list of modifications are in `src/vendor/minimax/NOTICE.md`). The model
  weights are **not bundled**: they are downloaded from Hugging Face the first
  time you use the AI object-removal feature.
- **Code:** the upstream repository (https://github.com/zibojia/MiniMax-Remover)
  publishes **no license file**; its source therefore remains all rights reserved
  by its authors. Better Brush ships only the single adapted module named above,
  keeps the upstream attribution in its header, and reproduces no other upstream
  file.
- **Weights:** Creative Commons Attribution-NonCommercial 4.0 International
  (CC-BY-NC-4.0) — **non-commercial use only**.
- **Source:** https://github.com/zibojia/MiniMax-Remover and
  https://huggingface.co/zibojia/minimax-remover

### LaMa (big-lama) — Samsung AI Center
- **Files:** `checkpoints/big-lama.pt`, `checkpoints/lama_fp32.onnx`
- **License:** Apache License 2.0
- **Source:** https://github.com/advimman/lama

### MediaPipe Face Landmarker — Google LLC
- **Files:** `checkpoints/face_landmarker.task`
- **License:** Apache License 2.0
- **Source:** https://developers.google.com/mediapipe

---

## Software components

| Component | License | Notes |
| --- | --- | --- |
| Qt / PySide6 / shiboken6 (The Qt Company) | LGPL-3.0 | Used unmodified; Qt is dynamically linked. Qt Virtual Keyboard modules, if present, are GPL-3.0. |
| PyTorch / torchvision (Meta) | BSD-3-Clause | |
| NumPy, SciPy | BSD-3-Clause | |
| OpenCV (opencv-python) | Apache-2.0 | |
| Pillow | MIT-CMU (HPND) | |
| FFmpeg (Gyan Doshi build) | GPL-3.0 | Bundled as `ffmpeg.exe`; invokes the GPL build, so the FFmpeg binaries are GPL-3.0. |
| PySceneDetect | BSD-3-Clause | |
| Hugging Face `diffusers`, `accelerate`, `huggingface_hub`, `timm`, `safetensors` | Apache-2.0 | |
| `lz4` (python-lz4) | BSD-2-Clause | |
| `pynvml` / `nvidia-ml-py` | BSD-3-Clause | |
| `psutil` | BSD-3-Clause | |
| `imageio`, `imageio-ffmpeg` | BSD-2-Clause / BSD | |
| `onnxruntime` (Microsoft) | MIT | |
| Hydra / OmegaConf (Meta) | MIT / BSD-3-Clause | |
| `iopath` (Meta) | MIT | |
| `websockets` | BSD-3-Clause | |
| `watchfiles` | MIT | |

Full license texts for the Python dependencies above are available from their
respective PyPI/upstream distributions. Where a component ships a license file
inside this package, it is retained at its original path.

---

## Notes on the non-commercial components

The NTU S-Lab (MatAnyone 1 & 2 code and MatAnyone 2 weights) and CC-BY-NC-4.0
(MiniMax-Remover weights) terms permit redistribution and use **for
non-commercial purposes only**, provided the copyright notices are retained.
Better Brush therefore:

- redistributes the MatAnyone code and MatAnyone 2 weights unmodified, with their
  license texts (`LICENSE.ntu-slab.txt`);
- ships them for use only in this free build, and does not grant any
  commercial rights to them beyond what their own licenses allow.

Two components in this list are **not** redistributed at all: the MatAnyone 1
weights and the MiniMax-Remover weights are downloaded from their upstream hosts
on first use, under the same non-commercial terms. The MiniMax-Remover
*architecture code* is a separate case — upstream publishes no license file, so
Better Brush ships only the single adapted module described above
(`src/vendor/minimax/`, with `NOTICE.md`) and no other upstream file.

If you need a commercially licensed build, contact dev@sneg.dev.
