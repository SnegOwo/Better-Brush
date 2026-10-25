"""Post-processing for a finished alpha sequence (the panel's toggle).

The desktop app's **Post-process masks** step runs on the engine's masks,
inside ``src.engine.pipeline.run_job``, where there is no knob on ``JobRequest``
to switch it off (hole-filling there is unconditional, and by design — see
``_close_holes``). What that step *gives the user* is two visible things, and
both can be applied to the finished alpha just as well, from outside:

* **temporal smoothing** — neighbouring frames mostly agree, so per-pixel
  flicker and matte "boiling" between held frames go away;
* **edge cleanup** — the alpha ramp is smoothed *inside the edge band only*, so
  staircase jaggies disappear without turning a solid interior soft.

This is the adapter's own pass (protocol stage ``post``), which is why the
toggle lives in the panel rather than in the engine. Every frame it rewrites is
counted in the returned dict and reported in the status message, so "on" and
"off" are distinguishable from the outside and never merely decorative.

Standard library + numpy/cv2 only; no torch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

#: Above this many frames the temporal pass would hold every alpha in memory at
#: once; fall back to the edge cleanup alone rather than swap.
_MAX_TEMPORAL_FRAMES = 400

#: Neighbouring frames to compare, each side.
DEFAULT_WINDOW = 2

#: Similarity falloff (alpha units) for the temporal blend: a neighbour 24 grey
#: levels away still contributes about a third, one 96 away contributes nothing.
DEFAULT_TAU = 24.0
DEFAULT_GATE = 96.0

#: The edge band, in alpha levels. Outside it a frame is unambiguously in or
#: out, and smoothing there could only soften a hard boundary.
_BAND_LOW = 12
_BAND_HIGH = 243


class PostProcessCancelled(Exception):
    """The adapter saw a cancel flag while post-processing."""


def _alphas(files: list[Path]) -> list[np.ndarray]:
    import cv2

    out: list[np.ndarray] = []
    for path in files:
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            raise ValueError(f"{path.name} is not an RGBA frame")
        out.append(rgba[:, :, 3])
    return out


def _temporal(
    alphas: list[np.ndarray],
    index: int,
    window: int,
    tau: float,
    gate: float,
) -> np.ndarray:
    """Similarity-gated mean of the surrounding frames, self included.

    A hard gate keeps a moving subject from smearing: a neighbour that
    disagrees by more than ``gate`` levels is dropped entirely, so a silhouette
    crossing the background blends with *itself*, not with what passes behind
    it.
    """
    ref = alphas[index].astype(np.float32)
    acc = ref.copy()
    weight = np.ones_like(ref)
    for other in range(index - window, index + window + 1):
        if other == index or other < 0 or other >= len(alphas):
            continue
        neighbour = alphas[other].astype(np.float32)
        distance = np.abs(neighbour - ref)
        w = np.exp(-distance / tau)
        w[distance > gate] = 0.0
        acc += neighbour * w
        weight += w
    return np.clip(acc / np.maximum(weight, 1e-6), 0.0, 255.0).astype(np.uint8)


def _edge_clean(alpha: np.ndarray) -> np.ndarray:
    """Smooth the ramp where it is a ramp, leave solid regions alone."""
    import cv2

    smoothed = cv2.medianBlur(alpha, 3)
    band = (alpha > _BAND_LOW) & (alpha < _BAND_HIGH)
    if not band.any():
        return alpha
    out = alpha.copy()
    out[band] = smoothed[band]
    return out


def post_process_sequence(
    png_dir: Path,
    *,
    window: int = DEFAULT_WINDOW,
    tau: float = DEFAULT_TAU,
    gate: float = DEFAULT_GATE,
    on_progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Rewrite every RGBA frame in *png_dir* in place. Returns what it did."""
    import cv2

    started = time.time()
    files = sorted(png_dir.glob("*.png"))
    total = len(files)
    if total == 0:
        return {"frames": 0, "temporal_window": 0, "seconds": 0.0}

    use_temporal = window > 0 and total <= _MAX_TEMPORAL_FRAMES
    # Pre-load only the alphas: the RGB of each frame is read and written back
    # one at a time, so memory stays at one frame plus the alpha stack.
    alphas = _alphas(files) if use_temporal else []

    rewritten = 0
    for i, path in enumerate(files):
        if cancelled is not None and cancelled():
            raise PostProcessCancelled()
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            continue  # a non-RGBA frame is left exactly as it was found
        alpha = rgba[:, :, 3]
        if use_temporal:
            alpha = _temporal(alphas, i, window, tau, gate)
        alpha = _edge_clean(alpha)
        rgba[:, :, 3] = alpha
        if not cv2.imwrite(str(path), rgba):
            raise ValueError(f"could not rewrite {path.name}")
        rewritten += 1
        if on_progress is not None and (i % 5 == 0 or i == total - 1):
            on_progress(i + 1, total, "smoothing the matte")

    return {
        "frames": rewritten,
        "temporal_window": window if use_temporal else 0,
        "seconds": round(time.time() - started, 3),
    }
