"""Cel-hold detection for anime mask sequences.

Hand-drawn animation is not redrawn every frame.  A "cel" is held — usually on
2s or 3s, sometimes for a dozen frames — so large stretches of an anime clip are
literally the same drawing.  A frame-by-frame tracker does not know that and
re-solves every frame anyway, which is how you get *boiling*: the matte contour
twitches on frames that are pixel-identical to the one before, while the artwork
is stone still.  It also wastes inference time on frames whose answer is already
known.

This module finds held frames cheaply (a downscaled signature comparison) and
copies the previous mask verbatim across a hold.  On a hold the matte is then
not merely stable, it is *identical* — boiling is zero by construction — and the
copied frames cost nothing to segment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

#: Default mean absolute difference (on a 0-255 gray signature) at or below
#: which two frames count as the same drawing.  Real cel holds sit near 0; live
#: action and camera pans sit far above this.
_DEFAULT_THRESHOLD = 2.0

#: Signature size; small enough to be cheap, large enough to catch a cut.
_SIGNATURE_SIZE = 64


@dataclass(frozen=True)
class CelHoldResult:
    """Outcome of a cel-hold pass."""

    masks: dict[int, np.ndarray]
    copied: tuple[int, ...]

    @property
    def copied_count(self) -> int:
        """Return how many frames were filled from a preceding hold."""
        return len(self.copied)


def frame_signature(frame: np.ndarray, size: int = _SIGNATURE_SIZE) -> np.ndarray:
    """Return a small gray signature for a frame (uint8, ``size`` x ``size``)."""
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] >= 3:
        gray = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        return np.zeros((size, size), dtype=np.uint8)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    if resized.dtype != np.uint8:
        resized = np.clip(resized.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    return resized


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute difference between two signatures."""
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    if left.shape != right.shape:
        # Resize the smaller comparison rather than failing; callers pass
        # signatures of one canonical size.
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return float(np.mean(np.abs(left - right)))


def frames_match(
    a: np.ndarray,
    b: np.ndarray,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """Return whether two frames look like the same held drawing."""
    return signature_distance(frame_signature(a), frame_signature(b)) <= float(
        threshold
    )


def find_holds(
    frames: dict[int, np.ndarray],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> set[int]:
    """Return the frame indices that are a held repeat of a previous frame."""
    indices = sorted(frames)
    holds: set[int] = set()
    previous_index: int | None = None
    for index in indices:
        if previous_index is not None and frames_match(
            frames[previous_index], frames[index], threshold=threshold
        ):
            holds.add(index)
        previous_index = index
    return holds


def apply_cel_hold(
    masks: dict[int, np.ndarray],
    frames: dict[int, np.ndarray],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> CelHoldResult:
    """Copy masks verbatim across held frames.

    Walks the frames in index order.  When a frame is a held repeat of the
    previous frame that has a mask, the previous *result* mask is copied onto
    it, so a run of holds collapses to the first frame's mask and stays exactly
    identical down the chain.

    Args:
        masks: Propagated masks keyed by frame index.  Never mutated.
        frames: Source frames keyed by frame index (superset of the mask keys
            is fine; only keys present in ``masks`` are written).
        threshold: Signature distance at or below which frames are the same cel.

    Returns:
        A :class:`CelHoldResult` with a new mask mapping and the copied indices.
    """
    result: dict[int, np.ndarray] = {idx: np.asarray(mask) for idx, mask in masks.items()}
    if len(result) < 2:
        return CelHoldResult(masks=result, copied=())

    copied: list[int] = []
    previous_index: int | None = None
    for index in sorted(result):
        if (
            previous_index is not None
            and previous_index in frames
            and index in frames
            and frames_match(
                frames[previous_index], frames[index], threshold=threshold
            )
        ):
            result[index] = result[previous_index]
            copied.append(index)
        previous_index = index
    return CelHoldResult(masks=result, copied=tuple(copied))


def hold_runs(
    frames: dict[int, np.ndarray],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[tuple[int, int]]:
    """Return ``(first, last)`` inclusive runs of a held drawing.

    Useful for showing "this cel is held for N frames" and for threading the
    same matte through the whole run in one go.
    """
    indices = sorted(frames)
    runs: list[tuple[int, int]] = []
    if not indices:
        return runs
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if frames_match(frames[previous], frames[index], threshold=threshold):
            previous = index
            continue
        if previous > start:
            runs.append((start, previous))
        start = index
        previous = index
    if previous > start:
        runs.append((start, previous))
    return runs


def held_frame_count(frames: Sequence[np.ndarray], threshold: float = _DEFAULT_THRESHOLD) -> int:
    """Return how many consecutive frames in a sequence are holds."""
    indexed = {i: frame for i, frame in enumerate(frames)}
    return len(find_holds(indexed, threshold=threshold))


__all__ = [
    "CelHoldResult",
    "apply_cel_hold",
    "find_holds",
    "frame_signature",
    "frames_match",
    "held_frame_count",
    "hold_runs",
    "signature_distance",
]
