"""Anime-specific matte QC metrics.

The generic inspector (:mod:`src.processing.matte_qc`) answers "is this matte
roughly right?"  It is good at hair, holes, spill and gross flicker, but it does
not know the two things that actually define a *good anime* roto:

* **Boiling** — how much the contour twitches on frames that are held drawings.
  On an anime cel the artwork is frozen, so *any* change is wrong.  This is the
  single most visible defect in an AMV and the generic flicker score misses it,
  because it does not know which frames are supposed to be identical.
* **Line integrity** — whether the contour rides the pencil line (good) or
  floats off it (fuzzy), and how much *background* line art the mask swallowed.

These are deliberately separate, cheap, dependency-light functions so they can
be shown in the inspector, logged per shot, or used to drive automatic keyframe
suggestions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from src.processing.cel_hold import _DEFAULT_THRESHOLD as _HOLD_THRESHOLD
from src.processing.cel_hold import frames_match
from src.processing.line_prior import (
    background_line_penalty,
    boundary_band,
    boundary_line_support,
    extract_lines,
)

#: Contour change above which a held pair is treated as boiling.
_DEFAULT_BOILING_THRESHOLD = 0.12


@dataclass
class AnimeQC:
    """Anime-specific quality summary for one matte (or a short sequence)."""

    boiling: float = 0.0
    line_support: float = 0.0
    background_line_ratio: float = 0.0
    held_pairs: int = 0
    pairs_checked: int = 0
    suggested_keyframes: tuple[int, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        """Clamp scores into range and normalize the keyframe tuple."""
        self.boiling = _clamp01(self.boiling)
        self.line_support = _clamp01(self.line_support)
        self.background_line_ratio = _clamp01(self.background_line_ratio)
        self.held_pairs = max(0, int(self.held_pairs))
        self.pairs_checked = max(0, int(self.pairs_checked))
        self.suggested_keyframes = tuple(int(i) for i in self.suggested_keyframes)
        self.message = str(self.message or "")

    @property
    def warning(self) -> bool:
        """Return whether any anime metric is at a concerning level."""
        return (
            self.boiling >= _DEFAULT_BOILING_THRESHOLD
            or self.background_line_ratio >= 0.2
            or (self.line_support < 0.2 and self.pairs_checked > 0)
        )

    @property
    def score(self) -> float:
        """Return a single 0-1 severity for the worst anime metric."""
        return max(self.boiling, self.background_line_ratio, 1.0 - self.line_support)

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly summary."""
        return {
            "boiling": self.boiling,
            "line_support": self.line_support,
            "background_line_ratio": self.background_line_ratio,
            "held_pairs": self.held_pairs,
            "pairs_checked": self.pairs_checked,
            "suggested_keyframes": list(self.suggested_keyframes),
            "warning": self.warning,
            "score": self.score,
            "message": self.message,
        }


def _clamp01(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _ordered_masks(
    masks: Mapping[int, np.ndarray] | Sequence[np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Return ``(index, mask)`` pairs in order for either input shape."""
    if isinstance(masks, Mapping):
        return [(int(idx), np.asarray(m)) for idx, m in sorted(masks.items())]
    return [(i, np.asarray(m)) for i, m in enumerate(masks)]


def contour_change(
    a: np.ndarray,
    b: np.ndarray,
    *,
    edge_width: int = 2,
) -> float:
    """Return the share of the contour that moved between two masks.

    Only pixels on either contour are counted, so a mask that slides as a whole
    (real motion) scores lower than a contour that twitches in place.  ``0``
    means the contours are identical.
    """
    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    if left.shape != right.shape or left.ndim != 2:
        return 0.0
    if not left.any() and not right.any():
        return 0.0
    band = boundary_band(left, width=edge_width) | boundary_band(right, width=edge_width)
    band_area = int(np.count_nonzero(band))
    if band_area == 0:
        return 0.0
    changed = np.logical_xor(left, right) & band
    return _clamp01(np.count_nonzero(changed) / band_area)


def boiling_score(
    masks: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    frames: Mapping[int, np.ndarray] | None = None,
    *,
    threshold: float = _HOLD_THRESHOLD,
    edge_width: int = 2,
) -> tuple[float, int, int]:
    """Return ``(boiling, held_pairs, pairs_checked)`` for a mask sequence.

    When *frames* is given, only consecutive pairs that are held drawings
    (the frames are near-identical) are scored, because on those pairs the
    matte is *supposed* to be identical and any difference is boiling.  Without
    frames, every consecutive pair is scored as a rough proxy.
    """
    ordered = _ordered_masks(masks)
    if len(ordered) < 2:
        return 0.0, 0, 0

    hold_pairs: list[float] = []
    all_pairs: list[float] = []
    for (prev_idx, prev_mask), (cur_idx, cur_mask) in pairwise(ordered):
        change = contour_change(prev_mask, cur_mask, edge_width=edge_width)
        all_pairs.append(change)
        if (
            frames is not None
            and prev_idx in frames
            and cur_idx in frames
            and frames_match(frames[prev_idx], frames[cur_idx], threshold=threshold)
        ):
            hold_pairs.append(change)

    chosen = hold_pairs or all_pairs
    if not chosen:
        return 0.0, 0, 0
    return _clamp01(float(np.mean(chosen))), len(hold_pairs), len(all_pairs)


def line_support(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    radius: int = 2,
    **extract_kwargs: object,
) -> float:
    """Return how well a mask contour rides the frame's line art (0-1)."""
    lines = extract_lines(frame, **extract_kwargs)  # type: ignore[arg-type]
    return boundary_line_support(mask, lines, radius=radius)


def background_line_ratio(
    frame: np.ndarray,
    mask: np.ndarray,
    points: Sequence[object],
    *,
    radius: int = 2,
    reach: int = 3,
    **extract_kwargs: object,
) -> float:
    """Return the share of the mask that is unprompted (background) line art."""
    lines = extract_lines(frame, **extract_kwargs)  # type: ignore[arg-type]
    return background_line_penalty(mask, lines, points, reach=reach)


def suggest_keyframes(
    masks: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    frames: Mapping[int, np.ndarray] | None = None,
    *,
    threshold: float = _DEFAULT_BOILING_THRESHOLD,
    edge_width: int = 2,
    limit: int = 32,
) -> tuple[int, ...]:
    """Return frame indices where the contour changes on a held drawing.

    These are the frames a fixer should look at first: the artwork has not
    changed, so the model produced a different matte for no reason.
    """
    ordered = _ordered_masks(masks)
    if len(ordered) < 2 or frames is None:
        return ()
    flagged: list[int] = []
    for (prev_idx, prev_mask), (cur_idx, cur_mask) in pairwise(ordered):
        if prev_idx not in frames or cur_idx not in frames:
            continue
        if not frames_match(frames[prev_idx], frames[cur_idx]):
            continue
        if contour_change(prev_mask, cur_mask, edge_width=edge_width) >= threshold:
            flagged.append(int(prev_idx))
            flagged.append(int(cur_idx))
    # Preserve order, drop duplicates.
    seen: list[int] = []
    for index in flagged:
        if index not in seen:
            seen.append(index)
    if limit > 0:
        seen = seen[:limit]
    return tuple(seen)


def analyze_anime_matte(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    previous_mask: np.ndarray | None = None,
    previous_frame: np.ndarray | None = None,
    points: Sequence[object] | None = None,
    radius: int = 2,
    **extract_kwargs: object,
) -> AnimeQC:
    """Inspect one matte (optionally against the previous frame) for anime issues.

    Args:
        frame: Current RGB frame.
        mask: Current binary matte.
        previous_mask: Optional previous matte; enables the boiling metric.
        previous_frame: Optional previous RGB frame; makes the boiling metric
            meaningful (it is only measured on held pairs).
        points: Optional prompt points, used to score background-line
            inclusion.
        radius: Line-snapping/inspection radius in pixels.

    Returns:
        An :class:`AnimeQC` summary.
    """
    values = np.asarray(mask, dtype=bool)
    support = line_support(frame, values, radius=radius, **extract_kwargs)
    bg_ratio = (
        background_line_ratio(frame, values, points, radius=radius, **extract_kwargs)
        if points
        else 0.0
    )

    boiling = 0.0
    held_pairs = 0
    pairs_checked = 0
    if previous_mask is not None:
        prev = np.asarray(previous_mask, dtype=bool)
        boiling = contour_change(prev, values)
        pairs_checked = 1
        if previous_frame is not None and frames_match(previous_frame, frame):
            held_pairs = 1

    parts: list[str] = []
    if held_pairs:
        parts.append("contour moves on a held cel" if boiling > 0 else "held cel is stable")
    if bg_ratio >= 0.2:
        parts.append("background line art included")
    if support < 0.2 and values.any():
        parts.append("contour off the line art")
    message = "; ".join(parts) if parts else "No anime-specific issues detected."

    return AnimeQC(
        boiling=boiling,
        line_support=support,
        background_line_ratio=bg_ratio,
        held_pairs=held_pairs,
        pairs_checked=pairs_checked,
        message=message,
    )


__all__ = [
    "AnimeQC",
    "analyze_anime_matte",
    "background_line_ratio",
    "boiling_score",
    "contour_change",
    "line_support",
    "suggest_keyframes",
]
