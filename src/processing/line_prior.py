"""Line-art prior for anime segmentation.

Anime (and any cel-shaded art) is built from line art: dark, thin, high-contrast
contours.  That is a gift a segmenter usually throws away.  Two failure modes
follow from ignoring it, and both are the classic anime complaints:

* **Background-line inclusion.** The character outline and the background's
  line art are the *same* black lines, so a mask happily grows along a fence,
  a desk edge or a background character's hair and swallows the background.
* **Edge wobble.**  The objectively correct anime matte edge is the pencil line
  itself.  When the model's edge floats a pixel off that line, the result reads
  as fuzzy no matter how high the IoU is.

This module extracts the line art once and provides three operations:

* :func:`extract_lines` — the line-art mask.
* :func:`background_line_penalty` — how much of a candidate mask is line art
  that the user never pointed at (a score for the selector).
* :func:`snap_to_lines` — nudge a mask edge onto the nearest line.

Everything degrades to a no-op when there is no line art to work with, so it is
safe on photographic footage too.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

#: Fraction of a contour that must already sit on line art before snapping is
#: allowed to run.  Below this the frame is treated as non-line-art (a photo, a
#: soft gradient) and the mask is returned untouched.
_SNAP_SUPPORT_FLOOR = 0.15


def _gray(frame: np.ndarray) -> np.ndarray | None:
    """Return a uint8 gray image for an RGB/RGBA/gray frame, or ``None``."""
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] >= 3:
        gray = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        return None
    if gray.dtype != np.uint8:
        gray = np.clip(gray.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    return gray


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )


def extract_lines(
    frame: np.ndarray,
    *,
    low: int = 50,
    high: int = 150,
    dilate: int = 1,
) -> np.ndarray:
    """Extract a boolean line-art mask from a frame.

    Canny finds the high-contrast contours that make up anime line art.  A small
    dilation closes the one-pixel gaps a pure edge detector leaves, which is
    what lets a matte snap onto a continuous line rather than a dotted one.

    Returns:
        Boolean ``(H, W)`` mask; all-False when the frame is unusable or has no
        edge-like content.
    """
    gray = _gray(frame)
    if gray is None or gray.size == 0:
        return np.zeros(np.asarray(frame).shape[:2], dtype=bool)
    edges = cv2.Canny(gray, int(low), int(high))
    if dilate > 0:
        edges = cv2.dilate(edges, _disk(dilate), iterations=1)
    return edges.astype(bool)


def boundary_band(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Return a boolean band around a mask boundary."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not values.any():
        return np.zeros(values.shape, dtype=bool)
    radius = max(0, int(width))
    if radius <= 0:
        # One-pixel contour: the mask minus its own erosion.
        eroded = cv2.erode(values.astype(np.uint8), _disk(1), iterations=1)
        return values & ~eroded.astype(bool)
    dilated = cv2.dilate(values.astype(np.uint8), _disk(radius), iterations=1)
    eroded = cv2.erode(values.astype(np.uint8), _disk(radius), iterations=1)
    return np.logical_xor(dilated.astype(bool), eroded.astype(bool))


def boundary_line_support(
    mask: np.ndarray,
    lines: np.ndarray,
    *,
    radius: int = 2,
) -> float:
    """Return the fraction of a mask contour that sits on line art.

    ``1.0`` means every boundary pixel is within *radius* of an extracted line;
    ``0.0`` means none are (or the mask is empty).
    """
    values = np.asarray(mask, dtype=bool)
    line_mask = np.asarray(lines, dtype=bool)
    if values.ndim != 2 or not values.any():
        return 0.0
    edge = boundary_band(values, width=1)
    edge_count = int(np.count_nonzero(edge))
    if edge_count == 0:
        return 0.0
    if line_mask.shape != values.shape or not line_mask.any():
        return 0.0
    near_lines = cv2.dilate(line_mask.astype(np.uint8), _disk(radius), iterations=1)
    supported = int(np.count_nonzero(edge & near_lines.astype(bool)))
    return float(np.clip(supported / edge_count, 0.0, 1.0))


def background_line_penalty(
    mask: np.ndarray,
    lines: np.ndarray,
    points: Sequence[object],
    *,
    reach: int = 3,
) -> float:
    """Return the fraction of a mask made of *unprompted* line art.

    Line-art components that overlap the mask but are nowhere near a positive
    prompt point are almost certainly background lines the model wandered onto
    (a fence, a background character, a bookshelf).  The score is the share of
    the mask's area such components occupy, in ``[0, 1]``.
    """
    values = np.asarray(mask, dtype=bool)
    line_mask = np.asarray(lines, dtype=bool)
    if values.ndim != 2 or not values.any() or line_mask.shape != values.shape:
        return 0.0
    mask_area = int(np.count_nonzero(values))
    if mask_area == 0:
        return 0.0

    positives = [p for p in points if getattr(p, "label", 0) == 1]
    if not positives:
        # Nothing to anchor to; do not pretend the mask is wrong.
        return 0.0

    overlaps = values & line_mask
    if not overlaps.any():
        return 0.0

    number, labels, _stats, _ = cv2.connectedComponentsWithStats(
        line_mask.astype(np.uint8), connectivity=8
    )
    if number <= 1:
        return 0.0

    h, w = values.shape
    seeds = np.zeros((h, w), dtype=np.uint8)
    for point in positives:
        x = max(0, min(w - 1, int(getattr(point, "x", 0))))
        y = max(0, min(h - 1, int(getattr(point, "y", 0))))
        seeds[y, x] = 1
    seeds = cv2.dilate(seeds, _disk(reach), iterations=1)
    connected_ids = set(np.unique(labels[seeds.astype(bool)]).tolist())
    connected_ids.discard(0)

    if not connected_ids:
        discouraged = overlaps
    else:
        discouraged = overlaps & ~np.isin(labels, list(connected_ids))
    penalty = float(np.count_nonzero(discouraged) / mask_area)
    return float(np.clip(penalty, 0.0, 1.0))


def snap_to_lines(
    mask: np.ndarray,
    lines: np.ndarray,
    *,
    radius: int = 2,
    allow_shrink: bool = True,
) -> np.ndarray:
    """Pull a mask boundary onto the nearest line art.

    Grows the mask onto line pixels just outside the current contour, and (when
    *allow_shrink*) trims contour pixels that stray more than *radius* from any
    line.  Interior pixels are never touched, and the whole operation is skipped
    when the contour does not overlap line art — so a non-line-art frame (or a
    soft subject) is returned unchanged.
    """
    values = np.asarray(mask, dtype=bool)
    line_mask = np.asarray(lines, dtype=bool)
    if values.ndim != 2 or not values.any():
        return values.copy()
    if line_mask.shape != values.shape or not line_mask.any():
        return values.copy()

    support = boundary_line_support(values, line_mask, radius=radius)
    if support < _SNAP_SUPPORT_FLOOR:
        return values.copy()

    result = values.copy()
    contour = boundary_band(values, width=1)
    grow_zone = cv2.dilate(contour.astype(np.uint8), _disk(radius), iterations=1)
    result |= line_mask & grow_zone.astype(bool)

    if allow_shrink:
        # Distance from every pixel to the nearest line pixel.
        distance = cv2.distanceTransform(
            (~line_mask).astype(np.uint8), cv2.DIST_L2, 3
        )
        shrink_zone = grow_zone.astype(bool) & result
        strayed = shrink_zone & (distance > float(radius) + 0.5)
        result &= ~strayed
    return result


def snap_to_anime_lines(
    mask: np.ndarray,
    frame: np.ndarray,
    *,
    radius: int = 2,
    allow_shrink: bool = True,
    **extract_kwargs: object,
) -> np.ndarray:
    """Convenience wrapper: extract line art from *frame*, then snap *mask*."""
    lines = extract_lines(frame, **extract_kwargs)  # type: ignore[arg-type]
    return snap_to_lines(mask, lines, radius=radius, allow_shrink=allow_shrink)


__all__ = [
    "background_line_penalty",
    "boundary_band",
    "boundary_line_support",
    "extract_lines",
    "snap_to_anime_lines",
    "snap_to_lines",
]
