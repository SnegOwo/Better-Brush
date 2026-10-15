"""Letterbox / pillarbox detection for anime & AMV sources.

Old anime releases — the mainstay of AMV editing — are routinely stored with
solid black bars baked into the frame: 4:3 masters pillarboxed into 16:9,
letterboxed film prints, or a combination of both.  Those bars are pure
background, but a segmenter has no idea: they are dark, static and huge, so
they are easy to accidentally include in a matte and they waste the GPU on
pixels nobody wants.

This module finds the bars from the image itself and returns a crop box.  It is
deliberately conservative: a frame that is uniformly dark (a night scene, a
fade) is *not* a letterbox, and the crop is capped so it can never eat the
content.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

#: A row/column is a bar when its high percentile luminance is at or below this.
_DEFAULT_THRESHOLD = 16.0

#: Percentile of a row/column used as its "brightest representative" value.  The
#: 95th percentile ignores isolated specks (dust, a stray bright pixel in the
#: bar) that a plain max would trip over, while still tracking real content.
_DEFAULT_PERCENTILE = 95.0

#: Never crop more than this fraction off any single edge, so a genuinely dark
#: frame can at worst lose a sliver rather than half the picture.
_MAX_EDGE_FRACTION = 0.45


@dataclass(frozen=True)
class Letterbox:
    """Black bars baked into a frame, in pixels, measured from each edge."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    @property
    def active(self) -> bool:
        """Return whether any bar was detected."""
        return bool(self.top or self.bottom or self.left or self.right)

    @property
    def horizontal(self) -> int:
        """Total bar height (top + bottom)."""
        return int(self.top + self.bottom)

    @property
    def vertical(self) -> int:
        """Total bar width (left + right)."""
        return int(self.left + self.right)

    @property
    def is_letterbox(self) -> bool:
        """Return whether there are top/bottom bars."""
        return bool(self.top or self.bottom)

    @property
    def is_pillarbox(self) -> bool:
        """Return whether there are left/right bars."""
        return bool(self.left or self.right)

    def box(self, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return the content region as an ``(x0, y0, x1, y1)`` exclusive box."""
        height = max(0, int(shape[0]))
        width = max(0, int(shape[1]))
        x0 = max(0, min(width, int(self.left)))
        y0 = max(0, min(height, int(self.top)))
        x1 = max(x0, width - max(0, int(self.right)))
        y1 = max(y0, height - max(0, int(self.bottom)))
        return (x0, y0, x1, y1)

    def content_shape(self, shape: tuple[int, int]) -> tuple[int, int]:
        """Return the ``(height, width)`` of the content region."""
        x0, y0, x1, y1 = self.box(shape)
        return (max(0, y1 - y0), max(0, x1 - x0))

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly description."""
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "active": self.active,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }


def _luminance(frame: np.ndarray) -> np.ndarray:
    """Return a float32 luma plane for an RGB/RGBA/gray frame."""
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[2] < 3:
        return np.zeros(array.shape[:2], dtype=np.float32)
    rgb = array[..., :3].astype(np.float32, copy=False)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _sorted_dimensions(profile: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean array: which lines (rows or cols) are bar-like."""
    return profile <= float(threshold)


def _leading_run(flags: np.ndarray, limit: int) -> int:
    """Count the consecutive True entries at the start of *flags*, up to *limit*."""
    count = 0
    for value in flags[:limit]:
        if bool(value):
            count += 1
        else:
            break
    return count


def _trailing_run(flags: np.ndarray, limit: int) -> int:
    """Count the consecutive True entries at the end of *flags*, up to *limit*."""
    count = 0
    for value in flags[::-1][:limit]:
        if bool(value):
            count += 1
        else:
            break
    return count


def detect_letterbox(
    frame: np.ndarray,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    percentile: float = _DEFAULT_PERCENTILE,
    min_bar: int = 2,
    max_edge_fraction: float = _MAX_EDGE_FRACTION,
) -> Letterbox:
    """Detect baked-in black bars in a single frame.

    Args:
        frame: RGB (or gray) image; any numeric dtype.
        threshold: Luma at or below which a row/column counts as bar.
        percentile: How bright a row/column must stay to count as content.
        min_bar: Minimum run length (pixels) to report a bar; filters noise.
        max_edge_fraction: Cap on the fraction of an edge that may be cropped.

    Returns:
        A :class:`Letterbox`.  A frame with no bars, a uniform frame, or a
        degenerate crop returns an all-zero box.
    """
    luma = _luminance(frame)
    height, width = luma.shape
    if height < 3 or width < 3:
        return Letterbox()

    percent = float(np.clip(percentile, 0.0, 100.0))
    # A uniform (all-black / all-white) frame has no separable content region, so
    # there is nothing to crop around; refuse rather than crop everything.
    if float(np.ptp(luma)) <= 1e-6:
        return Letterbox()

    row_profile = np.percentile(luma, percent, axis=1)
    col_profile = np.percentile(luma, percent, axis=0)
    row_bar = _sorted_dimensions(row_profile, threshold)
    col_bar = _sorted_dimensions(col_profile, threshold)

    max_vert = max(0, int(height * float(max_edge_fraction)))
    max_horz = max(0, int(width * float(max_edge_fraction)))

    top = _leading_run(row_bar, max_vert)
    bottom = _trailing_run(row_bar, max_vert)
    left = _leading_run(col_bar, max_horz)
    right = _trailing_run(col_bar, max_horz)

    # Never let top+bottom (or left+right) consume the whole frame.
    if top + bottom >= height:
        top = bottom = 0
    if left + right >= width:
        left = right = 0

    if top < min_bar:
        top = 0
    if bottom < min_bar:
        bottom = 0
    if left < min_bar:
        left = 0
    if right < min_bar:
        right = 0

    return Letterbox(top=top, bottom=bottom, left=left, right=right)


def combine_letterboxes(boxes: Sequence[Letterbox]) -> Letterbox:
    """Return the conservative (elementwise-minimum) box across frames.

    Bars are constant within a source, so the safest summary is the smallest
    bar any frame needed: that never crops content that a single frame kept.
    """
    if not boxes:
        return Letterbox()
    return Letterbox(
        top=min(box.top for box in boxes),
        bottom=min(box.bottom for box in boxes),
        left=min(box.left for box in boxes),
        right=min(box.right for box in boxes),
    )


def detect_letterbox_sequence(
    frames: Iterable[np.ndarray],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    percentile: float = _DEFAULT_PERCENTILE,
    min_bar: int = 2,
) -> Letterbox:
    """Detect bars across several frames and return a conservative box."""
    boxes = [
        detect_letterbox(
            frame,
            threshold=threshold,
            percentile=percentile,
            min_bar=min_bar,
        )
        for frame in frames
    ]
    return combine_letterboxes(boxes)


def crop_frame(frame: np.ndarray, box: Letterbox) -> np.ndarray:
    """Return the content region of *frame* as a view (no copy)."""
    x0, y0, x1, y1 = box.box(frame.shape[:2])
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1]


def crop_mask(mask: np.ndarray, box: Letterbox) -> np.ndarray:
    """Return the content region of a mask as a view (no copy)."""
    x0, y0, x1, y1 = box.box(mask.shape[:2])
    if x1 <= x0 or y1 <= y0:
        return mask
    return mask[y0:y1, x0:x1]


def embed_mask(
    mask: np.ndarray,
    box: Letterbox,
    shape: tuple[int, int],
) -> np.ndarray:
    """Place a cropped mask back into a full-frame canvas (bars stay empty)."""
    full = np.zeros((max(0, int(shape[0])), max(0, int(shape[1]))), dtype=bool)
    x0, y0, x1, y1 = box.box(shape)
    if x1 <= x0 or y1 <= y0:
        return full
    region = np.asarray(mask, dtype=bool)[: y1 - y0, : x1 - x0]
    full[y0 : y0 + region.shape[0], x0 : x0 + region.shape[1]] = region
    return full


__all__ = [
    "Letterbox",
    "combine_letterboxes",
    "crop_frame",
    "crop_mask",
    "detect_letterbox",
    "detect_letterbox_sequence",
    "embed_mask",
]
