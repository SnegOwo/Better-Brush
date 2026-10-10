"""Soft matte coverage and edge colour decontamination.

A binary matte is the root cause of two defects that no amount of despilling
can repair:

**Stair-stepped edges.**  With alpha restricted to 0 or 1, a one-pixel hair
strand is either fully opaque or fully gone, so the silhouette is a staircase.
Compositors get no partial coverage to anti-alias against, and the result reads
as "not smooth" no matter how good the mask is.

**Colour fringe.**  A boundary pixel is a blend of subject and background, so it
carries background colour at full strength even where it is nearly opaque.
Ship that as-is and every hair tip wears a halo of whatever was behind it.

Both are fixed together here, by turning the hard matte into a real coverage
ramp and then removing the background's contribution from the ramp.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Coverage below this is treated as fully transparent, above as fully opaque.
_PARTIAL_LO = 0.05
_PARTIAL_HI = 0.95


def soft_coverage(mask: np.ndarray, feather: float = 1.0) -> np.ndarray:
    """Turn a binary matte into an anti-aliased coverage ramp.

    Coverage is derived from the signed distance to the matte boundary, so the
    transition is centred exactly on the contour and its width is controlled in
    pixels.  This is a far better edge than blurring the mask, which shifts the
    silhouette and rounds off thin detail.

    Args:
        mask: Binary matte (H, W) bool.
        feather: Ramp width in pixels. 0 or less returns the hard matte.

    Returns:
        Float32 (H, W) coverage in ``[0, 1]``.
    """
    if mask is None:
        return np.zeros((0, 0), dtype=np.float32)
    feather = float(feather)
    if feather <= 0.0:
        return mask.astype(np.float32)
    source = mask.astype(np.uint8)
    if not source.any() or source.all():
        return mask.astype(np.float32)
    # distanceTransform measures to the nearest *opposite* pixel centre, so the
    # real boundary sits half a pixel further out; subtracting 0.5 puts the
    # 50% crossing exactly on the contour.  The sign has to be chosen per side -
    # combining the two fields directly is wrong and erases thin structures.
    inside = cv2.distanceTransform(source, cv2.DIST_L2, 3) - 0.5
    outside = cv2.distanceTransform(1 - source, cv2.DIST_L2, 3) - 0.5
    signed = np.where(mask, inside, -outside)
    return np.clip(0.5 + signed / feather, 0.0, 1.0).astype(np.float32)


def estimate_background(
    frame_rgb: np.ndarray, mask: np.ndarray | None
) -> np.ndarray | None:
    """Return the median background colour, or ``None`` if unknowable.

    The fully-transparent region is the best available estimate of what the
    subject was photographed against.
    """
    if frame_rgb is None or mask is None or not mask.any():
        return None
    outside = frame_rgb[~mask][:, :3]
    if outside.size < 96:
        return None
    return np.median(outside.astype(np.float32), axis=0)


def decontaminate_edges(
    frame_rgb: np.ndarray,
    mask: np.ndarray | None,
    alpha: np.ndarray,
) -> np.ndarray:
    """Remove background colour from partially covered edge pixels.

    A partially covered pixel is a composite::

        observed = alpha * subject + (1 - alpha) * background

    so the subject colour can be recovered by subtracting the known background
    contribution and dividing by the coverage.  The result is that a hair tip
    keeps the hair's own colour instead of a washed-out blend of hair and sky -
    which is what removes the fringe without leaving a grey halo behind.

    Fully transparent pixels are untouched, and fully opaque ones need no
    correction, so the interior of the subject is never altered.

    Args:
        frame_rgb: Source frame (H, W, 3) uint8 RGB.
        mask: Binary matte the coverage was derived from.
        alpha: Coverage ramp from :func:`soft_coverage`.

    Returns:
        A new uint8 RGB frame. The input is never modified.
    """
    if frame_rgb is None or alpha is None:
        return frame_rgb
    partial = (alpha > _PARTIAL_LO) & (alpha < _PARTIAL_HI)
    if not partial.any():
        return frame_rgb
    background = estimate_background(frame_rgb, mask)
    if background is None:
        return frame_rgb

    out = frame_rgb.copy()
    coverage = alpha[partial][:, None]
    observed = frame_rgb[:, :, :3].astype(np.float32)[partial]
    # alpha is small here, so the division is the delicate part: clamp the
    # divisor and let the coverage ramp do the rest rather than blowing up.
    recovered = (observed - (1.0 - coverage) * background) / np.maximum(
        coverage, _PARTIAL_LO
    )
    out[:, :, :3][partial] = np.clip(recovered, 0.0, 255.0).astype(np.uint8)
    return out


def finish_edges(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    feather: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the whole edge treatment: coverage ramp, then decontamination.

    Args:
        frame_rgb: Source frame (H, W, 3) uint8 RGB.
        mask: Binary matte (H, W) bool.
        feather: Coverage ramp width in pixels; 0 keeps the hard matte.

    Returns:
        The decontaminated ``(frame, alpha)`` pair.
    """
    alpha = soft_coverage(mask, feather)
    if feather <= 0.0:
        return frame_rgb, alpha
    return decontaminate_edges(frame_rgb, mask, alpha), alpha
