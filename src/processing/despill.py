"""Color spill suppression on the matte edge band.

A subject shot against a strongly colored background (blue sky, green screen)
picks up a fringe of that color along its silhouette.  The fringe lives in a
narrow band straddling the matte boundary, so every operation here is confined
to that band and the interior of the subject is never touched.

Two modes are offered because the two failure cases are different:

``despill``
    The color is wrong but the coverage is right - the subject really does
    extend that far, it just carries the background's hue.  Matching pixels are
    pulled toward their own luminance, which keeps the shape and the shading
    and removes only the chroma.

``alpha_key``
    The color is a symptom of partial coverage, not of a wide matte.  Matching
    pixels are faded out of the matte so the fringe becomes transparency
    instead of a colored halo.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Below this OpenCV saturation (0-255) a pixel reads as neutral gray.  A spill
#: fringe is always at least somewhat saturated, and gating on it stops gray
#: pixels from matching every possible spill color.
_MIN_SATURATION = 40

#: Saturation at which a pixel is treated as fully saturated, so ordinary
#: colored fringes score 1.0 and only genuinely washed-out pixels are held back.
_FULL_SATURATION = 128

#: Widest hue window the tolerance slider can open, in OpenCV hue units
#: (0-179, so 180 degrees).
_MAX_HUE_SPREAD = 90.0

#: Fraction of the tolerance window that scores a *full* match before the score
#: ramps down.  Without this plateau a linear falloff means a pixel has to sit
#: almost exactly on the target hue to be caught, so even at tolerance 100 most
#: of the window barely did anything.
_MATCH_PLATEAU = 0.5


def _structuring_element(radius: int) -> np.ndarray:
    """Return an elliptical kernel of the given radius."""
    size = int(radius) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


#: Gains are clamped to this range so a near-black reference channel cannot
#: blow the correction up into noise or clipping.
_MIN_CAST_GAIN = 0.6
_MAX_CAST_GAIN = 1.7


def _box(shape: tuple[int, int], region: tuple[int, int, int, int]) -> np.ndarray:
    """Return a bool mask of the rectangle, clipped to the frame."""
    area = np.zeros(shape, dtype=bool)
    height, width = shape
    x, y, box_w, box_h = region
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1 = min(width, int(x) + int(box_w))
    y1 = min(height, int(y) + int(box_h))
    if x1 > x0 and y1 > y0:
        area[y0:y1, x0:x1] = True
    return area


def _work_area(
    mask: np.ndarray,
    radius: int,
    region: tuple[int, int, int, int] | None,
    region_mode: str = "clean",
) -> np.ndarray:
    """Return the bool array of pixels the cleaner is allowed to touch.

    Without a region this is the narrow edge band, which is the safe default.
    With a region the user's rectangle replaces it: in ``clean`` mode the whole
    box is fair game, and in ``protect`` mode the box is shielded and everything
    else is. That is what makes it possible to strip a large sky patch while
    leaving a blue-haired subject on the other side of frame completely alone.
    """
    if region is None:
        return edge_band(mask, radius)
    box = _box(mask.shape[:2], region)
    if region_mode == "protect":
        return ~box
    return box


def _sample_area(
    mask: np.ndarray | None,
    region: tuple[int, int, int, int] | None,
    region_mode: str = "clean",
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the pixels treated as the neutral reference for colour casting.

    Prefers an explicit box, then the background outside the matte, because the
    background is what the scene light actually bounced off. Falls back to the
    whole frame when there is nothing else to go on.
    """
    if shape is None and mask is not None:
        shape = mask.shape[:2]
    if region is not None and shape is not None:
        # A box is self-contained: it does not need a matte to be sampled.
        # Previously a region *without* a mask fell through to an empty
        # ``(0, 0)`` area, so the cast was never corrected even though the
        # user had drawn an Area box.  ``_work_area(mask, 0, region, mode)``
        # was just ``_box(shape, region)`` with the protect/clean flip, so
        # reproducing it here keeps the with-mask behaviour identical.
        box = _box(shape, region)
        return ~box if region_mode == "protect" else box
    if mask is not None:
        background = ~mask
        if background.any():
            return background
    if shape is None:
        return np.ones((0, 0), dtype=bool)
    return np.ones(shape, dtype=bool)


def neutralize_cast(
    frame_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    region: tuple[int, int, int, int] | None = None,
    region_mode: str = "clean",
    strength: int = 100,
) -> np.ndarray:
    """Pull a global colour cast out of a shot.

    A strongly tinted light (a violet sky, a green screen) does not just fringe
    the matte, it tints the subject too, and no amount of edge work can recover
    a blonde read as blue. This measures the cast from a neutral reference area
    and applies a per-channel gain that returns it to neutral, which fixes the
    subject because both were lit by the same light.

    Only a *global* correction: per-channel gains, overall brightness preserved,
    so real shading survives. It deliberately does not touch local colour.

    Args:
        frame_rgb: Source frame (H, W, 3) uint8 RGB.
        mask: Binary matte; its outside is used as the reference when no region
            is given.
        region: Optional ``(x, y, w, h)`` reference box, typically drawn around
            the backdrop.
        region_mode: ``"clean"`` or ``"protect"`` for *region*.
        strength: 0-100. 0 is a no-op, 100 fully neutralizes the reference.

    Returns:
        A new uint8 RGB frame. The input is never modified.
    """
    if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        return frame_rgb
    try:
        amount = max(0, min(100, int(strength))) / 100.0
    except (TypeError, ValueError):
        amount = 1.0
    if amount <= 0.0:
        return frame_rgb

    source = frame_rgb[:, :, :3]
    area = _sample_area(mask, region, region_mode, source.shape[:2])
    if not area.any():
        return frame_rgb

    values = source.astype(np.float32)
    means = values[area].mean(axis=0)
    overall = float(means.mean())
    if not np.isfinite(overall) or overall < 1.0:
        return frame_rgb

    gain = np.clip(overall / np.maximum(means, 1.0), _MIN_CAST_GAIN, _MAX_CAST_GAIN)
    gain = 1.0 + (gain - 1.0) * amount
    out = frame_rgb.copy()
    out[:, :, :3] = np.clip(values * gain, 0.0, 255.0).astype(np.uint8)
    return out


def edge_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return the ring of pixels straddling the matte boundary.

    Args:
        mask: Binary mask (H, W) bool.
        radius: Band half-width in pixels (1-10).

    Returns:
        Bool mask that is ``True`` inside the morphological gradient
        ``dilate(mask) - erode(mask)``. Empty masks give an empty band.
    """
    if mask is None:
        return np.zeros((0, 0), dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    radius = max(1, min(10, int(radius)))
    source = mask.astype(np.uint8)
    kernel = _structuring_element(radius)
    return (cv2.dilate(source, kernel) - cv2.erode(source, kernel)).astype(bool)


def spill_match(
    frame_rgb: np.ndarray,
    spill_color: tuple[int, int, int],
    tolerance: int = 30,
) -> np.ndarray:
    """Return how strongly each pixel matches *spill_color*, in ``[0, 1]``.

    Matching is done on hue, which is circular, so blue and red are neighbours
    across the 0/180 wrap the way they are perceptually.  A pixel's score falls
    off linearly with hue distance and is scaled by how saturated it is, so a
    nearly gray pixel scores close to zero no matter which color was picked.

    Args:
        frame_rgb: Source frame (H, W, 3) uint8 RGB.
        spill_color: Target RGB triple.
        tolerance: 0-100; 0 matches only an exact hue, 100 opens the window to
            90 degrees either side.

    Returns:
        Float32 (H, W) match scores.
    """
    if frame_rgb is None or spill_color is None:
        return np.zeros((0, 0), dtype=np.float32)
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        return np.zeros(frame_rgb.shape[:2], dtype=np.float32)

    red, green, blue = (int(channel) for channel in spill_color)
    tolerance = max(0, min(100, int(tolerance)))

    hsv = cv2.cvtColor(frame_rgb[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.float32)
    target_hue = float(
        cv2.cvtColor(
            np.array([[[red, green, blue]]], dtype=np.uint8), cv2.COLOR_RGB2HSV
        )[0, 0, 0]
    )

    # Circular distance in OpenCV hue units (period 180).
    delta = np.abs(hue - target_hue)
    distance = np.minimum(delta, 180 - delta).astype(np.float32)
    if tolerance <= 0:
        spread = 0.5
    else:
        spread = max(0.5, _MAX_HUE_SPREAD * tolerance / 100.0)
    # Plateau then ramp: everything inside the inner half of the window counts
    # fully, which is what makes a high tolerance actually clean a fringe.
    score = np.clip((spread - distance) / (spread * (1.0 - _MATCH_PLATEAU)), 0.0, 1.0)

    # A gray pixel is not evidence of spill, however close its hue.
    score *= np.clip(
        (sat - _MIN_SATURATION) / (_FULL_SATURATION - _MIN_SATURATION), 0.0, 1.0
    )
    return score.astype(np.float32)


def despill_frame(
    frame_rgb: np.ndarray,
    mask: np.ndarray | None,
    spill_color: tuple[int, int, int],
    tolerance: int = 30,
    radius: int = 2,
    region: tuple[int, int, int, int] | None = None,
    region_mode: str = "clean",
) -> np.ndarray:
    """Neutralize spill-colored pixels along the matte edge.

    Matching pixels are blended toward their own luminance, so the fringe keeps
    its brightness and shading and loses only its color. Without a *region* the
    work is confined to the edge band, which keeps skin tones, costume and props
    of the same hue intact. With a *region* the user's rectangle is used
    instead, so a big sky patch can be cleared without touching matching colors
    elsewhere in frame.

    Args:
        frame_rgb: Source frame (H, W, 3) uint8 RGB.
        mask: Binary matte used to locate the edge band; ``None`` disables the
            band and returns the frame unchanged unless a region is given.
        spill_color: Target RGB triple to suppress.
        tolerance: Hue tolerance, 0-100.
        radius: Band half-width in pixels (1-10). Ignored when *region* is set.
        region: Optional ``(x, y, w, h)`` box in image coordinates limiting
            where the cleaner may act.
        region_mode: ``"clean"`` to work inside *region*, ``"protect"`` to
            work everywhere except it.

    Returns:
        A new uint8 RGB frame. The input is never modified.
    """
    if frame_rgb is None or spill_color is None:
        return frame_rgb
    if mask is not None:
        area = _work_area(mask, radius, region, region_mode)
    elif region is not None:
        area = _work_area(
            np.zeros(frame_rgb.shape[:2], dtype=bool), radius, region, region_mode
        )
    else:
        area = None
    if area is None or not area.any():
        return frame_rgb

    score = spill_match(frame_rgb, spill_color, tolerance)
    weight = score * area.astype(np.float32)
    if not weight.any():
        return frame_rgb

    source = frame_rgb[:, :, :3].astype(np.float32)
    # Rec. 601 luma: the achromatic version of the pixel, brightness preserved.
    luma = source @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    neutral = np.repeat(luma[:, :, None], 3, axis=2)
    blended = source * (1.0 - weight[:, :, None]) + neutral * weight[:, :, None]
    out = frame_rgb.copy()
    out[:, :, :3] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return out


def despill_alpha(
    mask: np.ndarray,
    frame_rgb: np.ndarray | None,
    spill_color: tuple[int, int, int],
    tolerance: int = 30,
    radius: int = 2,
    region: tuple[int, int, int, int] | None = None,
    region_mode: str = "clean",
) -> np.ndarray:
    """Fade spill-colored pixels out of the matte along its edge.

    Coverage is attenuated in proportion to the color match and the result is
    thresholded back to a bool matte, because every consumer in the app (canvas
    overlay, PNG sequence, RGBA composite) works on binary mattes. A pixel is
    dropped when the match is stronger than not, so a wide tolerance clears a
    wide fringe rather than only its exact center line.

    Args:
        mask: Binary matte (H, W) bool.
        frame_rgb: Frame the matte belongs to, used to measure the color.
        spill_color: Target RGB triple to suppress.
        tolerance: Hue tolerance, 0-100.
        radius: Band half-width in pixels (1-10). Ignored when *region* is set.
        region: Optional ``(x, y, w, h)`` box in image coordinates limiting
            where the cleaner may act.
        region_mode: ``"clean"`` to key inside *region*, ``"protect"`` to key
            everywhere except it.

    Returns:
        A new bool mask. The input is never modified.
    """
    if mask is None or frame_rgb is None or spill_color is None:
        return mask
    if not mask.any():
        return mask

    area = _work_area(mask, radius, region, region_mode)
    if not area.any():
        return mask

    score = spill_match(frame_rgb, spill_color, tolerance)
    coverage = mask.astype(np.float32)
    coverage[area] = np.minimum(coverage[area], 1.0 - score[area])
    return coverage >= 0.5


def apply_shot_transform(
    frame_rgb: np.ndarray | None,
    mask: np.ndarray | None,
    *,
    mask_inset: int = 0,
    spill_mode: str = "off",
    spill_color: tuple[int, int, int] | None = None,
    spill_tolerance: int = 30,
    spill_radius: int = 2,
    spill_region: tuple[int, int, int, int] | None = None,
    spill_region_mode: str = "clean",
    cast_neutralize: bool = False,
    cast_strength: int = 100,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Apply the per-shot view transforms to one frame and its matte.

    Every export path funnels through here so the rendered result always
    matches what the canvas previewed. Masks are cached un-transformed so the
    controls stay adjustable, so both inputs are treated as read-only.

    The cast correction runs first on purpose: once the global tint is gone the
    fringe is much closer to plain background color, so despill has far less to
    do and reaches a cleaner result at the same tolerance.

    Args:
        frame_rgb: Source frame, or ``None`` when only the matte is exported.
        mask: Binary matte, or ``None``.
        mask_inset: Matte erosion in pixels.
        spill_mode: ``"off"``, ``"despill"`` or ``"alpha_key"``.
        spill_color: Target RGB triple; ``None`` disables the cleaner.
        spill_tolerance: Hue tolerance, 0-100.
        spill_radius: Band half-width in pixels (1-10).
        spill_region: Optional ``(x, y, w, h)`` box limiting where the cleaner
            may act; ``None`` means the matte edge band.
        spill_region_mode: ``"clean"`` or ``"protect"`` for *spill_region*.
        cast_neutralize: Whether to remove a global colour cast first.
        cast_strength: Cast correction strength, 0-100.

    Returns:
        The transformed ``(frame, mask)``; each is returned unchanged when the
        corresponding transform does not apply.
    """
    from src.processing.mask_refine import erode_mask

    if mask is None:
        return frame_rgb, None
    result = erode_mask(mask, mask_inset)
    if frame_rgb is None:
        return frame_rgb, result
    if cast_neutralize:
        frame_rgb = neutralize_cast(
            frame_rgb, mask, spill_region, spill_region_mode, cast_strength
        )
    if spill_mode not in ("despill", "alpha_key") or spill_color is None:
        return frame_rgb, result
    if not result.any():
        return frame_rgb, result
    if spill_mode == "alpha_key":
        return frame_rgb, despill_alpha(
            result,
            frame_rgb,
            spill_color,
            spill_tolerance,
            spill_radius,
            spill_region,
            spill_region_mode,
        )
    return (
        despill_frame(
            frame_rgb,
            result,
            spill_color,
            spill_tolerance,
            spill_radius,
            spill_region,
            spill_region_mode,
        ),
        result,
    )
