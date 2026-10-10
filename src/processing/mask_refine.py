"""Guided-filter based mask refinement for anime content."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def erode_mask(mask: np.ndarray, inset: int) -> np.ndarray:
    """Shrink a binary mask by ``inset`` pixels to trim background bleed.

    Morphological erosion pulls the matte contour inward by ``inset`` pixels,
    which removes the fringe of background color that anti-aliased source
    frames leave around a subject. The elliptical structuring element gives a
    rounder contour than a rectangular one, so corners do not pinch.

    Args:
        mask: Binary mask (H, W) bool.
        inset: Shrink radius in pixels (0–10). 0 returns the mask unchanged.

    Returns:
        Eroded bool mask. Thin features thinner than ``2 * inset + 1`` pixels
        are removed entirely - lower ``inset`` for hair and line art.
    """
    if mask is None or inset is None:
        return mask
    try:
        radius = int(inset)
    except (TypeError, ValueError):
        return mask
    radius = max(0, min(10, radius))
    if radius <= 0 or not mask.any():
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def refine_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    eps: float = 1e-2,
    min_area: int = 0,
    smooth_kernel: int = 1,
    smooth_strength: float | None = None,
    detail_preservation: float = 0.75,
    mask_inset: int = 0,
) -> np.ndarray:
    """Refine a binary mask without destroying thin line-art details.

    The guided filter is used as a conservative edge-snapping proposal. A
    narrow band around the original contour is the only region that can grow
    or shrink, and pixels close to a contour are kept when they form a thin
    feature. This is important for anime hair, where an opening operation can
    erase a strand even when the underlying SAM mask is correct.

    Args:
        frame_rgb: RGB uint8 frame (H, W, 3).
        mask: Binary mask (H, W) bool.
        radius: Guided filter neighborhood radius (larger = smoother).
        eps: Regularization (higher = smoother, lower = sharper edges).
        min_area: Minimum area for newly-created components. Components that
            already exist in the source mask are retained.
        smooth_kernel: Retained for API compatibility; boundary morphology is
            intentionally not used because it removes anime hair strands.
        smooth_strength: Master strength (0–10). When provided it overrides
            ``radius``, ``eps`` and ``smooth_kernel`` so a single slider can
            control the result. 0 = raw mask (no refinement).
        detail_preservation: Fraction of fine boundary structure to protect
            from removal. Higher values favor hair and line art.
        mask_inset: Shrink radius in pixels (0-10) applied as a final
            morphological erosion to trim background color bleed.

    Returns:
        Refined bool mask. If edge refinement would be catastrophic (area
        collapse < 50 % or explosion > 180 %) the original mask is used for
        that stage, but ``mask_inset`` is always honored.
    """
    if mask is None:
        return mask
    refined = _guided_refine(
        frame_rgb, mask, radius, eps, min_area, smooth_strength, detail_preservation
    )
    return erode_mask(refined, mask_inset)


def _guided_refine(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    radius: int,
    eps: float,
    min_area: int,
    smooth_strength: float | None,
    detail_preservation: float,
) -> np.ndarray:
    """Run the guided-filter edge stage of :func:`refine_mask`."""
    if frame_rgb is None or mask is None or not mask.any():
        return mask

    h, w = mask.shape
    if frame_rgb.shape[:2] != (h, w):
        return mask

    try:
        preservation = max(0.0, min(1.0, float(detail_preservation)))
    except (TypeError, ValueError):
        preservation = 0.75
    strength = None
    if smooth_strength is not None:
        strength = max(0.0, min(10.0, float(smooth_strength)))
        if strength <= 0:
            return mask
        radius = 2 + min(2, int(strength / 4))
        eps = 1e-3 + strength * 5e-4

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = mask.astype(np.float64)
    q = _guided_filter(gray, p, radius, eps)

    if strength is None:
        remove_threshold = 0.25
        add_threshold = 0.55
    else:
        remove_threshold = 0.30 - strength * 0.005
        add_threshold = 0.50 + strength * 0.005

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    source = mask.astype(np.uint8)
    dilated = cv2.dilate(source, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(source, kernel, iterations=1).astype(bool)
    boundary = dilated ^ eroded
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 3)
    thin_features = mask & (distance <= 1.0 + preservation * 2.0)
    thin_features = (
        cv2.dilate(
            thin_features.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(bool)
        & mask
    )

    gray_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gray_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.magnitude(gray_x, gray_y)
    edge_support = cv2.dilate(
        (edge_strength > 0.04).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)

    refined = mask.copy()
    remove = mask & (q <= remove_threshold) & boundary & ~thin_features
    add = ~mask & (q > add_threshold) & boundary & edge_support
    refined[remove] = False
    refined[add] = True

    if min_area > 0:
        refined = _remove_small_components(refined, min_area, protected=mask)

    orig_area = int(mask.sum())
    new_area = int(refined.sum())
    if orig_area > 0:
        ratio = new_area / orig_area
        if ratio < 0.5 or ratio > 1.8:
            logger.debug(
                "refine_mask: area ratio %.2f outside safe range, returning original",
                ratio,
            )
            return mask

    return refined


# ------------------------------------------------------------------ #
# Guided filter internals (manual – cv2.ximgproc not available)       #
# ------------------------------------------------------------------ #
def _box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    """Box (mean) filter via integral image."""
    h, w = src.shape
    win = 2 * radius + 1
    im = np.pad(src, radius, mode="edge").astype(np.float64)
    ih, iw = im.shape
    S = np.zeros((ih + 1, iw + 1), dtype=np.float64)
    np.cumsum(np.cumsum(im, axis=0), axis=1, out=S[1:, 1:])

    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    i0, i1 = y, y + 2 * radius
    j0, j1 = x, x + 2 * radius
    out = S[i1 + 1, j1 + 1] - S[i0, j1 + 1] - S[i1 + 1, j0] + S[i0, j0]
    out /= float(win * win)
    return out


def _guided_filter(
    I: np.ndarray, p: np.ndarray, radius: int = 3, eps: float = 1e-2
) -> np.ndarray:
    """Edge-aware guided filter (He et al., 2013)."""
    mean_I = _box_filter(I, radius)
    mean_p = _box_filter(p, radius)
    corr_I = _box_filter(I * I, radius)
    corr_Ip = _box_filter(I * p, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    return mean_a * I + mean_b


def _remove_small_components(
    mask: np.ndarray, min_area: int, protected: np.ndarray | None = None
) -> np.ndarray:
    """Remove small components while retaining source-mask components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_labels):
        component = labels == i
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[component] = True
        elif protected is None or not np.any(component & protected):
            continue
        else:
            result[component] = True
    return result
