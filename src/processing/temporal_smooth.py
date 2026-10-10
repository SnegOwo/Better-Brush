"""Temporal smoothing for mask sequences using optical flow."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def temporal_smooth_masks(
    frames: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
    window: int = 1,
    on_progress: Callable[[str, int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[int, np.ndarray]:
    """Apply temporal smoothing to a mask sequence using optical flow.

    For each frame the neighbours within *window* are warped into the
    current frame's coordinates via Farneback optical flow.  A majority
    vote is performed only near mask contours (edge band) so that
    interiors stay untouched while flicker at the boundaries is reduced.
    """
    if not masks:
        return dict(masks)

    sorted_idx = sorted(masks.keys())
    if len(sorted_idx) < 2:
        return dict(masks)

    gray_cache: dict[int, np.ndarray] = {}

    def _get_gray(idx: int) -> np.ndarray | None:
        if idx in gray_cache:
            return gray_cache[idx]
        if idx not in frames:
            return None
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        gray_cache[idx] = gray
        return gray

    result: dict[int, np.ndarray] = {}
    total = len(sorted_idx)

    for i, idx in enumerate(sorted_idx):
        if cancel_event is not None and cancel_event.is_set():
            break

        if on_progress is not None:
            on_progress("Smoothing masks", i + 1, total)

        cur_mask = masks[idx]
        cur_gray = _get_gray(idx)
        if cur_mask is None or cur_gray is None:
            result[idx] = cur_mask
            continue

        votes = cur_mask.astype(np.float64).copy()
        vote_count = np.ones(cur_mask.shape, dtype=np.float64)

        for offset in range(1, window + 1):
            for neighbor_idx in (idx - offset, idx + offset):
                if neighbor_idx not in masks or neighbor_idx not in frames:
                    continue

                neighbor_gray = _get_gray(neighbor_idx)
                if neighbor_gray is None:
                    continue

                if neighbor_idx < idx:
                    flow = _compute_flow(neighbor_gray, cur_gray)
                    warped = _warp_mask(masks[neighbor_idx], flow)
                else:
                    flow = _compute_flow(cur_gray, neighbor_gray)
                    inv_flow = _invert_flow(flow)
                    warped = _warp_mask(masks[neighbor_idx], inv_flow)

                votes += warped.astype(np.float64)
                vote_count += 1.0

        smoothed = (votes / vote_count) >= 0.5
        edge = _edge_band(cur_mask, width=1)
        detail = _thin_features(cur_mask)
        detail_zone = cv2.dilate(
            detail.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(bool)
        final = np.where(edge & ~detail_zone, smoothed, cur_mask)
        result[idx] = final.astype(bool)

    return result


# ------------------------------------------------------------------ #
# Optical flow helpers                                                 #
# ------------------------------------------------------------------ #
def _compute_flow(
    gray1: np.ndarray, gray2: np.ndarray, downscale: int = 2
) -> np.ndarray:
    if downscale > 1:
        h, w = gray1.shape
        s1 = cv2.resize(gray1, (w // downscale, h // downscale))
        s2 = cv2.resize(gray2, (w // downscale, h // downscale))
    else:
        s1, s2 = gray1, gray2

    flow_s = cv2.calcOpticalFlowFarneback(
        s1, s2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    if downscale > 1:
        flow = cv2.resize(flow_s, (gray1.shape[1], gray1.shape[0]))
        flow[..., 0] *= downscale
        flow[..., 1] *= downscale
        return flow
    return flow_s


def _warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]
    warped = cv2.remap(
        mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped.astype(bool)


def _invert_flow(flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    neg = -flow
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = x + neg[..., 0]
    map_y = y + neg[..., 1]
    return cv2.remap(neg, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0)


def _edge_band(mask: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 0:
        return np.zeros_like(mask, dtype=bool)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1)
    )
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return dilated ^ eroded


def _thin_features(mask: np.ndarray) -> np.ndarray:
    source = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(
        source,
        cv2.CV_16U,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 3)
    return mask & (
        (neighbors <= 4) | ((neighbors == 6) & (distance <= 1.5))
    )
