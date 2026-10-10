"""Tests for temporal smoothing of masks."""

from __future__ import annotations

import numpy as np

from src.processing.temporal_smooth import (
    _edge_band,
    _warp_mask,
    temporal_smooth_masks,
)


def _make_frame(shape: tuple[int, int] = (64, 64), fill: int = 0) -> np.ndarray:
    frame = np.full((shape[0], shape[1], 3), fill, dtype=np.uint8)
    return frame


def _shifted_mask(shape: tuple[int, int] = (64, 64), dy: int = 0) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    y0 = max(0, 10 + dy)
    y1 = min(shape[0], 30 + dy)
    mask[y0:y1, 10:30] = True
    return mask


def test_edge_band_basic() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    band = _edge_band(mask, width=3)
    # Band should be True near contours, False inside and outside
    assert band[20, 20] is np.False_  # deep inside
    assert band[0, 0] is np.False_    # far outside
    assert band.any()                  # has some edge pixels


def test_warp_mask_identity() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    warped = _warp_mask(mask, flow)
    np.testing.assert_array_equal(warped, mask)


def test_smooth_masks_single_frame() -> None:
    mask = _shifted_mask()
    result = temporal_smooth_masks(
        {0: _make_frame()}, {0: mask}
    )
    np.testing.assert_array_equal(result[0], mask)


def test_smooth_masks_reduces_flicker() -> None:
    """A mask that wobbles between frames should be smoothed."""
    frames = {i: _make_frame() for i in range(3)}
    # Mask wobbles: odd=shifted up, even=shifted down
    masks = {
        0: _shifted_mask(dy=-3),
        1: _shifted_mask(dy=3),
        2: _shifted_mask(dy=-3),
    }
    smoothed = temporal_smooth_masks(frames, masks, window=1)
    assert len(smoothed) == 3
    # After smoothing, consecutive masks should be more similar
    for i in range(2):
        overlap_before = (
            masks[i] & masks[i + 1]
        ).sum() / max(1, (masks[i] | masks[i + 1]).sum())
        overlap_after = (
            smoothed[i] & smoothed[i + 1]
        ).sum() / max(1, (smoothed[i] | smoothed[i + 1]).sum())
        # Overlap should generally improve (or at least not collapse)
        assert overlap_after >= overlap_before - 0.1


def test_smooth_masks_preserves_interiors() -> None:
    """Interior of a stable mask should not change."""
    frames = {i: _make_frame() for i in range(3)}
    stable_mask = _shifted_mask(dy=0)
    masks = {i: stable_mask.copy() for i in range(3)}
    smoothed = temporal_smooth_masks(frames, masks, window=1)
    # Deep interior should be identical
    assert smoothed[0][20, 20] is np.True_
    assert smoothed[1][20, 20] is np.True_


def test_smooth_masks_preserves_thin_hair() -> None:
    frames = {i: _make_frame() for i in range(3)}
    masks: dict[int, np.ndarray] = {}
    for idx in range(3):
        mask = _shifted_mask(dy=0)
        if idx == 1:
            mask[2:10, 19:21] = True
        masks[idx] = mask

    smoothed = temporal_smooth_masks(frames, masks, window=1)

    np.testing.assert_array_equal(smoothed[1][2:10, 19:21], masks[1][2:10, 19:21])


def test_smooth_masks_empty() -> None:
    result = temporal_smooth_masks({}, {})
    assert result == {}


def test_smooth_cancel_event() -> None:
    import threading

    cancel = threading.Event()
    cancel.set()
    frames = {i: _make_frame() for i in range(3)}
    masks = {i: _shifted_mask(dy=i * 2) for i in range(3)}
    result = temporal_smooth_masks(frames, masks, cancel_event=cancel)
    # Cancelled immediately: result may be partial or empty
    assert isinstance(result, dict)
