"""Tests for guided-filter mask refinement."""

from __future__ import annotations

import numpy as np

from src.processing.mask_refine import (
    _box_filter,
    _guided_filter,
    _remove_small_components,
    erode_mask,
    refine_mask,
)


def test_box_filter_identity() -> None:
    """Box filter of a constant image should return the same constant."""
    img = np.full((64, 64), 0.5, dtype=np.float64)
    result = _box_filter(img, radius=3)
    np.testing.assert_allclose(result, 0.5, atol=1e-10)


def test_box_filter_preserves_shape() -> None:
    img = np.random.rand(32, 48).astype(np.float64)
    result = _box_filter(img, radius=2)
    assert result.shape == img.shape


def test_guided_filter_smooths_constant() -> None:
    I = np.full((32, 32), 0.5, dtype=np.float64)
    p = np.random.rand(32, 32).astype(np.float64)
    result = _guided_filter(I, p, radius=3, eps=1e-1)
    # Uniform guidance: filter approximates mean of p
    expected = p.mean()
    assert abs(result.mean() - expected) < 0.2


def test_remove_small_components() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    # Big component
    mask[5:20, 5:20] = True
    # Tiny noise spot
    mask[30, 30] = True
    result = _remove_small_components(mask, min_area=10)
    assert result[10, 10] is np.True_
    assert result[30, 30] is np.False_


def test_refine_mask_preserves_large_mask() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # Draw a white rectangle
    frame[10:30, 10:30] = 255
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    result = refine_mask(frame, mask, radius=3, eps=1e-2, min_area=0)
    assert result.dtype == bool
    assert result.shape == mask.shape
    # Area should be preserved (roughly)
    assert 0.7 < result.sum() / mask.sum() < 1.5


def test_refine_mask_returns_original_on_catastrophic_result() -> None:
    """If refinement would collapse the mask, return original."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # Mask that is a tiny speck — refinement may remove it entirely
    mask = np.zeros((64, 64), dtype=bool)
    mask[0, 0] = True
    # With large min_area, it would be removed, triggering fallback
    result = refine_mask(frame, mask, min_area=1000)
    # Should return original (1 px) since ratio would be 0
    assert result.sum() == 1


def test_refine_mask_handles_empty_mask() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    result = refine_mask(frame, mask)
    assert not result.any()


def test_refine_mask_handles_none_frame() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    result = refine_mask(None, mask)
    # Should return original mask
    np.testing.assert_array_equal(result, mask)


def test_refine_mask_alignment() -> None:
    """Guided filter should push mask edges toward luminance edges."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # Horizontal edge at row 32
    frame[:32, :] = 255
    mask = np.zeros((64, 64), dtype=bool)
    # Slightly misaligned mask (2px below the edge)
    mask[34:50, :] = True
    result = refine_mask(frame, mask, radius=5, eps=1e-3, min_area=0)
    # The refined mask should extend upward toward the edge
    assert result.sum() > 0


def test_refine_mask_strength_zero_is_noop() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:32, :] = 255
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    result = refine_mask(frame, mask, smooth_strength=0)
    np.testing.assert_array_equal(result, mask)


def test_refine_mask_strength_range_runs_and_keeps_area() -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[20:108, 20:108] = 255
    mask = np.zeros((128, 128), dtype=bool)
    mask[25:105, 25:105] = True
    for strength in (1, 5, 10):
        result = refine_mask(frame, mask.copy(), smooth_strength=strength, min_area=0)
        assert result.dtype == bool
        assert result.shape == mask.shape
        assert 0.5 < result.sum() / mask.sum() < 1.8


def test_refine_mask_preserves_thin_hair_at_high_strength() -> None:
    frame = np.full((96, 96, 3), 32, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[55:85, 30:66] = True
    mask[42:57, 36:38] = True
    mask[40:58, 46:48] = True
    mask[43:56, 56:58] = True
    hair = np.zeros_like(mask)
    hair[42:57, 36:38] = True
    hair[40:58, 46:48] = True
    hair[43:56, 56:58] = True

    result = refine_mask(frame, mask, smooth_strength=10, min_area=100)

    np.testing.assert_array_equal(result[hair], mask[hair])
    assert not np.any(result & ~mask)


def test_refine_mask_retains_small_source_components() -> None:
    frame = np.full((64, 64, 3), 32, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:15, 10:15] = True
    mask[40, 40] = True

    result = refine_mask(frame, mask, smooth_strength=8, min_area=100)

    assert result[40, 40]
    assert result[12, 12]


def _square_mask(
    size: int = 64, box: tuple[int, int, int, int] = (16, 16, 48, 48)
) -> np.ndarray:
    """Square bool mask; ``box`` is (row_start, col_start, row_end, col_end)."""
    mask = np.zeros((size, size), dtype=bool)
    mask[box[0] : box[2], box[1] : box[3]] = True
    return mask


def test_erode_mask_reduces_area() -> None:
    """A 2 px inset must shrink the matte and pull the contour inward."""
    mask = _square_mask()

    eroded = erode_mask(mask, 2)

    assert eroded.dtype == bool
    assert eroded.shape == mask.shape
    # 32x32 block eroded by a 5x5 ellipse -> 28x28 of interior pixels.
    assert int(mask.sum()) == 1024
    assert int(eroded.sum()) == 784
    # The contour moves inward, and erosion never adds pixels.
    assert not eroded[16, 30]
    assert not eroded[17, 30]
    assert eroded[30, 30]
    assert not np.any(eroded & ~mask)


def test_erode_mask_zero_and_negative_are_noops() -> None:
    mask = _square_mask()

    for inset in (0, -4):
        result = erode_mask(mask, inset)
        assert result is mask


def test_erode_mask_scales_monotonically() -> None:
    """Larger insets never grow the mask again."""
    mask = _square_mask(box=(20, 20, 60, 60))

    areas = [int(erode_mask(mask, inset).sum()) for inset in (0, 1, 2, 4, 8)]

    assert areas == [1600, 1444, 1296, 1024, 576]
    assert areas == sorted(areas, reverse=True)


def test_erode_mask_removes_thin_features() -> None:
    """Features thinner than 2 * inset + 1 px are eroded away entirely."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:34, 10:54] = True  # 4 px tall band

    assert erode_mask(mask, 1).any()
    assert not erode_mask(mask, 3).any()


def test_erode_mask_tolerates_bad_input() -> None:
    mask = _square_mask()

    assert erode_mask(mask, None) is mask
    assert erode_mask(mask, "nope") is mask
    assert not erode_mask(np.zeros((8, 8), dtype=bool), 4).any()


def test_refine_mask_applies_mask_inset() -> None:
    """mask_inset shrinks the mask even when edge refinement is disabled."""
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    mask = _square_mask()

    untouched = refine_mask(frame, mask, smooth_strength=0)
    inset = refine_mask(frame, mask, smooth_strength=0, mask_inset=2)

    assert int(untouched.sum()) == int(mask.sum())
    assert int(inset.sum()) < int(untouched.sum())


def test_refine_mask_default_inset_is_noop() -> None:
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    mask = _square_mask()

    result = refine_mask(frame, mask, smooth_strength=4, mask_inset=0)

    baseline = refine_mask(frame, mask, smooth_strength=4)
    np.testing.assert_array_equal(result, baseline)
