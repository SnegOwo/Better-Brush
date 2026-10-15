"""Regression tests for the removal composite's matte handling.

The removal pipeline reconstructs the subject's background and then blends it
back over the original frame through a grown/feathered matte.  These tests pin
the two blending bugs that made the composite silently do the wrong thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without package install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing.removal import RemovalSettings, _composite


def _frame(value: int, size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _square_mask(size: int = 16, inset: int = 4) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[inset : size - inset, inset : size - inset] = True
    return mask


def test_no_grow_no_feather_still_respects_the_mask() -> None:
    """With grow=0 and feather=0 the mask must still gate the composite.

    The old code returned the reconstruction for the *whole* frame whenever
    both were zero, wiping everything outside the subject.
    """
    original = _frame(10)
    removed = _frame(200)
    mask = _square_mask()
    out = _composite(original, removed, mask, RemovalSettings(mask_grow=0, feather=0))
    assert np.all(out[mask] == 200)
    assert np.all(out[~mask] == 10)


def test_boolean_matte_scales_to_full_alpha() -> None:
    """A grown (un-feathered) boolean matte must fully replace inside it.

    Casting the bool matte straight to uint8 produced 0/1 alphas, so the
    composite only applied ~0.4 % of the reconstruction.
    """
    original = _frame(0)
    removed = _frame(255)
    mask = _square_mask(inset=6)
    out = _composite(original, removed, mask, RemovalSettings(mask_grow=1, feather=0))
    assert np.all(out[mask] == 255)


def test_whole_region_matte_is_not_collapsed_by_normalize() -> None:
    """A matte covering the whole region must still replace it, not vanish.

    ``cv2.normalize`` maps a constant array to all-zero, which silently
    disabled the removal; the guard must keep the whole region repaired.
    """
    original = _frame(0)
    removed = _frame(255)
    mask = np.ones((16, 16), dtype=bool)
    out = _composite(original, removed, mask, RemovalSettings(mask_grow=0, feather=9))
    assert np.all(out == 255)


def test_empty_matte_returns_the_original() -> None:
    """Nothing masked means nothing is repaired."""
    original = _frame(10)
    removed = _frame(200)
    out = _composite(
        original, removed, np.zeros((16, 16), dtype=bool), RemovalSettings()
    )
    assert np.array_equal(out, original)
