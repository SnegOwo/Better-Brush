"""Tests for cel-hold detection and mask reuse."""

from __future__ import annotations

import numpy as np

from src.processing.cel_hold import (
    apply_cel_hold,
    find_holds,
    frame_signature,
    frames_match,
    held_frame_count,
    hold_runs,
    signature_distance,
)


def _flat(value: int, height: int = 32, width: int = 32) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _noise(seed: int, height: int = 32, width: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def test_signature_shape_and_dtype() -> None:
    sig = frame_signature(_flat(100))
    assert sig.shape == (64, 64)
    assert sig.dtype == np.uint8


def test_identical_frames_match() -> None:
    frame = _noise(1)
    assert frames_match(frame, frame.copy())


def test_different_frames_do_not_match() -> None:
    assert not frames_match(_flat(10), _flat(200))


def test_find_holds_marks_repeats_only() -> None:
    held = _noise(2)
    other = _noise(3)
    frames = {0: held, 1: held.copy(), 2: other, 3: other.copy()}
    assert find_holds(frames) == {1, 3}


def test_apply_cel_hold_copies_masks_verbatim() -> None:
    held = _noise(4)
    frames = {0: held, 1: held.copy(), 2: held.copy()}
    m0 = np.zeros((32, 32), dtype=bool)
    m0[1:30, 1:30] = True
    m1 = np.zeros((32, 32), dtype=bool)
    m1[5:20, 5:20] = True
    m2 = np.zeros((32, 32), dtype=bool)
    m2[8:25, 2:18] = True

    result = apply_cel_hold({0: m0, 1: m1, 2: m2}, frames)

    assert result.copied == (1, 2)
    assert result.copied_count == 2
    np.testing.assert_array_equal(result.masks[1], m0)
    np.testing.assert_array_equal(result.masks[2], m0)


def test_apply_cel_hold_keeps_real_motion() -> None:
    frames = {0: _noise(5), 1: _noise(6)}
    m0 = np.zeros((32, 32), dtype=bool)
    m0[1:10, 1:10] = True
    m1 = np.zeros((32, 32), dtype=bool)
    m1[20:30, 20:30] = True
    result = apply_cel_hold({0: m0, 1: m1}, frames)
    assert result.copied == ()
    np.testing.assert_array_equal(result.masks[1], m1)


def test_apply_cel_hold_does_not_mutate_input() -> None:
    held = _noise(7)
    frames = {0: held, 1: held.copy()}
    m0 = np.zeros((32, 32), dtype=bool)
    m0[1:10, 1:10] = True
    m1 = np.zeros((32, 32), dtype=bool)
    m1[20:30, 20:30] = True
    original = m1.copy()
    apply_cel_hold({0: m0, 1: m1}, frames)
    np.testing.assert_array_equal(m1, original)


def test_single_frame_has_no_holds() -> None:
    result = apply_cel_hold({0: np.ones((4, 4), dtype=bool)}, {0: _flat(1)})
    assert result.copied == ()
    assert find_holds({0: _flat(1)}) == set()


def test_hold_runs_groups_repeats() -> None:
    held = _noise(8)
    other = _noise(9)
    frames = {0: held, 1: held.copy(), 2: held.copy(), 3: other, 4: other.copy()}
    assert hold_runs(frames) == [(0, 2), (3, 4)]


def test_signature_distance_tolerates_tiny_change() -> None:
    a = _flat(100)
    b = a.copy()
    b[0, 0] = (110, 110, 110)
    assert signature_distance(frame_signature(a), frame_signature(b)) < 2.0
    assert frames_match(a, b, threshold=2.0)


def test_held_frame_count() -> None:
    held = _flat(120)
    assert held_frame_count([held, held.copy(), held.copy()]) == 2
