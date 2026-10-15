"""Tests for letterbox / pillarbox detection."""

from __future__ import annotations

import numpy as np

from src.processing.letterbox import (
    Letterbox,
    combine_letterboxes,
    crop_frame,
    crop_mask,
    detect_letterbox,
    detect_letterbox_sequence,
    embed_mask,
)


def _frame(
    height: int = 100,
    width: int = 120,
    bar_top: int = 0,
    bar_bottom: int = 0,
    bar_left: int = 0,
    bar_right: int = 0,
    content: tuple[int, int, int] = (120, 90, 60),
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[bar_top : height - bar_bottom, bar_left : width - bar_right] = content
    return frame


def test_no_bars_returns_empty_box() -> None:
    box = detect_letterbox(_frame())
    assert box == Letterbox()
    assert not box.active


def test_letterbox_top_and_bottom() -> None:
    box = detect_letterbox(_frame(bar_top=12, bar_bottom=8))
    assert box.top == 12
    assert box.bottom == 8
    assert box.left == 0
    assert box.right == 0
    assert box.is_letterbox
    assert not box.is_pillarbox


def test_pillarbox_left_and_right() -> None:
    box = detect_letterbox(_frame(bar_left=10, bar_right=6))
    assert box.left == 10
    assert box.right == 6
    assert box.is_pillarbox


def test_uniform_frame_is_never_cropped() -> None:
    assert detect_letterbox(np.zeros((60, 60, 3), dtype=np.uint8)) == Letterbox()
    assert detect_letterbox(np.full((60, 60, 3), 200, dtype=np.uint8)) == Letterbox()


def test_dark_content_is_not_treated_as_bars() -> None:
    # A dim night scene is above the black threshold, so nothing is cropped.
    box = detect_letterbox(_frame(content=(40, 38, 42)))
    assert box == Letterbox()


def test_single_row_bar_is_filtered_by_min_bar() -> None:
    box = detect_letterbox(_frame(bar_top=1), min_bar=2)
    assert box.top == 0


def test_threshold_is_configurable() -> None:
    frame = _frame(bar_top=10, bar_bottom=10, content=(30, 30, 30))
    # Default threshold: only the true black bars are detected.
    assert detect_letterbox(frame) == Letterbox(top=10, bottom=10)
    # Raise the threshold above the content luma and the dark content also
    # counts as a bar, so the detected run grows.
    raised = detect_letterbox(frame, threshold=40.0)
    assert raised.active
    assert raised.top > 10


def test_grayscale_input() -> None:
    frame = np.zeros((50, 50), dtype=np.uint8)
    frame[5:45, :] = 128
    box = detect_letterbox(frame)
    assert box.top == 5
    assert box.bottom == 5


def test_crop_frame_and_mask_roundtrip() -> None:
    box = Letterbox(top=10, bottom=5, left=4, right=6)
    frame = _frame(bar_top=10, bar_bottom=5, bar_left=4, bar_right=6)
    cropped = crop_frame(frame, box)
    assert cropped.shape == (85, 110, 3)
    assert int(cropped.min()) > 0  # only content remains (bars are pure black)

    mask = np.ones((100, 120), dtype=bool)
    cropped_mask = crop_mask(mask, box)
    assert cropped_mask.shape == (85, 110)

    restored = embed_mask(cropped_mask, box, (100, 120))
    assert restored.shape == (100, 120)
    assert not restored[0, 0]  # bar stays empty
    assert restored[50, 50]


def test_combine_is_conservative() -> None:
    a = Letterbox(top=10, bottom=10)
    b = Letterbox(top=6, bottom=10, left=2)
    combined = combine_letterboxes([a, b])
    assert combined.top == 6
    assert combined.bottom == 10
    assert combined.left == 0


def test_sequence_uses_the_smallest_bar() -> None:
    frames = [
        _frame(bar_top=12, bar_bottom=12),
        _frame(bar_top=10, bar_bottom=12),
    ]
    box = detect_letterbox_sequence(frames)
    assert box.top == 10
    assert box.bottom == 12
