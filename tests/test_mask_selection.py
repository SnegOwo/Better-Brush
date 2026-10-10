"""Tests for prompt-aware mask selection and background trimming."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QCoreApplication

from src.models.sam2_manager import PointPrompt
from src.processing.inference_worker import PostProcessWorker
from src.processing.mask_refine import refine_mask
from src.processing.mask_selection import (
    choose_prompt_mask_index,
    prompt_consistency,
    trim_prompt_background,
)


def _sky_frame() -> tuple[np.ndarray, np.ndarray, PointPrompt, PointPrompt]:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[:, :] = (210, 150, 70)
    frame[25:70, 45:80] = (230, 190, 150)
    mask = np.zeros((80, 120), dtype=bool)
    mask[:, :] = True
    mask[20:75, 35:90] = True
    positive = PointPrompt(60, 40, 1)
    negative = PointPrompt(10, 10, 0)
    return frame, mask, positive, negative


def test_prompt_selection_prefers_consistent_candidate() -> None:
    positive = PointPrompt(10, 10, 1)
    negative = PointPrompt(90, 10, 0)
    broad = np.ones((20, 100), dtype=bool)
    focused = np.zeros((20, 100), dtype=bool)
    focused[2:18, 2:40] = True

    index = choose_prompt_mask_index(
        [broad, focused], [0.99, 0.4], [positive, negative]
    )
    consistency, missed, hits = prompt_consistency(focused, [positive, negative])

    assert index == 1
    assert consistency == 1.0
    assert missed == 0
    assert hits == 0


def test_prompt_selection_limits_broad_sky_for_multiple_points() -> None:
    points = [PointPrompt(20, 20, 1), PointPrompt(60, 60, 1), PointPrompt(70, 70, 1)]
    subject = np.zeros((100, 100), dtype=bool)
    subject[10:85, 10:85] = True
    sky_takeover = subject.copy()
    sky_takeover[:, 85:] = True

    index = choose_prompt_mask_index(
        [sky_takeover, subject],
        [0.95, 0.72],
        points,
    )

    assert index == 1


def test_negative_prompt_trims_connected_sky() -> None:
    frame, mask, positive, negative = _sky_frame()
    cleaned = trim_prompt_background(frame, mask, [positive, negative])

    assert not cleaned[10, 10]
    assert cleaned[50, 60]
    assert int(cleaned.sum()) < int(mask.sum())


def test_postprocess_worker_protects_negative_background() -> None:
    _app = QCoreApplication.instance() or QCoreApplication([])
    frame, mask, positive, negative = _sky_frame()
    received: list[object] = []
    worker = PostProcessWorker(
        lambda _index: frame,
        {0: mask},
        refine_edges=False,
        temporal_smooth=False,
        prompt_points=[positive, negative],
    )
    worker.finished.connect(received.append)
    worker.run()

    assert received
    result = received[0]
    assert not isinstance(result, Exception)
    assert not result[0][10, 10]
    assert result[0][40, 60]


def test_smooth_strength_changes_refinement_output() -> None:
    rng = np.random.default_rng(4)
    frame = rng.integers(20, 235, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[14:50, 15:49] = True

    low = refine_mask(frame, mask, smooth_strength=1, detail_preservation=0.5)
    high = refine_mask(frame, mask, smooth_strength=9, detail_preservation=0.5)

    assert not np.array_equal(low, high)


# ------------------------------------------------------------------ #
# Per-frame profile resolver contract                                #
# ------------------------------------------------------------------ #
def _run_postprocess(resolver) -> object:
    """Run the worker synchronously and return whatever ``finished`` emitted."""
    import numpy as np
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.instance() or QCoreApplication([])
    frame = np.full((32, 32, 3), 120, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    worker = PostProcessWorker(
        lambda _i: frame,
        {0: mask},
        refine_edges=True,
        temporal_smooth=True,
        profile_resolver=resolver,
    )
    received: list[object] = []
    worker.finished.connect(received.append)
    worker.run()
    assert received, "finished must always be emitted, success or failure"
    return received[0]


def test_profile_resolver_accepts_the_canonical_five_tuple() -> None:
    result = _run_postprocess(lambda i: (3, True, 0.75, 2, 1))
    assert not isinstance(result, Exception)
    assert set(result) == {0}


def test_profile_resolver_accepts_a_legacy_four_tuple() -> None:
    """Regression: a 4-tuple used to raise 'expected 5, got 4' mid-worker.

    The missing mask_inset defaults to 0, which is the neutral value, so an
    older resolver keeps working instead of killing post-processing.
    """
    result = _run_postprocess(lambda i: (3, True, 0.75, 2))
    assert not isinstance(result, Exception), result
    assert set(result) == {0}


def test_profile_resolver_rejects_a_wrong_arity_with_a_clear_error() -> None:
    result = _run_postprocess(lambda i: (3, True, 0.75))
    assert isinstance(result, ValueError)
    message = str(result)
    assert "5 values" in message
    assert "got 3" in message


def test_profile_resolver_rejects_a_non_tuple() -> None:
    result = _run_postprocess(lambda i: [3, True, 0.75, 2, 1])
    assert isinstance(result, TypeError)
    assert "must return a tuple" in str(result)


def test_postprocess_failure_is_reported_not_swallowed() -> None:
    """A failure must reach the GUI as a value, never silence."""

    def explode(i: int):
        raise RuntimeError("boom")

    result = _run_postprocess(explode)
    assert isinstance(result, RuntimeError)
    assert str(result) == "boom"
