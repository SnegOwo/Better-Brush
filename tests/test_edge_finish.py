"""Soft matte coverage and edge decontamination.

The binary-matte defects these fix were measured on a real ProRes 4444 export:
alpha had exactly two values (0 and 255) with zero partial pixels, and edge
pixels carried a blue-shift of 20.8 that should not have been there at all.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.export.exporter import Exporter
from src.processing.edge_finish import (
    decontaminate_edges,
    estimate_background,
    finish_edges,
    soft_coverage,
)

SKY = (107, 230, 250)
HAIR = (245, 225, 195)


def _stranded(size: int = 120, width: int = 200):
    """A pale subject over a saturated sky, with a thin tail.

    Both shapes matter: the blob gives a genuine interior that must survive
    untouched, and the 2px tail is the thin structure that a naive
    signed-distance treatment erases entirely.
    """
    import cv2

    frame = np.empty((size, width, 3), dtype=np.uint8)
    frame[:] = SKY
    blob = np.zeros((size, width), dtype=np.uint8)
    cv2.circle(blob, (70, 55), 28, 1, -1)
    # A 2px tail reaching right, with ragged gaps like a real hair matte.
    tail = np.zeros((size, width), dtype=bool)
    tail[:, 96:150] = True
    for y in range(0, size, 7):
        tail[y, 96:150] = False
    mask = blob.astype(bool) | tail
    frame[mask] = HAIR
    # Composite the boundary the way a camera does: an edge pixel is a blend of
    # subject and background weighted by coverage. Setting it to pure sky would
    # leave nothing to recover, and decontaminating it would correctly - but
    # uselessly - hand the background straight back.
    alpha = soft_coverage(mask, 2.0)
    band = (alpha > 0.05) & (alpha < 0.95)
    weight = alpha[band][:, None]
    hair = np.array(HAIR, np.float32)
    sky = np.array(SKY, np.float32)
    frame[band] = (weight * hair + (1.0 - weight) * sky).astype(np.uint8)
    return frame, mask


def _blob_interior(mask: np.ndarray) -> np.ndarray:
    """Pixels deep inside the blob, far from any boundary."""
    return soft_coverage(mask, 2.0) == 1.0


def _blueness(pixels: np.ndarray) -> float:
    values = np.asarray(pixels, np.float32)
    return float((values[:, 2] - (values[:, 0] + values[:, 1]) / 2).mean())


# ------------------------------------------------------------------ #
# Coverage                                                           #
# ------------------------------------------------------------------ #
def test_hard_matte_has_no_partial_coverage() -> None:
    """The regression this module exists to remove."""
    _, mask = _stranded()
    hard = (mask * 255).astype(np.uint8)
    assert len(np.unique(hard)) == 2
    assert ((hard > 0) & (hard < 255)).sum() == 0


def test_soft_coverage_produces_a_ramp() -> None:
    _, mask = _stranded()
    alpha = soft_coverage(mask, 2.0)
    partial = (alpha > 0.05) & (alpha < 0.95)
    assert partial.sum() > 0, "the edge must anti-alias"
    levels = np.unique(np.round(alpha * 255).astype(np.uint8))
    assert len(levels) > 2
    assert alpha.min() == 0.0
    assert alpha.max() == 1.0, "the blob interior must be fully opaque"


def test_soft_coverage_keeps_thin_structures_alive() -> None:
    """A signed-distance bug here erases strands completely, so guard it."""
    _, mask = _stranded()
    alpha = soft_coverage(mask, 2.0)
    # The middle of the tail must survive at usable coverage...
    tail = alpha[:, 120]
    assert tail.mean() > 0.6, "a 2px strand must not be eroded away"
    # ...while the sky well away from any subject stays fully clear.
    assert alpha[:, 170].max() == 0.0


def test_soft_coverage_keeps_the_interior_solid() -> None:
    blob = np.zeros((60, 60), dtype=np.uint8)
    import cv2

    cv2.circle(blob, (30, 30), 20, 1, -1)
    alpha = soft_coverage(blob.astype(bool), 2.0)
    assert alpha[30, 30] == 1.0
    assert alpha[0, 0] == 0.0


def test_soft_coverage_of_zero_is_the_hard_matte() -> None:
    _, mask = _stranded()
    assert np.array_equal(soft_coverage(mask, 0), mask.astype(np.float32))


def test_soft_coverage_handles_empty_and_full_mattes() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    assert not soft_coverage(empty, 2.0).any()
    full = np.ones((10, 10), dtype=bool)
    assert soft_coverage(full, 2.0).all()


# ------------------------------------------------------------------ #
# Decontamination                                                    #
# ------------------------------------------------------------------ #
def test_decontamination_strips_the_fringe_from_the_edge() -> None:
    frame, mask = _stranded()
    fixed, alpha = finish_edges(frame, mask, 2.0)
    partial = (alpha > 0.05) & (alpha < 0.95)
    # Only pixels on the subject side carry genuine foreground signal, which
    # is the case the un-premultiply is valid for.
    inside = partial & mask
    assert inside.sum() > 0
    # The whole point: the recovered colour is the subject's own, not merely
    # less blue. A "neutral grey" despill would fail this.
    assert np.allclose(fixed[inside].mean(0), HAIR, atol=6), (
        f"edge recovered {fixed[inside].mean(0).round(1)}, expected ~{HAIR}"
    )
    hair_chroma = HAIR[2] - (HAIR[0] + HAIR[1]) / 2
    assert abs(_blueness(fixed[inside]) - hair_chroma) < 12
    assert _blueness(fixed[inside]) < _blueness(frame[inside])


def test_decontamination_leaves_interior_and_background_alone() -> None:
    frame, mask = _stranded()
    fixed, _ = finish_edges(frame, mask, 2.0)
    interior = _blob_interior(mask)
    assert interior.sum() > 500, "fixture needs a real interior"
    assert np.array_equal(fixed[interior], frame[interior]), "subject untouched"
    far = np.zeros(mask.shape, dtype=bool)
    far[:, :30] = True
    assert np.array_equal(fixed[far], frame[far]), "background untouched"


def test_decontamination_does_not_modify_its_input() -> None:
    frame, mask = _stranded()
    original = frame.copy()
    finish_edges(frame, mask, 2.0)
    assert np.array_equal(frame, original)


def test_estimate_background_uses_the_transparent_region() -> None:
    frame, mask = _stranded()
    background = estimate_background(frame, mask)
    assert background is not None
    assert np.allclose(background, SKY, atol=2.0)
    # Nothing outside the matte means nothing to estimate from.
    assert estimate_background(frame, np.ones_like(mask)) is None


def test_decontamination_is_a_noop_without_partial_pixels() -> None:
    frame, mask = _stranded()
    hard = mask.astype(np.float32)
    assert np.array_equal(decontaminate_edges(frame, mask, hard), frame)


# ------------------------------------------------------------------ #
# Exporter integration                                               #
# ------------------------------------------------------------------ #
def test_exporter_writes_soft_alpha_when_asked() -> None:
    frame, mask = _stranded()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp)
        exporter = Exporter(output_dir=out)
        assert exporter._compose_rgba(frame, mask)[..., 3].max() == 255
        assert len(np.unique(exporter._compose_rgba(frame, mask)[..., 3])) == 2

        alpha = soft_coverage(mask, 2.0)
        soft = exporter._compose_rgba(frame, mask, alpha=alpha)[..., 3]
        assert ((soft > 0) & (soft < 255)).sum() > 0


def test_exporter_anti_aliases_a_chroma_key_background() -> None:
    frame, mask = _stranded()
    exporter = Exporter(output_dir=Path("."))
    alpha = soft_coverage(mask, 2.0)
    keyed = exporter._compose_rgba(frame, mask, background=(0, 255, 0), alpha=alpha)
    # Fully opaque output, but the key edge is blended rather than stepped.
    assert keyed[..., 3].min() == 255
    hard = exporter._compose_rgba(frame, mask, background=(0, 255, 0))
    assert not np.array_equal(keyed[..., :3], hard[..., :3])


def test_exported_png_carries_a_partial_alpha() -> None:
    """End-to-end: the file on disk must not be a hard matte any more."""
    frame, mask = _stranded()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        exporter = Exporter(output_dir=Path(tmp))
        rgba = exporter._compose_rgba(frame, mask, alpha=soft_coverage(mask, 2.0))
        path = Path(tmp) / "tile.png"
        Image.fromarray(rgba, mode="RGBA").save(path)
        written = np.asarray(Image.open(path).convert("RGBA"))
        channel = written[..., 3]
        assert ((channel > 0) & (channel < 255)).sum() > 0
        assert len(np.unique(channel)) > 2


if __name__ == "__main__":
    import inspect

    tests = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
