"""The adapter's own post pass: what it smooths and what it must not touch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from effectcraft_bridge.postprocess import (
    PostProcessCancelled,
    post_process_sequence,
)

cv2 = pytest.importorskip("cv2")


def write_frames(png_dir: Path, alphas: list[np.ndarray], rgb: int = 90) -> list[Path]:
    png_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, alpha in enumerate(alphas):
        h, w = alpha.shape
        rgba = np.dstack([
            np.full((h, w), rgb, np.uint8),
            np.full((h, w), rgb, np.uint8),
            np.full((h, w), rgb, np.uint8),
            alpha,
        ])
        path = png_dir / f"{i:05d}.png"
        assert cv2.imwrite(str(path), rgba)
        files.append(path)
    return files


def read_alphas(png_dir: Path) -> list[np.ndarray]:
    return [
        cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[:, :, 3]
        for p in sorted(png_dir.glob("*.png"))
    ]


def test_flicker_between_frames_is_reduced(tmp_path: Path):
    """A held region that jumps between frames settles instead of strobing."""
    h, w = 32, 32
    # A gap inside the similarity gate (96 alpha levels): the blend can act.
    low = np.full((h, w), 100, np.uint8)
    high = np.full((h, w), 160, np.uint8)
    files = write_frames(tmp_path, [low, high, low, high])

    info = post_process_sequence(tmp_path)

    assert info["frames"] == 4
    assert info["temporal_window"] > 0
    after = read_alphas(tmp_path)
    gap_before = abs(int(high[0, 0]) - int(low[0, 0]))
    gap_after = abs(int(after[1][0, 0]) - int(after[0][0, 0]))
    assert gap_after < gap_before, "the temporal pass changed nothing visible"


def test_solid_interior_and_outside_survive_untouched(tmp_path: Path):
    """Smoothing the edge must never fade a solid region either way."""
    alpha = np.zeros((48, 48), np.uint8)
    alpha[8:40, 8:40] = 255
    files = write_frames(tmp_path, [alpha.copy(), alpha.copy()])

    post_process_sequence(tmp_path)

    after = read_alphas(tmp_path)
    for frame in after:
        assert int(frame[0, 0]) == 0        # outside stays fully out
        assert int(frame[24, 24]) == 255    # interior stays fully in


def test_rgb_channels_are_written_back_verbatim(tmp_path: Path):
    """Only the alpha may change; a colour shift would show as a tint."""
    alpha = np.full((24, 24), 128, np.uint8)
    files = write_frames(tmp_path, [alpha.copy()], rgb=37)
    before = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)

    post_process_sequence(tmp_path)

    after = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(after[:, :, :3], before[:, :, :3])


def test_a_cancel_flag_stops_the_pass(tmp_path: Path):
    write_frames(tmp_path, [np.zeros((8, 8), np.uint8)] * 3)

    with pytest.raises(PostProcessCancelled):
        post_process_sequence(tmp_path, cancelled=lambda: True)


def test_empty_folder_is_a_no_op(tmp_path: Path):
    assert post_process_sequence(tmp_path) == {
        "frames": 0,
        "temporal_window": 0,
        "seconds": 0.0,
    }


def test_window_zero_falls_back_to_edge_cleanup_only(tmp_path: Path):
    """Very long shots skip the temporal stack but still get the edge pass."""
    write_frames(tmp_path, [np.full((16, 16), 128, np.uint8)] * 2)

    info = post_process_sequence(tmp_path, window=0)

    assert info["frames"] == 2
    assert info["temporal_window"] == 0
