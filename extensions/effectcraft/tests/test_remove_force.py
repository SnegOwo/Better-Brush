"""A red stroke has to *delete*.

The engine trims the seed and re-propagates, so a large object creeps back on
other frames — in the field the same spot had to be painted six times. These
tests cover the adapter half of the fix:
``Correction.remove_force`` (1-5) grows the painted region, and
``erase_removed`` stamps it out of every finished frame the matte still covers.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from effectcraft_bridge.pipeline import erase_removed, remove_region
from effectcraft_bridge.protocol import Correction

from test_pipeline import correction_over, make_job, make_source


def rect(x0: int, y0: int, x1: int, y1: int):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def correction(**over) -> Correction:
    return Correction.from_dict(correction_over(**over))


# --------------------------------------------------------------------- #
# The wire format
# --------------------------------------------------------------------- #
def test_remove_force_defaults_to_three():
    data = correction_over()

    assert "remove_force" not in data
    assert Correction.from_dict(data).remove_force == 3


def test_remove_force_is_clamped_to_one_and_five():
    assert correction(remove_force=0).remove_force == 1
    assert correction(remove_force=-7).remove_force == 1
    assert correction(remove_force=99).remove_force == 5


def test_remove_force_round_trips():
    sent = correction(remove_force=5)

    assert Correction.from_dict(sent.to_dict()).remove_force == 5


# --------------------------------------------------------------------- #
# The region
# --------------------------------------------------------------------- #
def test_force_one_is_exactly_what_was_painted(tmp_path):
    job = make_job(tmp_path, make_source(tmp_path),
                   correction=correction_over(remove=[rect(20, 20, 40, 40)],
                                               remove_force=1))

    region = remove_region(job)

    assert region is not None
    assert region[30, 30]          # inside
    assert not region[5, 5]        # far outside
    assert not region[19, 30]      # one pixel outside the painted edge


def big_job(tmp_path, **over):
    """A 400 px wide job, so the force margin is measurable in pixels."""
    return make_job(
        tmp_path,
        make_source(tmp_path, size=(400, 300)),
        media={"width": 400, "height": 300, "frame_rate": 24.0},
        **over,
    )


def test_force_five_takes_a_margin(tmp_path):
    soft = big_job(tmp_path, correction=correction_over(
        remove=[rect(100, 100, 180, 180)], remove_force=1))
    hard = big_job(tmp_path, correction=correction_over(
        remove=[rect(100, 100, 180, 180)], remove_force=5))

    soft_region = remove_region(soft)
    hard_region = remove_region(hard)

    # 400 px wide * 0.25%/step * 4 steps = 4 px of margin at force 5.
    assert not soft_region[140, 98]      # 2 px outside: exactly what was painted
    assert hard_region[140, 98]          # ...but inside the grown region
    assert not hard_region[140, 90]      # 10 px outside is still untouched
    assert int(hard_region.sum()) > int(soft_region.sum())


def test_no_remove_strokes_means_no_region(tmp_path):
    job = make_job(tmp_path, make_source(tmp_path),
                   correction=correction_over(remove=[]))

    assert remove_region(job) is None


# --------------------------------------------------------------------- #
# The erase pass
# --------------------------------------------------------------------- #
def write_frame(folder: Path, name: str, size: tuple[int, int] = (80, 60)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    frame[:, :, 3] = 255
    path = folder / name
    cv2.imwrite(str(path), frame)
    return path


def read_alpha(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, 3]


def test_erase_stamps_the_region_out_of_every_frame(tmp_path):
    png_dir = tmp_path / "png"
    frames = [write_frame(png_dir, f"{i:05d}.png") for i in range(1, 4)]
    job = make_job(tmp_path, make_source(tmp_path),
                   correction=correction_over(remove=[rect(20, 20, 40, 40)]))

    erased = erase_removed(job, png_dir, lambda *args: None)

    assert erased == 3
    for frame in frames:
        alpha = read_alpha(frame)
        assert int(alpha[30, 30]) == 0
        assert int(alpha[5, 5]) == 255


def test_erase_only_touches_pixels_the_matte_still_covers(tmp_path):
    """An object that has moved away leaves the background under the stroke."""
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    frame = np.zeros((60, 80, 4), dtype=np.uint8)
    frame[20:45, 20:45, 3] = 255      # the subject sits only in the stroke area
    path = png_dir / "00001.png"
    cv2.imwrite(str(path), frame)
    job = make_job(tmp_path, make_source(tmp_path),
                   correction=correction_over(remove=[rect(20, 20, 40, 40)]))

    erase_removed(job, png_dir, lambda *args: None)

    alpha = read_alpha(path)
    assert int(alpha[2, 70]) == 0     # was transparent, stays transparent
    assert int(alpha[30, 30]) == 0    # covered, so deleted


def test_erase_skips_a_frame_that_is_not_the_jobs_size(tmp_path):
    png_dir = tmp_path / "png"
    odd = write_frame(png_dir, "00001.png", size=(40, 30))
    job = make_job(tmp_path, make_source(tmp_path),
                   correction=correction_over(remove=[rect(20, 20, 40, 40)]))

    assert erase_removed(job, png_dir, lambda *args: None) == 0
    assert int(read_alpha(odd)[10, 10]) == 255


def test_erase_reports_progress_with_a_known_stage(tmp_path):
    png_dir = tmp_path / "png"
    write_frame(png_dir, "00001.png")
    job = make_job(tmp_path, make_source(tmp_path),
                   correction=correction_over(remove=[rect(20, 20, 40, 40)]))
    seen: list[str] = []

    erase_removed(job, png_dir, lambda stage, *args: seen.append(stage))

    assert seen and set(seen) == {"post"}
