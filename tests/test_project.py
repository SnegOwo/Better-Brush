"""Round-trip tests for .bbproj project files."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.project import ProjectData, ProjectManager


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Keep the recent-projects registry inside tmp_path, not APPDATA."""
    monkeypatch.setattr(
        ProjectManager, "_registry_path", lambda: tmp_path / "registry.json"
    )
    yield


def test_roundtrip_preserves_session(tmp_path):
    data = ProjectData(
        video_path=str(tmp_path / "clip.mp4"),
        points=[{"x": 1, "y": 2, "label": 1}, {"x": 5, "y": 6, "label": 0}],
        keyframes=[0, 3, 7],
        masks={
            0: np.array([[True, False], [False, True]]),
            3: np.ones((4, 5), dtype=bool),
        },
        current_frame=3,
        edit_mode="brush",
        brush_size=12,
        timeline_selection=[2, 9],
        scene_cuts=[0, 12, 40],
    )
    out = ProjectManager.save(tmp_path / "proj.bbproj", data)

    loaded = ProjectManager.load(out)
    assert loaded.version == 1
    assert loaded.video_path == str(tmp_path / "clip.mp4")
    assert loaded.points == data.points
    assert loaded.keyframes == [0, 3, 7]
    assert loaded.current_frame == 3
    assert loaded.edit_mode == "brush"
    assert loaded.brush_size == 12
    assert loaded.timeline_selection == [2, 9]
    assert loaded.scene_cuts == [0, 12, 40]
    np.testing.assert_array_equal(loaded.masks[0], data.masks[0])
    np.testing.assert_array_equal(loaded.masks[3], data.masks[3])
    assert loaded.masks[0].dtype == bool


def test_extension_is_appended(tmp_path):
    out = ProjectManager.save(tmp_path / "no_suffix", ProjectData())
    assert out.name == "no_suffix.bbproj"


def test_load_missing_raises(tmp_path):
    with pytest.raises(OSError):
        ProjectManager.load(tmp_path / "nope.bbproj")


def test_load_corrupt_raises(tmp_path):
    p = tmp_path / "bad.bbproj"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        ProjectManager.load(p)


def test_recent_registry(tmp_path):
    proj = tmp_path / "proj.bbproj"
    ProjectManager.add_recent(proj, video_path=tmp_path / "clip.mp4")
    recent = ProjectManager.load_recent()
    assert recent[0]["project"] == str(proj)
    assert recent[0]["video_name"] == "clip.mp4"

    # Saving the same project again reorders to the top without duplicates
    ProjectManager.add_recent(proj, video_path=tmp_path / "clip.mp4")
    recent = ProjectManager.load_recent()
    assert len(recent) == 1

    ProjectManager.remove_recent(proj)
    assert ProjectManager.load_recent() == []


def test_first_run_flag(tmp_path):
    assert ProjectManager.is_first_run() is True
    ProjectManager.set_first_run_done()
    assert ProjectManager.is_first_run() is False
