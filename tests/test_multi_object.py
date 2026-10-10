"""Tests for the multi-object scene model and v2 project round-trip.

Run with:
    python -m pytest tests/test_multi_object.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.core import licensing
from src.core.project import ObjectRecord, ProjectData, ProjectManager, mask_to_b64
from src.core.scene import OBJECT_PALETTE
from src.ui.main_window import MainWindow

_APP = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _win() -> MainWindow:
    _app()
    win = MainWindow()
    win._canvas.setFixedSize(720, 480)
    win._canvas.set_frame(np.zeros((100, 160, 3), dtype=np.uint8))
    win.show()
    win.activateWindow()
    QTest.qWaitForWindowExposed(win)
    win._canvas.grab()
    return win


def _stub_video(win: MainWindow, tmp_path: Path, name: str = "clip.mp4") -> None:
    video = SimpleNamespace(
        path=tmp_path / name, fps=30.0, width=160, height=100, total_frames=10
    )
    win._video_reader = video


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Keep settings / autosave / registry files inside tmp_path."""
    monkeypatch.setattr(ProjectManager, "settings_dir", lambda: tmp_path)
    # Multi-object editing is a Pro feature — run these tests with Pro active.
    key = licensing.issue_key("CI Tester")
    ok, msg = licensing.activate(key)
    assert ok, msg
    yield
    licensing.deactivate()


def test_initial_single_object():
    win = _win()
    try:
        assert len(win._objects) == 1
        obj = win._active_object()
        assert obj.name == "Object 1"
        assert obj.color == OBJECT_PALETTE[0]
        assert win._active_object_id == obj.obj_id
        # Aliasing: main-window fields point at the active object's data.
        assert win._points is obj.points
        assert win._masks is obj.masks
        assert win._keyframes is obj.keyframes
    finally:
        win.close()


def test_add_object_switches_active():
    win = _win()
    try:
        win._add_object()
        assert len(win._objects) == 2
        obj2 = win._active_object()
        assert obj2.name == "Object 2"
        assert obj2.color == OBJECT_PALETTE[1]
        assert win._masks is obj2.masks

        # Editing goes to the active object only.
        win._on_canvas_point(10, 20, 1)
        assert len(obj2.points) == 1
        assert len(win._objects[0].points) == 0
    finally:
        win.close()


def test_switch_object_aliases_and_color():
    win = _win()
    try:
        first = win._objects[0]
        win._add_object()
        second = win._active_object()
        win._set_active_object(first.obj_id)
        assert win._masks is first.masks
        assert win._points is first.points
        assert win._canvas._mask_color.name().lower() == first.color.lower()
        win._set_active_object(second.obj_id)
        assert win._canvas._mask_color.name().lower() == second.color.lower()
    finally:
        win.close()


def test_cannot_delete_last_object():
    win = _win()
    try:
        win._delete_active_object()
        assert len(win._objects) == 1
        assert win._active_object() is not None
    finally:
        win.close()


def test_delete_object_switches_to_remaining():
    win = _win()
    try:
        first = win._objects[0]
        win._add_object()
        win._masks[0] = np.ones((100, 160), dtype=bool)  # mask on object 2
        win._delete_active_object()
        assert len(win._objects) == 1
        assert win._objects[0].obj_id == first.obj_id
        assert 0 not in win._masks, "deleted object masks must not leak"
        assert win._active_object() is first
    finally:
        win.close()


def test_combined_masks_unions_objects():
    win = _win()
    try:
        a = np.zeros((10, 10), dtype=bool)
        a[0:3, 0:3] = True
        b = np.zeros((10, 10), dtype=bool)
        b[6:9, 6:9] = True
        win._objects[0].masks[0] = a
        win._add_object()
        second = win._active_object()
        second.masks[0] = b
        second.masks[5] = np.ones((10, 10), dtype=bool)

        combined = win._combined_masks()
        assert 0 in combined and 5 in combined
        np.testing.assert_array_equal(combined[0], a | b)
        assert combined[5].all()
    finally:
        win.close()


def test_combo_reflects_objects():
    win = _win()
    try:
        assert win._combo_object.count() == 1
        assert win._combo_object.currentData() == win._active_object_id
        win._add_object()
        assert win._combo_object.count() == 2
        assert win._combo_object.currentData() == win._active_object_id
        assert win._btn_delete_object.isEnabled()
    finally:
        win.close()


def test_v2_project_roundtrip(tmp_path):
    data = ProjectData(
        video_path=str(tmp_path / "clip.mp4"),
        current_frame=3,
        edit_mode="brush",
        brush_size=14,
        objects=[
            ObjectRecord(
                obj_id=1,
                name="Hero",
                color="#00b4f0",
                points=[{"x": 1, "y": 2, "label": 1}],
                keyframes=[0],
                masks={0: np.array([[True, False], [False, True]], dtype=bool)},
            ),
            ObjectRecord(
                obj_id=2,
                name="Sidekick",
                color="#f4b400",
                points=[],
                keyframes=[4],
                masks={4: np.ones((4, 5), dtype=bool)},
            ),
        ],
        active_object=2,
    )
    out = ProjectManager.save(tmp_path / "proj.bbproj", data)
    loaded = ProjectManager.load(out)

    assert loaded.version == 3
    assert loaded.active_object == 2
    assert len(loaded.objects) == 2
    hero = next(r for r in loaded.objects if r.name == "Hero")
    assert hero.points == [{"x": 1, "y": 2, "label": 1}]
    assert hero.masks[0].dtype == bool
    np.testing.assert_array_equal(hero.masks[0], data.objects[0].masks[0])
    side = next(r for r in loaded.objects if r.obj_id == 2)
    np.testing.assert_array_equal(side.masks[4], data.objects[1].masks[4])
    assert sorted(side.keyframes) == [4]


def test_legacy_v1_project_loads_as_single_object(tmp_path):
    legacy = {
        "app": "Better Brush",
        "version": 1,
        "video_path": str(tmp_path / "old.mp4"),
        "points": [{"x": 3, "y": 4, "label": 1}],
        "keyframes": [2],
        "masks": {"2": mask_to_b64(np.ones((2, 3), dtype=bool))},
        "current_frame": 2,
        "edit_mode": "brush",
        "brush_size": 30,
    }
    p = tmp_path / "legacy.bbproj"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    data = ProjectManager.load(p)
    assert data.version == 1
    assert data.points == [{"x": 3, "y": 4, "label": 1}]
    assert data.keyframes == [2]
    assert 2 in data.masks
    assert data.objects == []


if __name__ == "__main__":
    raise SystemExit("run via pytest")
