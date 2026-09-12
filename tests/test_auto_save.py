"""Tests for the automatic-save feature and user-settings persistence.

Run with:
    python -m pytest tests/test_auto_save.py -v
"""

from __future__ import annotations

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

from src.core.project import ProjectManager
from src.ui.main_window import MainWindow

_APP = None


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(ProjectManager, "settings_dir", lambda: tmp_path)
    yield


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _win() -> MainWindow:
    _app()
    win = MainWindow()
    win._canvas.setFixedSize(480, 320)
    win.show()
    win.activateWindow()
    QTest.qWaitForWindowExposed(win)
    # Bypass the "Discard unsaved changes?" modal so close() never blocks.
    win._confirm_discard = lambda: True
    return win


def _stub_video(win: MainWindow, tmp_path: Path) -> None:
    video = SimpleNamespace(
        path=tmp_path / "demo.mp4", fps=30.0, width=160, height=120, total_frames=12
    )
    win._video_reader = video


def test_user_settings_defaults(tmp_path):
    data = ProjectManager.load_user_settings()
    assert data["auto_save_enabled"] is True
    assert data["auto_save_interval_s"] == 60


def test_user_settings_roundtrip(tmp_path):
    ProjectManager.save_user_settings({"auto_save_enabled": False})
    data = ProjectManager.load_user_settings()
    assert data["auto_save_enabled"] is False
    assert data["auto_save_interval_s"] == 60  # untouched field keeps default

    ProjectManager.save_user_settings({"auto_save_interval_s": 120})
    data = ProjectManager.load_user_settings()
    assert data["auto_save_interval_s"] == 120
    assert data["auto_save_enabled"] is False  # merged, not overwritten


def test_enabled_by_default_and_action_reflects_state(tmp_path):
    win = _win()
    try:
        assert win._auto_save_enabled is True
        assert win._act_auto_save.isChecked()
    finally:
        win.close()


def test_toggle_action_persists_and_stops_timer(tmp_path):
    win = _win()
    try:
        _stub_video(win, tmp_path)
        win._project_dirty = True
        win._update_auto_save_timer()
        assert win._auto_save_timer.isActive()

        win._act_auto_save.setChecked(False)
        assert win._auto_save_enabled is False
        assert not win._auto_save_timer.isActive()
        assert ProjectManager.load_user_settings()["auto_save_enabled"] is False
    finally:
        win.close()


def test_autosave_writes_file_and_does_not_clear_dirty(tmp_path):
    win = _win()
    try:
        _stub_video(win, tmp_path)
        win._project_dirty = True
        win._masks[0] = np.ones((120, 160), dtype=bool)

        win._perform_autosave()

        dest = ProjectManager.autosave_dir() / "demo.autosave.bbproj"
        assert dest.is_file(), "autosave file was not written"
        loaded = ProjectManager.load(dest)
        assert 0 in loaded.masks
        assert loaded.masks[0].all()
        assert len(loaded.objects) == 1
        # Autosave must not clear the dirty flag or touch the project path.
        assert win._project_dirty is True
        assert win._project_path is None
    finally:
        win.close()


def test_autosave_skipped_when_clean(tmp_path):
    win = _win()
    try:
        _stub_video(win, tmp_path)
        win._project_dirty = False
        win._masks[0] = np.ones((120, 160), dtype=bool)
        win._perform_autosave()
        dest = ProjectManager.autosave_dir() / "demo.autosave.bbproj"
        assert not dest.exists(), "clean sessions must not autosave"
    finally:
        win.close()


if __name__ == "__main__":
    raise SystemExit("run via pytest")
