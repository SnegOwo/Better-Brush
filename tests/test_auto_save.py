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
from PySide6.QtCore import Qt
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
    assert data["post_process_enabled"] is True
    assert data["smooth_strength"] == 3
    assert data["console_shortcut"] == "F7"


def test_legacy_zero_smoothing_stays_disabled(tmp_path):
    ProjectManager.settings_file().write_text(
        '{"smooth_strength": 0}', encoding="utf-8"
    )
    data = ProjectManager.load_user_settings()
    assert data["post_process_enabled"] is False


def test_post_process_toggle_persists_and_keeps_slider_editable(tmp_path):
    win = _win()
    try:
        assert win._post_process_enabled is True
        assert win._smooth_slider.isEnabled()
        win._chk_post_process.setChecked(False)
        assert win._post_process_enabled is False
        assert win._smooth_slider.isEnabled()
        assert ProjectManager.load_user_settings()["post_process_enabled"] is False
    finally:
        win.close()


def test_console_shortcut_is_configurable(tmp_path):
    win = _win()
    try:
        assert win._act_console.shortcut().toString() == "F7"
        result = win._console_shortcut(["F8"])
        assert "F8" in result
        assert win._act_console.shortcut().toString() == "F8"
        assert ProjectManager.load_user_settings()["console_shortcut"] == "F8"
    finally:
        win.close()


def test_f7_opens_diagnostics_console(tmp_path):
    win = _win()
    try:
        QTest.keyClick(win, Qt.Key.Key_F7)
        assert win._console is not None
        assert win._console.isVisible()
    finally:
        win.close()


def test_console_lists_commands_and_reports_unknown_names(tmp_path):
    win = _win()
    try:
        win._show_diagnostics_console()
        win._console._input.setText("list")
        win._console._btn_run.click()
        assert "gpu" in win._console._output.toPlainText()
        win._console._input.setText("gpui")
        win._console._btn_run.click()
        output = win._console._output.toPlainText()
        assert "unknown command" in output
        assert "gpu" in output
    finally:
        win.close()


def test_context_and_title_show_unsaved_work(tmp_path):
    win = _win()
    try:
        _stub_video(win, tmp_path)
        win._timeline_selection = (3, 8)
        win._mark_dirty()
        assert "In 0003-0008 (6f)" in win._lbl_context.text()
        assert "Unsaved" in win._lbl_context.text()
        assert win.windowTitle().endswith("*")
    finally:
        win.close()


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
