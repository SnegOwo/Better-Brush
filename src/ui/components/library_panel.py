"""Lightweight 'Library' panel: browse exported videos and recode them."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LibraryDialog(QDialog):
    """Wraps :class:`LibraryPanel` in a standalone dialog for the menu bar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Media Library")
        self.setMinimumSize(560, 440)
        self.setStyleSheet("background: #202024;")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self._panel = LibraryPanel()
        root.addWidget(self._panel)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    @property
    def panel(self) -> LibraryPanel:
        return self._panel

from src.core import constants
from src.core import licensing
from src.core.project import ProjectManager
from src.export.exporter import _resolve_ffmpeg

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}

# Recode presets: (label, ext, video args)
RECODE_PRESETS = [
    ("MP4 (H.264) — universal sharing", ".mp4",
     ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]),
    ("MOV (ProRes 422 HQ)", ".mov",
     ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"]),
    ("WebM (VP9) — web", ".webm",
     ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p"]),
]


def _size_label(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    return f"{size / 1_000:.0f} KB"


class RecodeWorker(threading.Thread):
    """Runs one ffmpeg recode off the GUI thread."""

    def __init__(
        self,
        src: Path,
        dst: Path,
        video_args: list[str],
        preserve_audio: bool,
    ) -> None:
        super().__init__(daemon=True)
        self._src = src
        self._dst = dst
        self._video_args = video_args
        self._preserve_audio = preserve_audio
        self.result: Optional[str] = None  # None = ok, else error message
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        ffmpeg = _resolve_ffmpeg()
        cmd = [ffmpeg, "-y", "-nostdin", "-i", str(self._src), *self._video_args]
        if self._preserve_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd.append(str(self._dst))
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = proc.communicate(timeout=7200)
        except Exception as exc:  # noqa: BLE001
            self.result = f"Failed to launch ffmpeg: {exc}"
            return
        if proc.returncode != 0:
            tail = " | ".join(stderr.strip().splitlines()[-4:]) if stderr.strip() else ""
            self.result = f"ffmpeg exited with code {proc.returncode}. {tail}"
            self._dst.unlink(missing_ok=True)


class RecodeDialog(QDialog):
    """Small modal to pick a target codec and recode a single video."""

    def __init__(self, parent: Optional[QWidget], src: Path) -> None:
        super().__init__(parent)
        self._src = src
        self.setWindowTitle(f"Recode: {src.name}")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setStyleSheet("background: #202024;")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        info = QLabel(f"Source: {src.name}  ({_size_label(src)})")
        info.setStyleSheet("color: #9a9aa4; font-size: 11px; background: transparent;")
        info.setWordWrap(True)
        root.addWidget(info)

        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel("Target:")
        lbl.setStyleSheet("color: #8b8b94; font-size: 11px; background: transparent;")
        row.addWidget(lbl)
        self._combo = QComboBox()
        for label, _ext, _args in RECODE_PRESETS:
            self._combo.addItem(label)
        self._combo.setCurrentIndex(0)
        row.addWidget(self._combo, stretch=1)
        root.addLayout(row)

        self._chk_audio = QCheckBox("Keep original audio (AAC)")
        self._chk_audio.setChecked(True)
        self._chk_audio.setStyleSheet("color: #c4c4cc; background: transparent;")
        root.addWidget(self._chk_audio)

        actions = QHBoxLayout()
        actions.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        actions.addWidget(btn_cancel)
        self._btn_go = QPushButton("Recode")
        self._btn_go.setProperty("cssClass", "accent")
        self._btn_go.clicked.connect(self._on_recode)
        actions.addWidget(self._btn_go)
        root.addLayout(actions)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.hide()
        root.addWidget(self._bar)

        self._worker: Optional[RecodeWorker] = None

    def _on_recode(self) -> None:
        label = self._combo.currentText()
        preset = next(p for p in RECODE_PRESETS if p[0] == label)
        _lbl, ext, video_args = preset
        dst = self._src.with_name(self._src.stem + "_re" + ext)

        self._btn_go.setEnabled(False)
        self._combo.setEnabled(False)
        self._chk_audio.setEnabled(False)
        self._bar.show()

        self._worker = RecodeWorker(
            self._src, dst, video_args, self._chk_audio.isChecked()
        )
        self._worker.start()
        # Block this dialog, keeping the busy bar alive, until the recode ends.
        while self._worker.is_alive():
            QApplication.processEvents()
            time.sleep(0.03)
        self._bar.hide()

        if self._worker.result:
            QMessageBox.warning(
                self, "Recode failed", f"Could not recode:\n{self._worker.result}"
            )
        else:
            self.accept()
            QMessageBox.information(self, "Recode OK", f"Saved to:\n{dst}")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.is_alive():
            self._worker.cancel()
        super().closeEvent(event)


class LibraryPanel(QWidget):
    """Lists video files from an export folder with a simple recode action."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dir = self._load_dir()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._lbl_dir = QLabel(str(self._dir))
        self._lbl_dir.setWordWrap(True)
        self._lbl_dir.setStyleSheet(
            "color: #6b6b74; font-size: 10px; background: transparent;"
        )
        root.addWidget(self._lbl_dir)

        btns = QHBoxLayout()
        btns.setSpacing(6)
        btn_pick = QPushButton("Choose")
        btn_pick.setToolTip("Point the library at any folder")
        btn_pick.clicked.connect(self._choose_folder)
        btns.addWidget(btn_pick)
        btn_open = QPushButton("Open")
        btn_open.setToolTip("Open the folder in Explorer")
        btn_open.clicked.connect(self._open_folder)
        btns.addWidget(btn_open)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        btns.addWidget(btn_refresh)
        root.addLayout(btns)

        self._list = QListWidget()
        self._list.setMinimumHeight(120)
        self._list.itemDoubleClicked.connect(self._open_video)
        self._list.setStyleSheet(
            "QListWidget { background: #16161b; border: 1px solid #2e2e33; "
            "border-radius: 4px; color: #d4d4d8; }"
        )
        root.addWidget(self._list)

        self._btn_recode = QPushButton("Recode Selected…")
        self._btn_recode.setEnabled(False)
        self._btn_recode.clicked.connect(self._recode_selected)
        root.addWidget(self._btn_recode)

        self._list.currentItemChanged.connect(self._on_selection)
        self.refresh()
        self._apply_edition()

    def _apply_edition(self) -> None:
        """Disable recode button when Pro is not activated."""
        if licensing.is_pro():
            self._btn_recode.setEnabled(bool(self._list.currentItem()))
            self._btn_recode.setToolTip("")
        else:
            self._btn_recode.setEnabled(False)
            self._btn_recode.setToolTip(
                "Pro feature — activate Pro to recode (Help → Activate Pro)"
            )

    # ── Public API ─────────────────────────────────────────────────────────
    def refresh(self) -> None:
        self._list.clear()
        if not self._dir.is_dir():
            lbl = QListWidgetItem("Folder not found — pick another one.")
            lbl.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(lbl)
            return
        items = []
        for path in sorted(self._dir.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                items.append(path)
        if not items:
            lbl = QListWidgetItem("No videos exported yet.")
            lbl.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(lbl)
            return
        for path in items:
            modified = path.stat().st_mtime
            from datetime import datetime

            when = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
            item = QListWidgetItem(f"{path.name}  ({_size_label(path)}, {when})")
            item.setToolTip(str(path))
            self._list.addItem(item)

    def current_path(self) -> Optional[Path]:
        item = self._list.currentItem()
        if item is None:
            return None
        p = Path(item.toolTip())
        return p if p.is_file() else None

    # ── Handlers ───────────────────────────────────────────────────────────
    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Library Folder", str(self._dir))
        if path:
            self._dir = Path(path)
            self._save_dir()
            self._lbl_dir.setText(str(self._dir))
            self.refresh()

    def _open_folder(self) -> None:
        if self._dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._dir)))

    def _open_video(self, item: QListWidgetItem) -> None:
        path = Path(item.toolTip())
        if path.is_file():
            os.startfile(str(path))  # open with the system video player

    def _on_selection(self, current: QListWidgetItem, _prev) -> None:
        selected = current is not None and Path(current.toolTip()).is_file()
        self._btn_recode.setEnabled(selected and licensing.is_pro())

    def _recode_selected(self) -> None:
        if not licensing.is_pro():
            QMessageBox.information(
                self,
                "Pro feature",
                "Recoding videos is a Better Brush Pro feature.\n\n"
                "Activate Pro via Help → Activate Pro…",
            )
            return
        path = self.current_path()
        if path is None:
            return
        RecodeDialog(self, path).exec()

    # ── Persistence ────────────────────────────────────────────────────────
    def _load_dir(self) -> Path:
        settings = ProjectManager.load_user_settings()
        saved = settings.get("library_dir")
        if saved and Path(saved).is_dir():
            return Path(saved)
        return constants.EXPORTS_DIR

    def _save_dir(self) -> None:
        settings = ProjectManager.load_user_settings()
        settings["library_dir"] = str(self._dir)
        ProjectManager.save_user_settings(settings)