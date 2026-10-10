"""Version history browser with mask thumbnails and restore action."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.history import HistoryEntry, ProjectHistory


def _mask_pixmap(data) -> QPixmap:
    width = 360
    height = 220
    background = np.full((height, width, 3), 35, dtype=np.uint8)
    masks = getattr(data, "masks", {}) if data is not None else {}
    if masks:
        try:
            frame_index = min(masks)
            mask = np.asarray(masks[frame_index], dtype=bool)
            image = np.repeat(mask[..., None], 3, axis=2).astype(np.uint8) * 70
            image[mask] = (0, 190, 240)
            h, w = mask.shape[:2]
            y = max(0, (height - min(h, height)) // 2)
            x = max(0, (width - min(w, width)) // 2)
            view = image[: min(h, height), : min(w, width)]
            background[y : y + view.shape[0], x : x + view.shape[1]] = view
        except (TypeError, ValueError, IndexError):
            pass
    array = np.ascontiguousarray(background)
    qimage = QImage(
        array.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimage.copy())


class HistoryDialog(QDialog):
    """Compare project-history mask snapshots and request a restore."""

    restore_requested = Signal(int)

    def __init__(self, history: ProjectHistory, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project History")
        self.setModal(True)
        self.resize(900, 620)
        self._history = history
        self._entries: list[HistoryEntry] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(10)
        title = QLabel("Project History")
        title.setStyleSheet("font-size: 17px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(title)
        intro = QLabel(
            "Checkpoints are versioned, checksummed, and stored beside the project. "
            "Select a version to compare its mask with the previous checkpoint."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9a9aa4; font-size: 11px;")
        root.addWidget(intro)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._selection_changed)
        root.addWidget(self._list, stretch=1)

        previews = QHBoxLayout()
        self._before = QLabel("Before")
        self._after = QLabel("After")
        for label in (self._before, self._after):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(180)
            label.setStyleSheet("color: #a0a0a8; border: 1px solid #3a3a40;")
            previews.addWidget(label, stretch=1)
        root.addLayout(previews)

        actions = QHBoxLayout()
        self._restore = QPushButton("Restore Selected")
        self._restore.setProperty("cssClass", "accent")
        self._restore.clicked.connect(self._restore_selected)
        actions.addWidget(self._restore)
        actions.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        actions.addWidget(close)
        root.addLayout(actions)

    def refresh(self) -> None:
        self._entries = self._history.list()
        selected = self._selected_version()
        self._list.clear()
        for entry in self._entries:
            item = QListWidgetItem(
                f"v{entry.version}  ·  {entry.timestamp}  ·  {entry.label}"
            )
            item.setData(Qt.ItemDataRole.UserRole, entry.version)
            item.setToolTip(str(entry.path))
            self._list.addItem(item)
            if entry.version == selected:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(self._list.count() - 1)
        self._update_previews()

    def _selected_version(self) -> int | None:
        item = self._list.currentItem()
        return None if item is None else int(item.data(Qt.ItemDataRole.UserRole))

    def _selection_changed(self, _current, _previous) -> None:
        self._update_previews()

    def _update_previews(self) -> None:
        version = self._selected_version()
        if version is None:
            self._before.setPixmap(QPixmap())
            self._after.setPixmap(QPixmap())
            self._restore.setEnabled(False)
            return
        index = next(
            (i for i, entry in enumerate(self._entries) if entry.version == version),
            0,
        )
        after_entry = self._entries[index]
        before_entry = self._entries[min(index + 1, len(self._entries) - 1)]
        before_data = self._history.load(before_entry.version)
        after_data = self._history.load(after_entry.version)
        self._before.setText(f"v{before_entry.version} · {before_entry.label}")
        self._after.setText(f"v{after_entry.version} · {after_entry.label}")
        self._before.setPixmap(_mask_pixmap(before_data))
        self._after.setPixmap(_mask_pixmap(after_data))
        self._restore.setEnabled(True)

    def _restore_selected(self) -> None:
        version = self._selected_version()
        if version is None:
            return
        answer = QMessageBox.question(
            self,
            "Restore checkpoint",
            f"Restore version {version}? Current unsaved changes will be kept in a new checkpoint.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.restore_requested.emit(version)
            self.accept()


__all__ = ["HistoryDialog"]
