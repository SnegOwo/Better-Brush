"""Timeline widget for frame navigation and keyframe display."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class Timeline(QWidget):
    """Horizontal timeline with playhead and keyframe markers."""

    frame_selected = Signal(int)  # frame index clicked
    playhead_moved = Signal(int)  # frame index dragged

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self._total_frames = 0
        self._current_frame = 0
        self._keyframes: set[int] = set()
        self._dragging = False

    # ------------------------------------------------------------------ #
    # Setters
    # ------------------------------------------------------------------ #
    def set_duration(self, total_frames: int) -> None:
        self._total_frames = max(1, total_frames)
        self.update()

    def set_current_frame(self, idx: int) -> None:
        self._current_frame = max(0, min(idx, self._total_frames - 1))
        self.update()

    def set_keyframes(self, frames: set[int]) -> None:
        self._keyframes = set(frames)
        self.update()

    def add_keyframe(self, idx: int) -> None:
        self._keyframes.add(idx)
        self.update()

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(35, 35, 40))

        if self._total_frames <= 1:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No video")
            return

        # Draw keyframe ticks
        pen = QPen(QColor(120, 120, 120))
        painter.setPen(pen)
        for kf in self._keyframes:
            x = int((kf / self._total_frames) * w)
            painter.drawLine(x, 0, x, h)

        # Draw playhead
        x = int((self._current_frame / self._total_frames) * w)
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawLine(x, 0, x, h)

        # Draw frame number
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(8, h - 8, f"Frame {self._current_frame} / {self._total_frames}")

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging = True
        self._seek_to(event.pos().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._seek_to(event.pos().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging = False

    def _seek_to(self, x: int) -> None:
        ratio = max(0.0, min(1.0, x / self.width()))
        idx = int(ratio * (self._total_frames - 1))
        self.set_current_frame(idx)
        self.frame_selected.emit(idx)
