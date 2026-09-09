"""Interactive video canvas: display frame, mask overlay, prompt points."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from src.models.sam2_manager import PointPrompt

logger = logging.getLogger(__name__)


class VideoCanvas(QWidget):
    """Widget that shows a video frame with mask overlay and interactive points."""

    # Emitted when user adds or moves a point
    point_added = Signal(int, int, int)  # x, y, label (1=pos, 0=neg)
    point_removed = Signal(int, int)  # x, y
    frame_navigate = Signal(int)  # delta frames (wheel)
    request_segment = Signal()  # user pressed Space or clicked segment
    play_pause = Signal()  # user pressed play

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # bool HxW
        self._points: list[PointPrompt] = []
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ #
    # Setters
    # ------------------------------------------------------------------ #
    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        self._frame = frame
        self.update()

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self._mask = mask.astype(bool) if mask is not None else None
        self.update()

    def set_points(self, points: list[PointPrompt]) -> None:
        self._points = list(points)
        self.update()

    def clear_mask(self) -> None:
        self._mask = None
        self.update()

    def clear_points(self) -> None:
        self._points = []
        self.update()

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(20, 20, 25))

        if self._frame is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No video loaded")
            return

        # Compute scaled rect keeping aspect ratio
        img_h, img_w = self._frame.shape[:2]
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / img_w, widget_h / img_h)
        self._scale = scale
        draw_w, draw_h = int(img_w * scale), int(img_h * scale)
        off_x = (widget_w - draw_w) // 2
        off_y = (widget_h - draw_h) // 2
        self._offset = QPoint(off_x, off_y)

        # Draw frame
        pixmap = self._np_to_pixmap(self._frame)
        target = QRect(off_x, off_y, draw_w, draw_h)
        painter.drawPixmap(target, pixmap, pixmap.rect())

        # Draw mask overlay
        if self._mask is not None:
            mask_pix = self._mask_to_pixmap(self._mask)
            painter.drawPixmap(target, mask_pix, mask_pix.rect())

        # Draw points
        for pt in self._points:
            cx = int(off_x + pt.x * scale)
            cy = int(off_y + pt.y * scale)
            color = QColor(50, 255, 50) if pt.label == 1 else QColor(255, 50, 50)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(cx - 5, cy - 5, 10, 10)

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._frame is None:
            return
        pos = event.pos()
        img_x, img_y = self._widget_to_image(pos.x(), pos.y())
        if img_x is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.point_added.emit(img_x, img_y, 1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.point_added.emit(img_x, img_y, 0)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        direction = 1 if delta < 0 else -1  # natural scroll feel
        self.frame_navigate.emit(direction)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space:
            self.request_segment.emit()
        elif event.key() == Qt.Key.Key_Left:
            self.frame_navigate.emit(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.frame_navigate.emit(1)
        elif event.key() == Qt.Key.Key_P:
            self.play_pause.emit()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _widget_to_image(self, wx: int, wy: int) -> tuple[Optional[int], Optional[int]]:
        if self._scale == 0:
            return None, None
        ix = int((wx - self._offset.x()) / self._scale)
        iy = int((wy - self._offset.y()) / self._scale)
        if self._frame is not None:
            h, w = self._frame.shape[:2]
            if 0 <= ix < w and 0 <= iy < h:
                return ix, iy
        return None, None

    @staticmethod
    def _np_to_pixmap(arr: np.ndarray) -> QPixmap:
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_ARGB32
        image = QImage(arr.data, w, h, bytes_per_line, fmt)
        return QPixmap.fromImage(image.copy())

    @staticmethod
    def _mask_to_pixmap(mask: np.ndarray) -> QPixmap:
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask] = [0, 200, 255, 120]  # cyan semi-transparent
        bytes_per_line = 4 * w
        image = QImage(rgba.data, w, h, bytes_per_line, QImage.Format.Format_ARGB32)
        return QPixmap.fromImage(image.copy())
