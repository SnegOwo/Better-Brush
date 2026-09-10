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

    # Point prompting
    point_added = Signal(int, int, int)  # x, y, label (1=pos, 0=neg)
    point_removed = Signal(int, int)  # x, y
    frame_navigate = Signal(int)  # delta frames (wheel)
    request_segment = Signal()  # user pressed Space or clicked segment
    play_pause = Signal()  # user pressed Play/Pause
    # Brush editing
    mask_edited = Signal()  # emitted after a brush stroke modified the mask

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # bool HxW
        self._points: list[PointPrompt] = []
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        # Interaction state
        self._edit_mode = "points"  # "points" | "brush"
        self._brush_size = 20  # diameter in widget px
        self._brush_erase = False
        self._preview_mode = "mask"  # "frame" | "mask" | "cutout"
        self._brush_hover: Optional[tuple[int, int]] = None
        self._painting = False
        self._painted = False
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

    def get_mask(self) -> Optional[np.ndarray]:
        """Return a copy of the current mask (bool HxW) or None."""
        return None if self._mask is None else self._mask.copy()

    # ------------------------------------------------------------------ #
    # Tools / preview
    # ------------------------------------------------------------------ #
    def set_edit_mode(self, mode: str) -> None:
        """Switch tool: ``"points"`` (SAM 2 prompts) or ``"brush"``
        (manual mask painting for frame-by-frame correction)."""
        self._edit_mode = mode if mode in ("points", "brush") else "points"
        self.update()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(int(size), 500))
        self.update()

    def set_brush_erase(self, erase: bool) -> None:
        self._brush_erase = bool(erase)
        self.update()

    def set_preview_mode(self, mode: str) -> None:
        """``"frame"`` raw video, ``"mask"`` cyan overlay, ``"cutout"`` the
        subject cut on a checkerboard (preview of the exported result)."""
        self._preview_mode = mode if mode in ("frame", "mask", "cutout") else "mask"
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

        # Draw preview (frame / mask / cutout)
        if self._preview_mode == "cutout" and self._mask is not None:
            self._draw_cutout(painter, target)
        elif self._preview_mode == "mask" and self._mask is not None:
            mask_pix = self._mask_to_pixmap(self._mask)
            painter.drawPixmap(target, mask_pix, mask_pix.rect())

        # Draw prompt points (points mode only)
        if self._edit_mode == "points":
            for pt in self._points:
                cx = int(off_x + pt.x * scale)
                cy = int(off_y + pt.y * scale)
                color = QColor(50, 255, 50) if pt.label == 1 else QColor(255, 50, 50)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawEllipse(cx - 5, cy - 5, 10, 10)

        # Brush cursor (brush mode)
        if self._edit_mode == "brush" and self._brush_hover is not None:
            hx, hy = self._brush_hover
            painter.setPen(QPen(QColor(255, 220, 40), 1))
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            r = self._brush_size
            painter.drawEllipse(hx - r, hy - r, r * 2, r * 2)

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._frame is None:
            return
        if self._edit_mode == "brush":
            pos = event.position().toPoint()
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self._painting = True
                self._painted = False
                self._apply_brush(pos.x(), pos.y())
            return
        pos = event.position().toPoint()
        img_x, img_y = self._widget_to_image(pos.x(), pos.y())
        if img_x is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.point_added.emit(img_x, img_y, 1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.point_added.emit(img_x, img_y, 0)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._frame is None:
            return
        pos = event.position().toPoint()
        if self._edit_mode == "brush":
            self._brush_hover = (pos.x(), pos.y())
            if self._painting:
                self._apply_brush(pos.x(), pos.y())
            self.update()
            return
        self._brush_hover = None

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._edit_mode == "brush" and self._painting:
            self._painting = False
            if self._painted:
                self.mask_edited.emit()
            self.update()

    def mouseLeaveEvent(self, event) -> None:  # noqa: N802
        self._brush_hover = None
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._edit_mode == "brush" and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            delta = event.angleDelta().y()
            self.set_brush_size(self._brush_size + (1 if delta > 0 else -1))
            return
        delta = event.angleDelta().y()
        direction = 1 if delta < 0 else -1  # natural scroll feel
        self.frame_navigate.emit(direction)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.request_segment.emit()
        elif key == Qt.Key.Key_Left:
            self.frame_navigate.emit(-1)
        elif key == Qt.Key.Key_Right:
            self.frame_navigate.emit(1)
        elif key == Qt.Key.Key_P:
            self.play_pause.emit()
        elif key == Qt.Key.Key_B:
            self.set_edit_mode("brush" if self._edit_mode == "points" else "points")
        elif key == Qt.Key.Key_BracketLeft:
            self.set_brush_size(self._brush_size - 2)
        elif key == Qt.Key.Key_BracketRight:
            self.set_brush_size(self._brush_size + 2)
        else:
            super().keyPressEvent(event)

    def _apply_brush(self, wx: int, wy: int) -> None:
        """Paint or erase a brush circle (widget coords) on the mask."""
        img_x, img_y = self._widget_to_image(wx, wy)
        if img_x is None or self._frame is None:
            return
        h, w = self._frame.shape[:2]
        if self._mask is None:
            self._mask = np.zeros((h, w), dtype=bool)
        radius = max(1, int(self._brush_size / self._scale))
        x0, x1 = max(0, img_x - radius), min(w, img_x + radius + 1)
        y0, y1 = max(0, img_y - radius), min(h, img_y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = ((xx - img_x) ** 2 + (yy - img_y) ** 2) <= radius**2
        if self._brush_erase:
            self._mask[y0:y1, x0:x1][inside] = False
        else:
            self._mask[y0:y1, x0:x1][inside] = True
        self._painted = True
        self.update()

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

    def _draw_cutout(self, painter: QPainter, target: QRect) -> None:
        """Compose the cut-out subject (masked frame) on a checkerboard."""
        if self._frame is None or self._mask is None:
            return
        h, w = self._mask.shape
        rgb = self._frame[:, :, :3].astype(np.uint8)
        mask = self._mask
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((xx // 8) + (yy // 8)) % 2 == 0
        bg = np.where(checker[..., None], np.uint8(70), np.uint8(40))
        keep = mask[..., None]
        cut = rgb * keep.astype(np.uint8) + bg * (~keep).astype(np.uint8)
        pix = self._np_to_pixmap(cut)
        painter.drawPixmap(target, pix, pix.rect())
