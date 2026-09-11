"""Interactive video canvas: display frame, mask overlay, prompt points."""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
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

# ── Visual constants ──────────────────────────────────────────────────────
_BG = QColor(20, 20, 24)
_POINT_POS = QColor(72, 199, 142)    # muted green
_POINT_NEG = QColor(230, 80, 80)     # muted red
_POINT_RING = QColor(255, 255, 255, 200)
_BRUSH_ADD = QColor(255, 255, 255, 180)
_BRUSH_ERASE = QColor(240, 90, 90, 180)
_MASK_COLOR = QColor(0, 180, 240, 100)
_CHECKER_A = QColor(55, 55, 60)
_CHECKER_B = QColor(40, 40, 44)


class VideoCanvas(QWidget):
    """Widget that shows a video frame with mask overlay and interactive points."""

# Point prompting
    point_added = Signal(int, int, int)  # x, y, label (1=pos, 0=neg)
    point_removed = Signal(object)  # PointPrompt that was deleted
    point_selected = Signal(int)  # index of selected point
    frame_navigate = Signal(int)  # delta frames (wheel)
    request_segment = Signal()  # requested via the Segment Frame button
    play_pause = Signal()  # user pressed Space/P or clicked Play/Pause
    # Brush editing
    brush_press = Signal()  # emitted at the start of a brush stroke
    mask_edited = Signal()  # emitted after a brush stroke modified the mask

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # bool HxW
        self._points: list[PointPrompt] = []
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        # View transform (zoom via Ctrl+scroll, pan via middle-drag)
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._fit_origin = QPoint(0, 0)
        self._panning = False
        self._pan_start = QPoint(0, 0)
        self._mask_color = _MASK_COLOR
        # Interaction state
        self._edit_mode = "points"  # "points" | "brush"
        self._brush_size = 20  # diameter in widget px
        self._brush_erase = False
        self._shift_held = False
        self._preview_mode = "mask"  # "frame" | "mask" | "cutout"
        self._brush_hover: Optional[tuple[int, int]] = None
        self._painting = False
        self._painted = False
        self._selected_point: Optional[int] = None  # index into _points
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background: transparent;")

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
        if self._selected_point is not None and self._selected_point >= len(self._points):
            self._selected_point = None
        self.update()

    def clear_mask(self) -> None:
        self._mask = None
        self.update()

    def clear_points(self) -> None:
        self._points = []
        self._selected_point = None
        self.update()

    def get_mask(self) -> Optional[np.ndarray]:
        """Return a copy of the current mask (bool HxW) or None."""
        return None if self._mask is None else self._mask.copy()

    # ------------------------------------------------------------------ #
    # Tools / preview
    # ------------------------------------------------------------------ #
    def set_edit_mode(self, mode: str) -> None:
        self._edit_mode = mode if mode in ("points", "brush") else "points"
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def get_edit_mode(self) -> str:
        """Return the active tool: ``"points"`` or ``"brush"``."""
        return self._edit_mode

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(int(size), 500))
        self.update()

    def set_brush_erase(self, erase: bool) -> None:
        self._brush_erase = bool(erase)
        self.update()

    def set_mask_color(self, color: QColor) -> None:
        """Set the overlay tint for the mask. Opacity is capped so the
        mask never fully hides the frame beneath it."""
        c = QColor(color)
        c.setAlpha(min(c.alpha(), 80))
        self._mask_color = c
        self.update()

    # ------------------------------------------------------------------ #
    # View transform (zoom / pan)
    # ------------------------------------------------------------------ #
    def _zoom_at(self, pos: QPoint, factor: float) -> None:
        """Zoom by ``factor`` keeping the image point under ``pos`` stationary."""
        if self._frame is None:
            return
        new_zoom = max(0.05, min(64.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        zf = new_zoom / self._zoom
        self._zoom = new_zoom
        old = self._offset
        fx, fy = self._fit_origin.x(), self._fit_origin.y()
        nx = pos.x() - (pos.x() - old.x()) * zf
        ny = pos.y() - (pos.y() - old.y()) * zf
        self._pan = QPoint(int(nx - fx), int(ny - fy))
        self.update()

    def _pan_by(self, dx: int, dy: int) -> None:
        self._pan = QPoint(self._pan.x() + dx, self._pan.y() + dy)
        self.update()

    def set_preview_mode(self, mode: str) -> None:
        self._preview_mode = mode if mode in ("frame", "mask", "cutout") else "mask"
        self.update()

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BG)

        if self._frame is None:
            painter.end()
            return

        # Compute scaled rect keeping aspect ratio, then apply zoom + pan.
        img_h, img_w = self._frame.shape[:2]
        widget_w, widget_h = self.width(), self.height()
        fit_scale = min(widget_w / img_w, widget_h / img_h)
        fit_w, fit_h = int(img_w * fit_scale), int(img_h * fit_scale)
        fit_x = (widget_w - fit_w) // 2
        fit_y = (widget_h - fit_h) // 2
        self._fit_origin = QPoint(fit_x, fit_y)
        self._scale = fit_scale * self._zoom
        off_x = int(fit_x + self._pan.x())
        off_y = int(fit_y + self._pan.y())
        self._offset = QPoint(off_x, off_y)

        target = QRect(off_x, off_y, int(img_w * self._scale), int(img_h * self._scale))

        # Subtle border around the frame
        painter.setPen(QPen(QColor(40, 40, 44), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(target.adjusted(-1, -1, 0, 0))

        # Draw frame
        pixmap = self._np_to_pixmap(self._frame)
        painter.drawPixmap(target, pixmap, pixmap.rect())

        # Draw preview overlay
        if self._preview_mode == "cutout" and self._mask is not None:
            self._draw_cutout(painter, target)
        elif self._preview_mode == "mask" and self._mask is not None:
            mask_pix = self._mask_to_pixmap(self._mask)
            painter.drawPixmap(target, mask_pix, mask_pix.rect())

        # Draw prompt points (points mode only)
        if self._edit_mode == "points":
            for i, pt in enumerate(self._points):
                cx = int(off_x + pt.x * self._scale)
                cy = int(off_y + pt.y * self._scale)
                color = _POINT_POS if pt.label == 1 else _POINT_NEG
                selected = (i == self._selected_point)
                radius = 8 if selected else 6
                # Outer ring
                ring = _POINT_RING if not selected else QColor(255, 255, 255, 255)
                painter.setPen(QPen(ring, 2.0 if selected else 1.5))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
                # Inner dot
                if not selected:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
                    painter.drawEllipse(cx - 2, cy - 2, 4, 4)

        # Brush cursor
        if self._edit_mode == "brush" and self._brush_hover is not None:
            self._draw_brush_cursor(painter)

        painter.end()

    def _draw_brush_cursor(self, painter: QPainter) -> None:
        """Draw a professional brush cursor: circle + center dot."""
        hx, hy = self._brush_hover
        r = self._brush_size
        erase = self._brush_erase or self._shift_held
        color = _BRUSH_ERASE if erase else _BRUSH_ADD

        # Outer circle
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(hx - r, hy - r, r * 2, r * 2)

        # Semi-transparent fill
        fill_color = QColor(240, 90, 90, 30) if erase else QColor(255, 255, 255, 20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill_color))
        painter.drawEllipse(hx - r, hy - r, r * 2, r * 2)

        # Center dot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(hx - 2, hy - 2, 4, 4)

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._frame is None:
            return
        # Pan with the middle button from any mode
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._edit_mode == "brush":
            pos = event.position().toPoint()
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self._painting = True
                self._painted = False
                self._shift_held = bool(
                    event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                )
                self.brush_press.emit()
                self._apply_brush(pos.x(), pos.y())
            return
        pos = event.position().toPoint()
        img_x, img_y = self._widget_to_image(pos.x(), pos.y())
        if img_x is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_point(img_x, img_y)
            if idx is not None:
                # Left-click on an existing point selects it (for deletion).
                self._selected_point = idx
                self.point_selected.emit(idx)
                self.update()
                return
            self._selected_point = None
            self.point_added.emit(img_x, img_y, 1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.point_added.emit(img_x, img_y, 0)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._frame is None:
            return
        pos = event.position().toPoint()
        if self._panning:
            delta = pos - self._pan_start
            self._pan_start = pos
            self._pan_by(delta.x(), delta.y())
            return
        if self._edit_mode == "brush":
            self._brush_hover = (pos.x(), pos.y())
            self._shift_held = bool(
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            )
            if self._painting:
                self._apply_brush(pos.x(), pos.y())
            self.update()
            return
        self._brush_hover = None

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self._edit_mode == "brush" and self._painting:
            self._painting = False
            self._shift_held = False
            if self._painted:
                self.mask_edited.emit()
            self.update()

    def mouseLeaveEvent(self, event) -> None:  # noqa: N802
        self._brush_hover = None
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 ** (delta / 120)
            self._zoom_at(event.position().toPoint(), factor)
            return
        direction = 1 if delta < 0 else -1
        self.frame_navigate.emit(direction)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.play_pause.emit()
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
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_point()
        else:
            super().keyPressEvent(event)

    def _hit_point(self, img_x: int, img_y: int) -> Optional[int]:
        """Return the index of a prompt point near (img_x, img_y) or None."""
        if not self._points:
            return None
        threshold = max(2.0, 8.0 / self._scale)
        best: Optional[int] = None
        best_d = threshold
        for i, pt in enumerate(self._points):
            d = ((pt.x - img_x) ** 2 + (pt.y - img_y) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = i
        return best

    def delete_selected_point(self) -> bool:
        """Remove the currently selected point and emit point_removed."""
        idx = self._selected_point
        if idx is None or idx >= len(self._points):
            return False
        removed = self._points[idx]
        self._points.pop(idx)
        self._selected_point = None
        self.point_removed.emit(removed)
        self.update()
        return True

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
        erase = self._brush_erase or self._shift_held
        if erase:
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

    def _mask_to_pixmap(self, mask: np.ndarray) -> QPixmap:
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask] = [
            self._mask_color.red(),
            self._mask_color.green(),
            self._mask_color.blue(),
            self._mask_color.alpha(),
        ]
        bytes_per_line = 4 * w
        image = QImage(rgba.data, w, h, bytes_per_line, QImage.Format.Format_ARGB32)
        return QPixmap.fromImage(image.copy())

    def _draw_cutout(self, painter: QPainter, target: QRect) -> None:
        """Compose the cut-out subject on a refined checkerboard."""
        if self._frame is None or self._mask is None:
            return
        h, w = self._mask.shape
        rgb = self._frame[:, :, :3].astype(np.uint8)
        mask = self._mask
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((xx // 8) + (yy // 8)) % 2 == 0
        r = np.where(checker, _CHECKER_A.red(), _CHECKER_B.red())
        g = np.where(checker, _CHECKER_A.green(), _CHECKER_B.green())
        b = np.where(checker, _CHECKER_A.blue(), _CHECKER_B.blue())
        bg = np.stack([r, g, b], axis=-1).astype(np.uint8)
        keep = mask[..., None]
        cut = rgb * keep.astype(np.uint8) + bg * (~keep).astype(np.uint8)
        pix = self._np_to_pixmap(cut)
        painter.drawPixmap(target, pix, pix.rect())
