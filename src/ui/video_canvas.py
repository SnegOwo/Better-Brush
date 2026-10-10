"""Interactive video canvas: display frame, mask overlay, prompt points."""

from __future__ import annotations

import logging

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

# ── Visual constants ──────────────────────────────────────────────────────
_BG = QColor(20, 20, 24)
#: Half-width of the patch the spill eyedropper medians over.
_SPILL_PICK_RADIUS = 2
_POINT_POS = QColor(72, 199, 142)  # muted green
_POINT_NEG = QColor(230, 80, 80)  # muted red
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
    edit_mode_changed = Signal(str)
    brush_size_changed = Signal(int)
    # Brush editing
    brush_press = Signal()  # emitted at the start of a brush stroke
    mask_edited = Signal()  # emitted after a brush stroke modified the mask
    # Region picking (Clean Up guide)
    region_selected = Signal(int, int, int, int)  # x, y, w, h (image coords)
    region_cancelled = Signal()
    # Eyedropper (spill cleaner)
    color_picked = Signal(int, int, int)  # r, g, b
    color_pick_cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._mask: np.ndarray | None = None  # bool HxW
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
        self._brush_hover: tuple[int, int] | None = None
        self._painting = False
        self._painted = False
        self._selected_point: int | None = None  # index into _points
        # Region picking (Clean Up guide)
        self._picking_region = False
        self._pick_start: tuple[int, int] | None = None
        self._pick_rect: tuple[int, int, int, int] | None = None  # image coords
        self._region_hint = "Drag a rectangle over the region"
        # Eyedropper (spill cleaner)
        self._picking_color = False
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background: transparent;")

    # ------------------------------------------------------------------ #
    # Setters
    # ------------------------------------------------------------------ #
    def set_frame(self, frame: np.ndarray | None) -> None:
        self._frame = frame
        self.update()

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Set the matte: bool for binary masks, float32 0..1 for soft alpha."""
        if mask is None:
            self._mask = None
        elif mask.dtype == bool:
            self._mask = mask
        else:
            self._mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        self.update()

    def set_points(self, points: list[PointPrompt]) -> None:
        self._points = list(points)
        if self._selected_point is not None and self._selected_point >= len(
            self._points
        ):
            self._selected_point = None
        self.update()

    def clear_mask(self) -> None:
        self._mask = None
        self.update()

    def clear_points(self) -> None:
        self._points = []
        self._selected_point = None
        self.update()

    def get_mask(self) -> np.ndarray | None:
        """Return a copy of the current mask (bool HxW) or None."""
        return None if self._mask is None else self._mask.copy()

    # ------------------------------------------------------------------ #
    # Tools / preview
    # ------------------------------------------------------------------ #
    def set_edit_mode(self, mode: str) -> None:
        mode = mode if mode in ("points", "brush") else "points"
        if mode == self._edit_mode:
            return
        self._edit_mode = mode
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.edit_mode_changed.emit(mode)
        self.update()

    def get_edit_mode(self) -> str:
        """Return the active tool: ``"points"`` or ``"brush"``."""
        return self._edit_mode

    def set_brush_size(self, size: int) -> None:
        size = max(1, min(int(size), 500))
        if size == self._brush_size:
            return
        self._brush_size = size
        self.brush_size_changed.emit(size)
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
    # Region picking (Clean Up guide)
    # ------------------------------------------------------------------ #
    def begin_region_pick(self, hint: str = "") -> None:
        """Enter 'drag a rectangle' mode (no point/brush interactions)."""
        self._picking_region = True
        self._pick_start = None
        self._pick_rect = None
        if hint:
            self._region_hint = hint
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def end_region_pick(self) -> None:
        self._picking_region = False
        self._pick_start = None
        self._pick_rect = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def is_picking_region(self) -> bool:
        return self._picking_region

    # ------------------------------------------------------------------ #
    # Eyedropper (spill cleaner)
    # ------------------------------------------------------------------ #
    def begin_color_pick(self) -> None:
        """Enter eyedropper mode: the next click samples a frame pixel."""
        self._picking_color = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def end_color_pick(self) -> None:
        """Leave eyedropper mode and restore the normal cursor."""
        self._picking_color = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def is_picking_color(self) -> bool:
        """Return whether the eyedropper is waiting for a click."""
        return self._picking_color

    def pixel_at(self, img_x: int, img_y: int) -> tuple[int, int, int] | None:
        """Return the RGB value at image coordinates, or ``None`` if outside.

        A 5x5 neighbourhood is reduced by its per-channel median rather than
        reading the single pixel: video is full of compression speckle and
        mosquito noise, and one stray pixel is enough to aim the spill cleaner
        at the wrong hue. The median ignores both without smearing edges the way
        an average would.
        """
        if self._frame is None:
            return None
        height, width = self._frame.shape[:2]
        if not 0 <= img_x < width or not 0 <= img_y < height:
            return None
        radius = _SPILL_PICK_RADIUS
        x0, x1 = max(0, img_x - radius), min(width, img_x + radius + 1)
        y0, y1 = max(0, img_y - radius), min(height, img_y + radius + 1)
        patch = self._frame[y0:y1, x0:x1, :3].reshape(-1, 3)
        if patch.size == 0:
            return None
        return (
            int(np.median(patch[:, 0])),
            int(np.median(patch[:, 1])),
            int(np.median(patch[:, 2])),
        )

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
    def paintEvent(self, event) -> None:
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
                selected = i == self._selected_point
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

        # Region-picking overlay (Clean Up guide)
        if self._picking_region:
            self._draw_region_pick(painter, off_x, off_y)

        # Eyedropper hint (spill cleaner)
        if self._picking_color:
            self._draw_color_pick_hint(painter, off_x, off_y)

        painter.end()

    def _draw_color_pick_hint(self, painter: QPainter, off_x: int, off_y: int) -> None:
        """Draw the eyedropper hint while the user is choosing a spill color."""
        painter.setPen(QPen(QColor(255, 255, 255, 235)))
        painter.drawText(
            QRect(off_x + 10, off_y + 10, max(220, int(self.width() * 0.6)), 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Click the color fringe to sample it  (Esc to cancel)",
        )

    def _draw_region_pick(self, painter: QPainter, off_x: int, off_y: int) -> None:
        """Draw the dotted rectangle + hint while the user drags a region."""
        # Hint text pinned to the top-left of the frame.
        painter.setPen(QPen(QColor(255, 255, 255, 235)))
        painter.drawText(
            QRect(off_x + 10, off_y + 10, max(200, int(self.width() * 0.6)), 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._region_hint,
        )
        if self._pick_rect is None:
            return
        ax0, ay0, ax1, ay1 = self._pick_rect
        x0 = int(off_x + ax0 * self._scale)
        y0 = int(off_y + ay0 * self._scale)
        x1 = int(off_x + (ax1 + 1) * self._scale)
        y1 = int(off_y + (ay1 + 1) * self._scale)
        painter.setPen(QPen(QColor(120, 190, 255, 255), 2, Qt.PenStyle.DashLine))
        painter.setBrush(QColor(120, 190, 255, 40))
        painter.drawRect(x0, y0, x1 - x0, y1 - y0)
        painter.setBrush(QColor(120, 190, 255, 255))
        painter.drawEllipse(x0 - 3, y0 - 3, 6, 6)

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
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._frame is None:
            return
        if self._picking_color:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position().toPoint()
                img_x, img_y = self._widget_to_image(pos.x(), pos.y())
                if img_x is not None:
                    rgb = self.pixel_at(img_x, img_y)
                    if rgb is not None:
                        self.end_color_pick()
                        self.color_picked.emit(*rgb)
            return
        if self._picking_region:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position().toPoint()
                img = self._widget_to_image(pos.x(), pos.y())
                if img[0] is not None:
                    self._pick_start = (img[0], img[1])
                    self._pick_rect = (img[0], img[1], img[0], img[1])
                    self.update()
            return
        # Pan with the middle button from any mode
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._edit_mode == "brush":
            pos = event.position().toPoint()
            if event.button() in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
            ):
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._frame is None:
            return
        pos = event.position().toPoint()
        if self._picking_region:
            if self._pick_start is not None:
                img = self._widget_to_image(pos.x(), pos.y())
                if img[0] is not None:
                    sx, sy = self._pick_start
                    self._pick_rect = (
                        min(sx, img[0]),
                        min(sy, img[1]),
                        max(sx, img[0]),
                        max(sy, img[1]),
                    )
                    self.update()
            return
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._picking_region:
            rect = self._pick_rect
            self._pick_start = None
            if rect is not None and event.button() == Qt.MouseButton.LeftButton:
                ax0, ay0, ax1, ay1 = rect
                # Ignore accidental clicks / tiny drags.
                if ax1 - ax0 >= 8 and ay1 - ay0 >= 8:
                    self.region_selected.emit(ax0, ay0, ax1 - ax0 + 1, ay1 - ay0 + 1)
            return
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

    def mouseLeaveEvent(self, event) -> None:
        self._brush_hover = None
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 ** (delta / 120)
            self._zoom_at(event.position().toPoint(), factor)
            return
        direction = 1 if delta < 0 else -1
        self.frame_navigate.emit(direction)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape and self._picking_color:
            self.end_color_pick()
            self.color_pick_cancelled.emit()
        elif key == Qt.Key.Key_Escape and self._picking_region:
            self._pick_start = None
            self._pick_rect = None
            self.region_cancelled.emit()
            self.update()
        elif key == Qt.Key.Key_Space:
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

    def _hit_point(self, img_x: int, img_y: int) -> int | None:
        """Return the index of a prompt point near (img_x, img_y) or None."""
        if not self._points:
            return None
        threshold = max(2.0, 8.0 / self._scale)
        best: int | None = None
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
    def _widget_to_image(self, wx: int, wy: int) -> tuple[int | None, int | None]:
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
        if mask.dtype == bool:
            rgba[mask] = [
                self._mask_color.red(),
                self._mask_color.green(),
                self._mask_color.blue(),
                self._mask_color.alpha(),
            ]
        else:
            # Soft matte: partial coverage becomes partial colour alpha.
            coverage = mask.astype(np.float32)
            rgba[:, :, 0] = int(self._mask_color.red())
            rgba[:, :, 1] = int(self._mask_color.green())
            rgba[:, :, 2] = int(self._mask_color.blue())
            rgba[:, :, 3] = (
                coverage * self._mask_color.alpha()
            ).astype(np.uint8)
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
        # Soft mattes blend partially-covered pixels against the checkerboard.
        keep = mask.astype(np.float32)[..., None]
        cut = rgb * keep + bg * (1.0 - keep)
        pix = self._np_to_pixmap(cut.astype(np.uint8))
        painter.drawPixmap(target, pix, pix.rect())
