"""Professional timeline widget for frame navigation, keyframe display, and
in/out range selection (Premiere-style)."""

from __future__ import annotations

import logging
import math

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# ── Visual constants ──────────────────────────────────────────────────────
_BG = QColor(22, 22, 26)
_RULER_BG = QColor(28, 28, 32)
_RULER_TICK = QColor(55, 55, 60)
_RULER_TICK_MAJOR = QColor(70, 70, 76)
_RULER_TEXT = QColor(100, 100, 108)
_TRACK_BG = QColor(22, 22, 26)
_MASKED_BAND = QColor(37, 99, 235, 140)
_KEYFRAME_COLOR = QColor(245, 158, 11)
_PLAYHEAD_COLOR = QColor(239, 68, 68)
_FRAME_TEXT = QColor(120, 120, 128)

_SEL_FILL = QColor(59, 130, 246, 68)
_SEL_BORDER = QColor(59, 130, 246, 225)

_RULER_H = 18  # height of the ruler area
_TRACK_Y = 20  # y offset where the track starts
_HANDLE_W = 5  # half-width of the selection edge grabbers

_INCLUSIVE_STEP = 1  # selection is inclusive: [start, end]


class Timeline(QWidget):
    """Horizontal timeline with ruler, playhead, keyframe markers, masked
    band, and an export range selection (shift-drag, I/O keys)."""

    frame_selected = Signal(int)  # frame index clicked
    playhead_moved = Signal(int)  # frame index dragged
    selection_changed = Signal(int, int)  # inclusive [start, end]; (-1, -1) = cleared

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(72)
        self.setMaximumHeight(100)
        self.setFixedHeight(84)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._total_frames = 0
        self._current_frame = 0
        self._keyframes: set[int] = set()
        self._masked_frames: set[int] = set()
        self._dragging = False
        self._sel_start: int | None = None
        self._sel_end: int | None = None
        self._sel_drag: str | None = None  # "create" | "move" | "left" | "right"
        self._drag_anchor_frame = 0
        self._drag_orig = (0, 0)
        self.setStyleSheet("background: transparent;")
        self.setToolTip(
            "Timeline range selection (export in/out):\n"
            "Shift+drag to draw/resize/move a range\n"
            "I = set in-point · O = set out-point · X = clear"
        )

    # ------------------------------------------------------------------ #
    # Setters
    # ------------------------------------------------------------------ #
    def set_duration(self, total_frames: int) -> None:
        self._total_frames = max(1, total_frames)
        self._clamp_selection()
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

    def set_masked_frames(self, frames: set[int]) -> None:
        self._masked_frames = set(frames)
        self.update()

    # ------------------------------------------------------------------ #
    # Selection (in/out range)
    # ------------------------------------------------------------------ #
    def set_selection(self, start: int, end: int) -> None:
        if self._total_frames <= 0:
            return
        s, e = sorted((start, end))
        s = max(0, min(s, self._total_frames - 1))
        e = max(0, min(e, self._total_frames - 1))
        self._sel_start, self._sel_end = s, e
        self.selection_changed.emit(s, e)
        self.update()

    def clear_selection(self) -> None:
        self._sel_start = self._sel_end = None
        self.selection_changed.emit(-1, -1)
        self.update()

    def get_selection(self) -> tuple[int, int] | None:
        """Return inclusive (start, end) or None if no selection set."""
        if self._sel_start is None or self._sel_end is None:
            return None
        return (self._sel_start, self._sel_end)

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(self.rect(), _BG)

        if self._total_frames <= 1:
            painter.setPen(_FRAME_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No video")
            painter.end()
            return

        track_top, track_h = self._track_geometry(h)

        # ── Ruler ─────────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, _RULER_H, _RULER_BG)
        self._draw_ruler(painter, w)

        # Right-aligned hint about the export-range keys (discoverability)
        if w > 620:
            hint_font = QFont("Segoe UI", 7)
            painter.setFont(hint_font)
            painter.setPen(_RULER_TEXT)
            painter.drawText(
                QRect(0, 0, w - 14, _RULER_H),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                "Shift+drag: range · I/O: in/out · X: clear",
            )

        # ── Track background ──────────────────────────────────────────────
        painter.fillRect(0, track_top, w, track_h, _TRACK_BG)

        # Top/bottom track borders
        painter.setPen(QPen(QColor(35, 35, 40), 1))
        painter.drawLine(0, track_top, w, track_top)
        painter.drawLine(0, track_top + track_h, w, track_top + track_h)

        # ── Masked frames band ────────────────────────────────────────────
        if self._masked_frames:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_MASKED_BAND))
            for mf in self._masked_frames:
                x = self._frame_to_x(mf, w)
                painter.drawRect(x, track_top, max(2, 1), track_h)

        # ── Selection overlay (in/out range) ─────────────────────────────
        if self._sel_start is not None and self._sel_end is not None:
            sx0 = self._frame_to_x(self._sel_start, w)
            sx1 = self._frame_to_x(self._sel_end, w)
            sel = QRect(sx0, track_top, max(1, sx1 - sx0), track_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_SEL_FILL))
            painter.drawRect(sel)
            painter.setPen(QPen(_SEL_BORDER, 1))
            painter.drawRect(sel.adjusted(0, 0, -1, -1))

            # Edge grabbers
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_SEL_BORDER))
            painter.drawRect(sx0 - _HANDLE_W, track_top, _HANDLE_W * 2, track_h)
            painter.drawRect(sx1 - _HANDLE_W, track_top, _HANDLE_W * 2, track_h)

        # ── Keyframe diamonds ─────────────────────────────────────────────
        if self._keyframes:
            diamond_size = 5
            painter.setPen(QPen(_KEYFRAME_COLOR, 1))
            painter.setBrush(QBrush(_KEYFRAME_COLOR))
            for kf in self._keyframes:
                x = self._frame_to_x(kf, w)
                cy = track_top + 6
                diamond = QPolygon(
                    [
                        QPoint(x, cy - diamond_size),
                        QPoint(x + diamond_size, cy),
                        QPoint(x, cy + diamond_size),
                        QPoint(x - diamond_size, cy),
                    ]
                )
                painter.drawPolygon(diamond)

        # ── Playhead ──────────────────────────────────────────────────────
        ph_x = self._frame_to_x(self._current_frame, w)

        # Playhead line (full track height)
        painter.setPen(QPen(_PLAYHEAD_COLOR, 1.5))
        painter.drawLine(ph_x, track_top, ph_x, track_top + track_h)

        # Playhead handle (small triangle at top of ruler)
        handle_size = 5
        handle = QPolygon(
            [
                QPoint(ph_x - handle_size, 0),
                QPoint(ph_x + handle_size, 0),
                QPoint(ph_x, handle_size + 2),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_PLAYHEAD_COLOR))
        painter.drawPolygon(handle)

        # ── Frame counter (bottom) ────────────────────────────────────────
        counter_y = h - 14
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.setPen(_FRAME_TEXT)
        painter.drawText(
            8,
            counter_y,
            f"Frame {self._current_frame} / {self._total_frames - 1}",
        )

        # Right-aligned percentage / masked count
        right_text = ""
        if self._masked_frames:
            pct = len(self._masked_frames) / self._total_frames * 100
            right_text = f"Masked: {pct:.0f}%"
        if self._sel_start is not None and self._sel_end is not None:
            n = self._sel_end - self._sel_start + 1
            suffix = f"   Sel: {self._sel_start}-{self._sel_end} ({n}f)"
            right_text = (right_text + suffix).strip()
        if right_text:
            painter.drawText(w - 260, counter_y, right_text)

        painter.end()

    def _draw_ruler(self, painter: QPainter, w: int) -> None:
        """Draw tick marks and frame numbers in the ruler area."""
        if self._total_frames <= 0:
            return

        # Calculate nice tick interval
        # We want roughly 5-15 major ticks across the width
        raw_step = self._total_frames / max(1, w / 80)
        magnitude = 10 ** int(math.log10(max(1, raw_step)))
        if raw_step / magnitude < 2:
            major_step = int(magnitude)
        elif raw_step / magnitude < 5:
            major_step = int(2 * magnitude)
        else:
            major_step = int(5 * magnitude)
        major_step = max(1, major_step)

        minor_step = max(1, major_step // 5)

        font = QFont("Segoe UI", 7)
        painter.setFont(font)

        for f in range(0, self._total_frames, minor_step):
            x = self._frame_to_x(f, w)
            is_major = f % major_step == 0

            if is_major:
                # Major tick
                painter.setPen(QPen(_RULER_TICK_MAJOR, 1))
                painter.drawLine(x, _RULER_H - 6, x, _RULER_H)
                # Frame number
                painter.setPen(_RULER_TEXT)
                painter.drawText(x + 3, _RULER_H - 3, str(f))
            else:
                # Minor tick
                painter.setPen(QPen(_RULER_TICK, 1))
                painter.drawLine(x, _RULER_H - 3, x, _RULER_H)

    def _frame_to_x(self, frame_idx: int, w: int) -> int:
        return int((frame_idx / max(1, self._total_frames - 1)) * (w - 1))

    def _track_geometry(self, h: int) -> tuple[int, int]:
        track_top = _RULER_H + 2
        track_h = h - track_top - 18  # leave room for frame counter
        return track_top, track_h

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._total_frames <= 1 or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.pos().x()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._start_selection_edit(x, event.pos().y())
            event.accept()
            return
        self._dragging = True
        self._seek_to(x)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._sel_drag is not None:
            self._apply_selection_drag(self._frame_at_x(event.pos().x()))
            return
        if self._dragging:
            self._seek_to(event.pos().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False
        self._sel_drag = None

    def keyPressEvent(self, event) -> None:
        key = event.key()
        cur = self._current_frame
        if key == Qt.Key.Key_I:  # set in-point at playhead
            if self._sel_start is None or self._sel_end is None:
                self.set_selection(cur, cur)
            else:
                self.set_selection(cur, self._sel_end)
        elif key == Qt.Key.Key_O:  # set out-point at playhead
            if self._sel_start is None or self._sel_end is None:
                self.set_selection(cur, cur)
            else:
                self.set_selection(self._sel_start, cur)
        elif key == Qt.Key.Key_X:  # clear range
            self.clear_selection()
        else:
            super().keyPressEvent(event)

    # ── Selection editing ─────────────────────────────────────────────────
    def _start_selection_edit(self, x: int, y: int) -> None:
        track_top, track_h = self._track_geometry(self.height())
        if y < track_top or y > track_top + track_h:
            return
        mode = self._hit_handle_or_body(x)
        if mode in ("left", "right"):
            self._sel_drag = mode
            self._drag_anchor_frame = self._frame_at_x(x)
        elif mode == "move":
            self._sel_drag = "move"
            self._drag_orig = (self._sel_start or 0, self._sel_end or 0)
            self._drag_anchor_frame = self._frame_at_x(x)
        else:
            self._sel_drag = "create"
            f0 = self._frame_at_x(x)
            self._sel_start = self._sel_end = f0
            self._drag_anchor_frame = f0
            self.selection_changed.emit(f0, f0)
            self.update()

    def _hit_handle_or_body(self, x: int) -> str | None:
        """Return "left", "right", "move", or None for a click at x."""
        if self._sel_start is None or self._sel_end is None:
            return None
        w = self.width()
        sx0 = self._frame_to_x(self._sel_start, w)
        sx1 = self._frame_to_x(self._sel_end, w)
        if abs(x - sx0) <= _HANDLE_W + 2:
            return "left"
        if abs(x - sx1) <= _HANDLE_W + 2:
            return "right"
        if sx0 <= x <= sx1:
            return "move"
        return None

    def _apply_selection_drag(self, f: int) -> None:
        if self._sel_drag is None or self._sel_start is None or self._sel_end is None:
            return
        f = max(0, min(f, self._total_frames - 1))
        if self._sel_drag == "create":
            self._sel_start = min(self._drag_anchor_frame, f)
            self._sel_end = max(self._drag_anchor_frame, f)
        elif self._sel_drag == "left":
            self._sel_start = min(f, self._sel_end)
        elif self._sel_drag == "right":
            self._sel_end = max(f, self._sel_start)
        elif self._sel_drag == "move":
            delta = f - self._drag_anchor_frame
            orig_s, orig_e = self._drag_orig
            span = orig_e - orig_s
            new_s = max(0, min(orig_s + delta, self._total_frames - 1 - span))
            self._sel_start, self._sel_end = new_s, new_s + span
        self.selection_changed.emit(self._sel_start, self._sel_end)
        self.update()

    def _clamp_selection(self) -> None:
        if self._sel_start is not None and self._total_frames > 0:
            self._sel_start = max(0, min(self._sel_start, self._total_frames - 1))
            self._sel_end = max(0, min(self._sel_end, self._total_frames - 1))

    def _frame_at_x(self, x: int) -> int:
        ratio = max(0.0, min(1.0, x / self.width()))
        return int(ratio * (self._total_frames - 1))

    def _seek_to(self, x: int) -> None:
        ratio = max(0.0, min(1.0, x / self.width()))
        idx = int(ratio * (self._total_frames - 1))
        self.set_current_frame(idx)
        self.frame_selected.emit(idx)
