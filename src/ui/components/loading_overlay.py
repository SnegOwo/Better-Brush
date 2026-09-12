"""Animated full-window loading overlay for long-running tasks."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget

_BG_OVERLAY = QColor(12, 12, 16, 175)
_TRACK = QColor(60, 60, 70)
_ARC = QColor(0, 180, 240)


class LoadingOverlay(QWidget):
    """Semi-transparent overlay with an animated spinner and task text.

    Child of the central widget; fills its parent's geometry and blocks
    interaction while a long-running worker is active.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)

        self._title = QLabel("Processing", self)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            "color: #e6e6ea; font-size: 15px; font-weight: 600; background: transparent;"
        )
        self._sub = QLabel("", self)
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub.setStyleSheet(
            "color: #9a9aa4; font-size: 12px; background: transparent;"
        )

        self.hide()

    # ── Public API ─────────────────────────────────────────────────────────
    def show_message(self, title: str = "Processing", sub: str = "") -> None:
        """Display the overlay over its parent widget."""
        self._title.setText(title)
        self._sub.setText(sub)
        self.reposition_overlay()
        self.raise_()
        self.show()
        self._timer.start()

    def set_progress(self, text: str) -> None:
        """Update the progress caption (keeps the overlay visible)."""
        self._sub.setText(text)
        self._reposition_labels()

    def hide_message(self) -> None:
        self._timer.stop()
        self.hide()

    def reposition_overlay(self) -> None:
        """Resize to cover the parent and re-center labels."""
        self.setGeometry(self.parentWidget().rect())
        self._reposition_labels()

    def is_active(self) -> bool:
        return self.isVisible()

    # ── Internals ──────────────────────────────────────────────────────────
    def _advance(self) -> None:
        self._angle = (self._angle + 9) % 360
        self.update()

    def _reposition_labels(self) -> None:
        cx = self.width() // 2
        base = self.height() // 2
        self._title.adjustSize()
        self._title.move(cx - self._title.width() // 2, base + 4)
        self._sub.adjustSize()
        self._sub.move(cx - self._sub.width() // 2, base + 30)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BG_OVERLAY)

        cx = self.width() // 2
        cy = self.height() // 2 - 70
        radius = 24

        # Track ring
        painter.setPen(QPen(_TRACK, 5))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Animated arc
        pen = QPen(_ARC, 5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(
            cx - radius,
            cy - radius,
            radius * 2,
            radius * 2,
            self._angle * 16,
            90 * 16,
        )
        painter.end()
