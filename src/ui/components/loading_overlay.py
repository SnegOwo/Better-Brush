"""Animated full-window loading overlay for long-running tasks."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel, QProgressBar, QPushButton, QWidget

_BG_OVERLAY = QColor(12, 12, 16, 175)
_TRACK = QColor(60, 60, 70)
_ARC = QColor(0, 180, 240)


class LoadingOverlay(QWidget):
    """Semi-transparent overlay with an animated spinner and task text.

    Child of the central widget; fills its parent's geometry and blocks
    interaction while a long-running worker is active.  Long tasks that
    report progress can opt into a determinate progress bar and a Cancel
    button via :meth:`set_progress_bar` / :meth:`enable_cancel`.
    """

    cancel_requested = Signal()

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

        # Determinate progress bar (hidden unless opted in by a task).
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setFixedWidth(240)
        self._bar.setStyleSheet(
            "QProgressBar { background: #2a2a30; border: none; border-radius: 4px; }"
            "QProgressBar::chunk { background: #3b82f6; border-radius: 4px; }"
        )
        self._bar.hide()

        # Optional Cancel button (hidden unless a cancellable task shows it).
        self._btn_cancel = QPushButton("Cancel", self)
        self._btn_cancel.setFixedWidth(80)
        self._btn_cancel.setStyleSheet(
            "QPushButton { background: #2e2e33; color: #d0d0d6; border: none;"
            "  border-radius: 4px; padding: 5px 12px; font-size: 11px; }"
            "QPushButton:hover { background: #3a3a40; }"
            "QPushButton:pressed { background: #45454d; }"
        )
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_cancel.hide()

        self.hide()

    # ── Public API ─────────────────────────────────────────────────────────
    def show_message(self, title: str = "Processing", sub: str = "") -> None:
        """Display the overlay over its parent widget."""
        self._title.setText(title)
        self._sub.setText(sub)
        self._bar.reset()
        self._bar.hide()
        self._btn_cancel.hide()
        self.reposition_overlay()
        self.raise_()
        self.show()
        self._timer.start()

    def set_progress(self, text: str) -> None:
        """Update the progress caption (keeps the overlay visible)."""
        self._sub.setText(text)
        self._reposition_labels()

    def set_progress_bar(self, maximum: int) -> None:
        """Show a determinate progress bar with the given maximum."""
        self._bar.setRange(0, max(maximum, 1))
        self._bar.setValue(0)
        self._bar.show()
        self._bar.raise_()
        self._btn_cancel.raise_()
        self._reposition_labels()

    def set_progress_value(self, value: int) -> None:
        """Advance the determinate progress bar."""
        self._bar.setValue(max(0, int(value)))

    def enable_cancel(self, enabled: bool = True) -> None:
        """Show / hide the Cancel button (emits ``cancel_requested``)."""
        self._btn_cancel.setVisible(enabled)
        self._btn_cancel.raise_()
        if enabled:
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
    def _on_cancel(self) -> None:
        self.cancel_requested.emit()

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
        if self._bar.isVisible():
            self._bar.move(cx - self._bar.width() // 2, base + 58)
        if self._btn_cancel.isVisible():
            self._btn_cancel.adjustSize()
            self._btn_cancel.move(cx - self._btn_cancel.width() // 2, base + 78)

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