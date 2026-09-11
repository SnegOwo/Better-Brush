"""Empty state overlay shown when no video is loaded."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QMouseEvent, QPainter
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget


class EmptyStateWidget(QWidget):
    """Full-screen overlay with branding, drop zone and open button."""

    open_clicked = Signal()
    file_dropped = Signal(str)  # file path

    _VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf", ".ts")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Branding
        self._title = "Better Brush"
        self._subtitle = "AI-powered rotoscoping"

        # Open button
        self._btn_open = QPushButton("Open Video")
        self._btn_open.setFixedWidth(180)
        self._btn_open.setFixedHeight(36)
        self._btn_open.setProperty("cssClass", "accent")
        self._btn_open.clicked.connect(self.open_clicked.emit)
        layout.addWidget(self._btn_open, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(8)

        self._drop_hint = "or drop a video file here"
        self._dragging = False

    # ── Painting ──────────────────────────────────────────────────────────
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), "#141418")

        cx = self.width() // 2
        cy = self.height() // 2

        # Title
        title_font = QFont("Segoe UI", 22, QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen("#d4d4d8")
        title_rect = self.rect().adjusted(0, -80, 0, -80)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, self._title)

        # Subtitle
        sub_font = QFont("Segoe UI", 12)
        painter.setFont(sub_font)
        painter.setPen("#8b8b94")
        sub_rect = self.rect().adjusted(0, -36, 0, -36)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, self._subtitle)

        # Drop hint (below the button area)
        hint_font = QFont("Segoe UI", 10)
        painter.setFont(hint_font)
        painter.setPen("#555560" if not self._dragging else "#4a9eff")
        hint_rect = self.rect().adjusted(0, 40, 0, 40)
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self._drop_hint)

        # Drop zone border when dragging
        if self._dragging:
            painter.setPen("#4a9eff")
            painter.setBrush(Qt.BrushStyle.NoBrush)
            m = 16
            painter.drawRoundedRect(m, m, self.width() - 2 * m, self.height() - 2 * m, 8, 8)

        painter.end()

    # ── Drag and drop ─────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if any(path.endswith(ext) for ext in self._VIDEO_EXTS):
                    event.acceptProposedAction()
                    self._dragging = True
                    self.update()
                    return

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        self.update()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._dragging = False
        self.update()
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if any(path.lower().endswith(ext) for ext in self._VIDEO_EXTS):
                self.file_dropped.emit(path)
                return
