"""Modal dialog: let the user draw a rectangle to guide subtitle/logo removal.

Auto-detection is heuristic and cannot always find always-on overlays (big
burned-in subtitles, logos that blend in).  When it fails, the user should be
able to say *where* the text / watermark is: they drag a rectangle on the
current frame, and the cleanup plate is built directly from that region.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.video_canvas import VideoCanvas

_KIND_TITLE = {"subtitles": "Remove Subtitles", "logo": "Remove Logo"}
_KIND_HINT = {
    "subtitles": "Drag a rectangle over the subtitles, then click Analyze.",
    "logo": "Drag a rectangle over the logo/watermark, then click Analyze.",
}


class RegionPickDialog(QDialog):
    """Ask the user to mark a region, or fall back to auto-detection.

    Attributes after ``exec()`` (when accepted):
      - ``choice`` : ``"region"`` | ``"auto"`` | ``"cancel"``
      - ``region`` : (x, y, w, h) in image coordinates or ``None``
    """

    def __init__(
        self,
        parent: QWidget | None,
        frame_rgb,
        kind: str = "subtitles",
    ) -> None:
        super().__init__(parent)
        self.choice = "cancel"
        self.region: tuple[int, int, int, int] | None = None

        self.setWindowTitle(_KIND_TITLE.get(kind, "Clean Up"))
        self.setMinimumSize(760, 540)
        self.resize(920, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        heading = QLabel(
            "Auto-detection could not find the "
            f"{'subtitles' if kind == 'subtitles' else 'logo'} on its own.\n"
            "Draw a tight rectangle around it, or let the app scan the whole video."
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        hint = QLabel(_KIND_HINT.get(kind, ""))
        hint.setStyleSheet("color: #78beff;")
        layout.addWidget(hint)

        self._canvas = VideoCanvas(self)
        self._canvas.set_frame(None if frame_rgb is None else frame_rgb.copy())
        self._canvas.set_preview_mode("frame")
        self._canvas.begin_region_pick(_KIND_HINT.get(kind, ""))
        self._canvas.region_selected.connect(self._on_region_selected)
        self._canvas.region_cancelled.connect(self._on_region_cancelled)
        layout.addWidget(self._canvas, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self._btn_analyze = QPushButton("Analyze this region")
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self._on_analyze)
        buttons.addWidget(self._btn_analyze)

        self._btn_auto = QPushButton("Auto-detect instead")
        self._btn_auto.clicked.connect(self._on_auto)
        buttons.addWidget(self._btn_auto)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self._on_cancel)
        buttons.addWidget(self._btn_cancel)

        layout.addLayout(buttons)

    # ── slots ────────────────────────────────────────────────────────────
    def _on_region_selected(self, x: int, y: int, w: int, h: int) -> None:
        self.region = (int(x), int(y), int(w), int(h))
        self._btn_analyze.setEnabled(True)

    def _on_region_cancelled(self) -> None:
        self.region = None
        self._btn_analyze.setEnabled(False)

    def _on_analyze(self) -> None:
        if self.region is None:
            return
        self.choice = "region"
        self.accept()

    def _on_auto(self) -> None:
        self.choice = "auto"
        self.accept()

    def _on_cancel(self) -> None:
        self.choice = "cancel"
        self.reject()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
            return
        super().keyPressEvent(event)