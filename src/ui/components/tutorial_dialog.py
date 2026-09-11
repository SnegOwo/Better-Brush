"""Quick step-by-step tutorial dialog for first-time users."""

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

_STEPS = [
    (
        "1 · Import a video",
        "Drag and drop a video onto the window, or use File → Open Video (Ctrl+O).",
    ),
    (
        "2 · Choose a Preset (AMV / Human / Object)",
        "In the Content panel, pick a preset that matches your footage:\n"
        "• AMV — anime / cartoon outlines, emphasises hard edges.\n"
        "• Human — faces and people, preserves skin detail.\n"
        "• Object — general objects, good starting point for anything.",
    ),
    (
        "3 · Add points on your subject",
        "With the Points tool, left-click places a green marker (include) on the "
        "subject you want to cut out. Right-click places a red marker (exclude) "
        "on stuff SAM 2 should ignore.",
    ),
    (
        "4 · Create the first mask",
        "Press Segment Frame. SAM 2 turns your points into a crisp alpha matte "
        "for the current frame.",
    ),
    (
        "5 · Propagate across the timeline",
        "Use the Prop buttons (◀ Prop / Prop ▶) to step one frame at a time, "
        "or click Propagate to End to extend the mask through the entire video. "
        "Jump between keyframes and correct any that look off.",
    ),
    (
        "6 · Refine with the brush",
        "Press B to switch to the Brush tool: paint to add to the mask. "
        "Hold Shift while painting to erase. "
        "Space or P plays and pauses the clip.",
    ),
    (
        "7 · Export and save",
        "Ctrl+E opens the export dialog — H.264 MP4 with a solid "
        "background (green-screen, blue, etc.) is great for compositing. "
        "Ctrl+S saves your project, keeping every point and mask for later.",
    ),
    (
        "8 · Library & recode",
        "File → Media Library (Ctrl+L) shows all your exports. "
        "Right-click any video to recode it to a different format (MP4 → MOV ProRes, "
        "WebM, etc.) without re-rendering the mask.",
    ),
]


class TutorialDialog(QDialog):
    """Minimal numbered walkthrough with Skip/Next navigation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("How Better Brush works")
        self.setModal(True)
        self.setFixedWidth(520)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(12)

        self._title = QLabel("")
        self._title.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #d4d4d8;"
        )
        root.addWidget(self._title)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setStyleSheet("color: #a0a0a8; font-size: 12px; line-height: 1.5;")
        root.addWidget(self._body)

        self._count = QLabel("")
        self._count.setStyleSheet("color: #555560; font-size: 10px;")
        root.addWidget(self._count)

        actions = QHBoxLayout()
        actions.addStretch()
        self._btn_prev = QPushButton("Back")
        self._btn_prev.clicked.connect(self._go_prev)
        actions.addWidget(self._btn_prev)
        self._btn_next = QPushButton("Next")
        self._btn_next.setProperty("cssClass", "accent")
        self._btn_next.clicked.connect(self._go_next)
        actions.addWidget(self._btn_next)
        self._btn_skip = QPushButton("Skip")
        self._btn_skip.clicked.connect(self.reject)
        actions.addWidget(self._btn_skip)

        root.addLayout(actions)

        self._step = 0
        self._show_step()

    # ── Navigation ────────────────────────────────────────────────────────
    def _show_step(self) -> None:
        title, body = _STEPS[self._step]
        self._title.setText(title)
        self._body.setText(body)
        self._count.setText(f"Step {self._step + 1} of {len(_STEPS)}")
        self._btn_prev.setEnabled(self._step > 0)
        last = self._step == len(_STEPS) - 1
        self._btn_next.setText("Finish" if last else "Next")
        self._btn_skip.setVisible(not last)

    def _go_prev(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._show_step()

    def _go_next(self) -> None:
        if self._step < len(_STEPS) - 1:
            self._step += 1
            self._show_step()
        else:
            self.accept()