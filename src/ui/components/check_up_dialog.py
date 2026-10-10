"""Check-up dialog: a plain-language status list for the cleanup pipeline."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.processing.plates import DirtyPlate, inpaint_engine_status


class CheckUpDialog(QDialog):
    """Show what the cleanup engine detected, in which mode it will run,
    and which inpainting backend is active.

    ``plates`` is the list of active :class:`DirtyPlate` instances from the
    main window (may be empty — the dialog then explains that nothing is
    armed yet).
    """

    def __init__(
        self,
        plates: list[DirtyPlate],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Clean up check-up")
        self.setModal(True)
        self.setFixedWidth(560)
        self.setMinimumHeight(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(12)

        title = QLabel("Clean up check-up")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(title)

        body = QLabel(
            "A plain-language look at what Better Brush will remove, how it "
            "will remove it, and whether the heavy AI inpainting engine is "
            "available."
        )
        body.setWordWrap(True)
        body.setStyleSheet("color: #a0a0a8; font-size: 12px; line-height: 1.5;")
        root.addWidget(body)

        root.addSpacing(4)

        engine = QLabel(f"● Inpainting engine: {inpaint_engine_status()}")
        engine.setWordWrap(True)
        engine.setStyleSheet("color: #d4d4d8; font-size: 12px;")
        root.addWidget(engine)

        engine_note = QLabel(
            "LaMa (AI) reproduces fine line-art texture on busy anime frames. "
            "If it is not available the app silently falls back to the "
            "built-in OpenCV donor/Telea repair — no action needed."
        )
        engine_note.setWordWrap(True)
        engine_note.setStyleSheet("color: #7a7a84; font-size: 11px;")
        root.addWidget(engine_note)

        root.addSpacing(6)

        if not plates:
            empty = QLabel(
                "● No cleanup regions are armed yet.\n\n"
                "Press Remove Subtitles / Remove Logo / Remove Both and, if "
                "prompted, drag a box over the text or watermark."
            )
            empty.setWordWrap(True)
            empty.setStyleSheet("color: #a0a0a8; font-size: 12px;")
            root.addWidget(empty)
        else:
            root.addWidget(self._heading("Regions to clean"))
            for idx, plate in enumerate(plates, start=1):
                root.addWidget(
                    self._plate_row(idx, plate)
                )

        root.addStretch(1)

        actions = QHBoxLayout()
        actions.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setProperty("cssClass", "accent")
        btn_close.clicked.connect(self.accept)
        actions.addWidget(btn_close)
        root.addLayout(actions)

    def _heading(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #d4d4d8;"
            "padding-top: 4px; border-top: 1px solid #2a2a30;"
        )
        return lbl

    def _plate_row(self, idx: int, plate: DirtyPlate) -> QLabel:
        kind = "Subtitles" if plate.kind == "text" else "Logo"
        x, y, w, h = plate.region
        loc = f"at ({x}, {y}) {w}×{h}px"
        if kind == "Subtitles" and plate.track:
            loc += ", tracking the band frame-by-frame"
        if plate.ref_mask is not None:
            loc += ", full-coverage reference mask"
        mode = {
            "inpaint": "per-frame inpainting",
            "subtitles": "flicker plate",
            "logo": "static gate",
        }.get(plate.mode, plate.mode)
        row = QLabel(f"{idx}. {kind}: {mode} {loc}")
        row.setWordWrap(True)
        row.setStyleSheet("color: #a0a0a8; font-size: 12px; line-height: 1.5;")
        return row