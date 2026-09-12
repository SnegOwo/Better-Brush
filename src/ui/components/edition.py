"""Edition helpers: the Pro vs Lite comparison dialog and a first-launch
Lite notice (WinRAR-style, shown exactly once, never again)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core import licensing


def buy_url() -> QUrl:
    return QUrl.fromUserInput("https://sneg.dev/BetterBrush")


# Feature matrix used by the comparison dialog (and only the comparison).
_FEATURES = [
    ("SAM 2 segmentation, brush & keyframes", True, True),
    ("Multi-object projects", False, True),
    ("H.264 MP4 export with chroma-key backgrounds", True, True),
    ("PNG alpha sequence export", True, True),
    ("ProRes 4444 & QuickTime Animation (MOV)", True, True),
    ("Export up to 1920×1080", True, True),
    ("Resize exports above 1080p (up to 4K/8K)", False, True),
    ("Original audio in exports", False, True),
    ("Library recoding (MP4 ↔ MOV ↔ WebM)", False, True),
    ("Watermark on video exports", True, False),
]

_ROWS_GOOD = ("✅", "❌")


class ProFeaturesDialog(QDialog):
    """Clean, one-screen 'what is locked behind Pro' comparison."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Better Brush Pro vs Lite")
        self.setModal(True)
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18)
        root.setSpacing(10)

        head = QLabel("What is included in each edition?")
        head.setStyleSheet("font-size: 15px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(head)

        cols = QHBoxLayout()
        cols.setSpacing(10)
        cols.addWidget(self._col_label("FEATURE"), 1)
        cols.addWidget(self._col_label("LITE (free)"), 0)
        cols.addWidget(self._col_label("PRO (€5)"), 0)
        root.addLayout(cols)

        for name, lite, pro in _FEATURES:
            row = QHBoxLayout()
            row.setSpacing(10)
            lbl = QLabel(name)
            lbl.setStyleSheet(
                "color: #c4c4cc; font-size: 12px; background: transparent;"
            )
            lbl.setWordWrap(True)
            row.addWidget(lbl, 1)
            row.addWidget(self._mark(lite), 0, alignment=Qt.AlignmentFlag.AlignHCenter)
            row.addWidget(self._mark(pro), 0, alignment=Qt.AlignmentFlag.AlignHCenter)
            root.addLayout(row)

        root.addSpacing(4)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #2e2e33; max-height: 1px;")
        root.addWidget(sep)

        note = QLabel(
            f"Prices: <b>€{licensing.PRO_PRICE_EUR} one-time</b> per machine. Offline "
            "activation, no account. Buy at <b>sneg.dev/BetterBrush</b>."
            if not licensing.is_pro()
            else "Your copy is <b>Pro</b> — every feature is unlocked. "
            "Activation is offline and stays on this machine."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8b8b94; font-size: 12px; background: transparent;")
        root.addWidget(note)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        if not licensing.is_pro():
            btn_buy = QPushButton(f"Buy Pro (€{licensing.PRO_PRICE_EUR})")
            btn_buy.setToolTip("https://sneg.dev/BetterBrush")
            btn_buy.clicked.connect(
                lambda: (QDesktopServices.openUrl(buy_url()), self.accept())
            )
            btn_row.addWidget(btn_buy)
        btn_ok = QPushButton("Close")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

    @staticmethod
    def _col_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #555560; font-size: 10px; font-weight: 600; "
            "letter-spacing: 1px; background: transparent;"
        )
        return lbl

    @staticmethod
    def _mark(ok: bool) -> QLabel:
        lbl = QLabel(_ROWS_GOOD[0] if ok else _ROWS_GOOD[1])
        lbl.setStyleSheet("background: transparent; font-size: 12px;")
        return lbl


class LiteNoticeDialog(QDialog):
    """One-time, first-launch 'you are on the free Lite edition' nudge.

    Shown a single time per installation (Lite only) and never again — the
    app stays quiet afterwards.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Better Brush — Lite")
        self.setModal(True)
        self.setFixedWidth(430)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 22)
        root.setSpacing(12)

        title = QLabel("You're using Better Brush <b>Lite</b> — free")
        title.setStyleSheet("font-size: 15px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(title)

        body = QLabel(
            "The <b>Pro</b> upgrade is a one-time payment of "
            f"<b>€{licensing.PRO_PRICE_EUR}</b> and unlocks multi-object editing, "
            "4K exports, original audio, watermark-free video and Library recoding.\n\n"
            "Lite stays free forever with no expiry."
        )
        body.setWordWrap(True)
        body.setStyleSheet("color: #a0a0a8; font-size: 12px; background: transparent;")
        root.addWidget(body)

        show = QLabel("sneg.dev/BetterBrush")
        show.setStyleSheet("color: #4a9eff; font-size: 11px; background: transparent;")
        root.addWidget(show)

        btns = QHBoxLayout()
        btns.addStretch()
        btn_whats_in = QPushButton("What's in Pro?")
        btn_whats_in.setToolTip("See the full feature comparison")
        btn_whats_in.clicked.connect(self._on_whats_in)
        btns.addWidget(btn_whats_in)
        btn_later = QPushButton("Keep Lite")
        btn_later.setDefault(True)
        btn_later.clicked.connect(self.accept)
        btns.addWidget(btn_later)
        btn_buy = QPushButton(f"Buy Pro (€{licensing.PRO_PRICE_EUR})")
        btn_buy.setProperty("cssClass", "accent")
        btn_buy.clicked.connect(self._on_buy)
        btns.addWidget(btn_buy)
        root.addLayout(btns)

    def _on_buy(self) -> None:
        QDesktopServices.openUrl(buy_url())
        self.accept()

    def _on_whats_in(self) -> None:
        ProFeaturesDialog(self).exec()
