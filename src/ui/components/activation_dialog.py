"""Activate Pro dialog — paste an offline key to unlock Pro features."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core import licensing

logger = logging.getLogger(__name__)


class ActivationDialog(QWidget):
    """Non-modal helper window for activating / deactivating the Pro edition."""

    changed = Signal()  # emitted after a successful activation or deactivation

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Activate Better Brush Pro")
        self.setFixedSize(480, 280)
        self.setStyleSheet("background: #202024;")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            "color: #c4c4cc; font-size: 12px; background: transparent;"
        )
        root.addWidget(self._status_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #2e2e33; max-height: 1px;")
        root.addWidget(sep)

        inp_lbl = QLabel("Paste your Pro key below (starts with BBP-):")
        inp_lbl.setStyleSheet(
            "color: #8b8b94; font-size: 11px; background: transparent;"
        )
        root.addWidget(inp_lbl)

        inp_row = QHBoxLayout()
        self._ed_key = QLineEdit()
        self._ed_key.setPlaceholderText("BBP-XXXX-XXXX-XXXX…")
        self._ed_key.textChanged.connect(self._refresh)
        self._ed_key.returnPressed.connect(self._on_activate)
        inp_row.addWidget(self._ed_key, stretch=1)
        self._btn_activate = QPushButton("Activate")
        self._btn_activate.clicked.connect(self._on_activate)
        inp_row.addWidget(self._btn_activate)
        root.addLayout(inp_row)

        self._hint_lbl = QLabel("Keys work fully offline — no internet required.")
        self._hint_lbl.setStyleSheet(
            "color: #555560; font-size: 10px; background: transparent;"
        )
        root.addWidget(self._hint_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_deactivate = QPushButton("Deactivate (switch to Lite)")
        self._btn_deactivate.clicked.connect(self._on_deactivate)
        btn_row.addWidget(self._btn_deactivate)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        self._refresh()

    def _refresh(self) -> None:
        if licensing.is_pro():
            who = licensing.current_licensee() or "a user"
            self._status_lbl.setText(
                f"<b style='color:#00b4f0'>Pro edition active</b> — licensed to <i>{who}</i>"
            )
            self._ed_key.setEnabled(False)
            self._btn_activate.setEnabled(False)
        else:
            self._status_lbl.setText("Running in <b>Lite</b> edition (free)")
            self._ed_key.setEnabled(True)
            self._btn_activate.setEnabled(bool(self._ed_key.text().strip()))

    def _on_activate(self) -> None:
        key = self._ed_key.text().strip()
        if not key:
            return
        ok, msg = licensing.activate(key)
        if ok:
            QMessageBox.information(self, "Activation successful", msg)
            self._ed_key.clear()
            self._refresh()
            self.changed.emit()
            logger.info("Pro activated: %s", msg)
        else:
            QMessageBox.warning(self, "Activation failed", msg)

    def _on_deactivate(self) -> None:
        if not licensing.is_pro():
            QMessageBox.information(
                self, "Already Lite", "Your installation is already in Lite mode."
            )
            return
        ret = QMessageBox.question(
            self,
            "Confirm deactivation",
            "This will remove your Pro activation and revert to the free Lite edition.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            licensing.deactivate()
            QMessageBox.information(
                self, "Deactivated", "Reverted to the free Lite edition."
            )
            self._refresh()
            self.changed.emit()
