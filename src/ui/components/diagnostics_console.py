"""Small read-only diagnostics console for Better Brush."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DiagnosticsConsole(QDialog):
    """Non-modal command console for support and power users."""

    command_submitted = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Better Brush Diagnostics")
        self.setModal(False)
        self.resize(760, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        intro = QLabel(
            "Read-only diagnostics. Type help for commands; commands never run a shell."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b8b94; font-size: 11px;")
        root.addWidget(intro)

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 10))
        self._output.setStyleSheet(
            "QPlainTextEdit { background: #141418; color: #d4d4d8; "
            "border: 1px solid #3a3a40; padding: 8px; }"
        )
        root.addWidget(self._output, stretch=1)

        input_row = QHBoxLayout()
        prompt = QLabel(">", objectName="prompt")
        prompt.setStyleSheet("color: #4a9eff; font-family: Consolas; font-size: 13px;")
        input_row.addWidget(prompt)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a command, for example: gpu")
        self._input.setStyleSheet(
            "QLineEdit { background: #202024; color: #d4d4d8; "
            "border: 1px solid #3a3a40; padding: 6px; }"
        )
        self._input.returnPressed.connect(self._submit)
        input_row.addWidget(self._input, stretch=1)
        self._btn_run = QPushButton("Run")
        self._btn_run.setProperty("cssClass", "accent")
        self._btn_run.setToolTip("Run the entered command")
        self._btn_run.clicked.connect(self._submit)
        input_row.addWidget(self._btn_run)
        root.addLayout(input_row)

        actions = QHBoxLayout()
        actions.addStretch()
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self.clear_output)
        actions.addWidget(self._btn_clear)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        root.addLayout(actions)

        self.append_output("Better Brush diagnostics console. Type help for commands.")

    def _submit(self) -> None:
        command = self._input.text().strip()
        if not command:
            return
        self._input.clear()
        self.append_output(f"> {command}")
        self.command_submitted.emit(command)

    def append_output(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self._output.appendPlainText(line)
        scrollbar = self._output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self._output.ensureCursorVisible()

    def append_error(self, text: str) -> None:
        self.append_output(f"ERROR: {text}")

    def clear_output(self) -> None:
        self._output.clear()
        self._input.setFocus()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._input.setFocus(Qt.FocusReason.ShortcutFocusReason)
