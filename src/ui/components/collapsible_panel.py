"""Collapsible section panel for the right sidebar."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A titled section that can be expanded or collapsed."""

    toggled = Signal(bool)  # True = expanded

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QPushButton()
        header.setCheckable(True)
        header.setChecked(expanded)
        header.setProperty("cssClass", "section-header")
        header.setStyleSheet(
            "QPushButton { text-align: left; }"
        )
        self._header = header
        self._title_label = title.upper()

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(6)

        self._arrow = QLabel("\u25BE")
        self._arrow.setStyleSheet("color: #8b8b94; font-size: 10px; border: none;")
        header_layout.addWidget(self._arrow)

        lbl = QLabel(self._title_label)
        lbl.setStyleSheet(
            "color: #8b8b94; font-size: 11px; font-weight: 600; "
            "letter-spacing: 1px; border: none; background: transparent;"
        )
        header_layout.addWidget(lbl)
        header_layout.addStretch()

        header.clicked.connect(self._toggle)
        root.addWidget(header)

        # Content area
        self._content = QWidget()
        self._content.setStyleSheet("background: transparent;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 8, 8)
        self._content_layout.setSpacing(6)
        root.addWidget(self._content)

        self._set_arrow(expanded)
        self._content.setVisible(expanded)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)

    def add_stretch(self) -> None:
        self._content_layout.addStretch()

    def _toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._set_arrow(checked)
        self.toggled.emit(checked)

    def _set_arrow(self, expanded: bool) -> None:
        self._arrow.setText("\u25BE" if expanded else "\u25B8")


class Separator(QWidget):
    """Thin horizontal separator line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: #2e2e33;")
