"""Collapsible section panel for the right sidebar."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

# Header sub-widget colours. Kept as constants because the hover and
# expanded states each need their own pair, and they must agree with the
# theme's section-header colours or the header flickers on hover.
_ARROW_CLOSED = (
    "color: #6b6b74; font-size: 10px; border: none; background: transparent;"
)
_ARROW_OPEN = "color: #4a9eff; font-size: 10px; border: none; background: transparent;"
_ARROW_HOVER = "color: #d4d4d8; font-size: 10px; border: none; background: transparent;"
_LABEL_CLOSED = (
    "color: #8b8b94; font-size: 11px; font-weight: 600; "
    "letter-spacing: 1px; border: none; background: transparent;"
)
_LABEL_OPEN = (
    "color: #d4d4d8; font-size: 11px; font-weight: 600; "
    "letter-spacing: 1px; border: none; background: transparent;"
)
_LABEL_HOVER = (
    "color: #ffffff; font-size: 11px; font-weight: 600; "
    "letter-spacing: 1px; border: none; background: transparent;"
)


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
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.setToolTip(f"Expand or collapse {title}")
        # Only `text-align` is forced here: a widget-level stylesheet otherwise
        # shadows the theme's hover and checked rules for this button.
        header.setStyleSheet("QPushButton { text-align: left; }")
        self.setAccessibleName(title)
        self._header = header
        self._title_label = title.upper()

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(6)

        # The arrow and label are recoloured on hover and on expand so the
        # header reads as something you can click, and so an open section is
        # distinguishable from a closed one at a glance.
        self._arrow = QLabel("▾")
        self._arrow.setStyleSheet(
            "color: #6b6b74; font-size: 10px; border: none; background: transparent;"
        )
        header_layout.addWidget(self._arrow)

        self._label = QLabel(self._title_label)
        self._label.setStyleSheet(_LABEL_CLOSED)
        header_layout.addWidget(self._label)
        header_layout.addStretch()

        header.clicked.connect(self._toggle)
        # QPushButton has no hovered signal, so enter/leave come through an
        # event filter instead.
        header.installEventFilter(self)
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

    def eventFilter(self, watched, event) -> bool:
        """Track the header hover state (QPushButton has no hovered signal)."""
        if watched is self._header:
            if event.type() == QEvent.Type.Enter:
                self._on_header_hovered(True)
            elif event.type() == QEvent.Type.Leave:
                self._on_header_hovered(False)
        return super().eventFilter(watched, event)

    def _toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._set_arrow(checked)
        self.toggled.emit(checked)

    def _set_arrow(self, expanded: bool) -> None:
        self._arrow.setText("\u25be" if expanded else "\u25b8")
        self._apply_header_colors()

    def _on_header_hovered(self, inside: bool) -> None:
        """Brighten the arrow and label while the header is under the cursor."""
        if inside:
            self._arrow.setStyleSheet(_ARROW_HOVER)
            self._label.setStyleSheet(_LABEL_HOVER)
        else:
            self._apply_header_colors()

    def _apply_header_colors(self) -> None:
        """Colour the arrow and label for the current expanded state.

        An open section gets the accent tint so expanded and collapsed are
        distinguishable at a glance, not only by the arrow glyph.
        """
        expanded = self._header.isChecked()
        self._arrow.setStyleSheet(_ARROW_OPEN if expanded else _ARROW_CLOSED)
        self._label.setStyleSheet(_LABEL_OPEN if expanded else _LABEL_CLOSED)


class Separator(QWidget):
    """Thin horizontal separator line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: #2e2e33;")
