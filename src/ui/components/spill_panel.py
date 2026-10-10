"""Colour controls for the matte edge: spill cleaner and edge finishing.

Lives inside the Shot Profile dialog rather than the sidebar.  It carries its
own small preview of the current frame so the eyedropper and the area box can
be used while the dialog is open - a modal dialog would otherwise block the
main canvas, which is what made these controls awkward to reach before.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.core.profiles import ShotProfile
from src.ui.video_canvas import VideoCanvas

_LBL = "color: #a0a0a8; font-size: 11px; background: transparent;"
_VAL = "color: #8b8b94; font-size: 11px; background: transparent;"
_DIM = "color: #7a7a84; font-size: 10px; background: transparent;"


class SpillPanel(QWidget):
    """Spill cleaner, colour cast fix, and edge softening for one shot.

    Signals:
        changed: Any control changed; the host should persist and repaint.
        spill_color_picked: Emitted with an ``(r, g, b)`` sample.
        area_drawn: Emitted with an ``(x, y, w, h)`` box, or ``None`` to clear.
    """

    changed = Signal()
    spill_color_picked = Signal(int, int, int)
    area_drawn = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pending_area_mode = "clean"
        self._build_ui()
        self.set_profile(ShotProfile())

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Live preview, also the eyedropper and area-box surface.
        self._canvas = VideoCanvas(self)
        self._canvas.setMinimumSize(320, 180)
        self._canvas.setMaximumHeight(190)
        self._canvas.set_preview_mode("frame")
        self._canvas.color_picked.connect(self._on_canvas_color)
        self._canvas.region_selected.connect(self._on_canvas_region)
        self._canvas.region_cancelled.connect(self._on_canvas_region_cancel)
        root.addWidget(self._canvas)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._mode = QComboBox()
        self._mode.addItem("Off", "off")
        self._mode.addItem("Despill", "despill")
        self._mode.addItem("Alpha key", "alpha_key")
        self._mode.setToolTip(
            "Off leaves the edge alone.\n"
            "Despill keeps the coverage and removes the background hue.\n"
            "Alpha key fades the fringe out of the matte instead."
        )
        self._mode.currentIndexChanged.connect(self._on_any)
        row.addWidget(self._mode, stretch=1)
        self._pick = QPushButton("Pick")
        self._pick.setToolTip("Click the fringe on the preview to sample it")
        self._pick.clicked.connect(self._on_pick_clicked)
        row.addWidget(self._pick)
        self._swatch = QLabel("-")
        self._swatch.setFixedSize(22, 20)
        self._swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._swatch.setToolTip("The spill colour being suppressed")
        row.addWidget(self._swatch)
        root.addLayout(row)

        area = QHBoxLayout()
        area.setSpacing(6)
        self._area = QPushButton("Set Area")
        self._area.setToolTip(
            "Clean only inside a box you draw on the preview.\n"
            "Use when the spill is in one part of frame, e.g. sky on the right."
        )
        self._area.clicked.connect(lambda: self._begin_area("clean"))
        area.addWidget(self._area)
        self._protect = QPushButton("Protect")
        self._protect.setToolTip(
            "Shield a box and clean everywhere else.\n"
            "Use when part of the subject is genuinely the spill colour,\n"
            "such as blue hair under a blue sky."
        )
        self._protect.clicked.connect(lambda: self._begin_area("protect"))
        area.addWidget(self._protect)
        self._clear_area = QPushButton("All Frame")
        self._clear_area.setToolTip("Back to cleaning the whole matte edge")
        self._clear_area.clicked.connect(self._on_clear_area)
        area.addWidget(self._clear_area)
        root.addLayout(area)

        self._area_label = QLabel("Whole frame - matte edge only")
        self._area_label.setWordWrap(True)
        self._area_label.setStyleSheet(_DIM)
        root.addWidget(self._area_label)

        root.addWidget(self._slider("Tolerance", 0, 100, self._on_any, _LBL))
        self._tolerance = root.layout().itemAt(root.layout().count() - 1).widget()
        root.addWidget(self._slider("Edge band radius", 1, 10, self._on_any, _LBL))
        self._radius = root.layout().itemAt(root.layout().count() - 1).widget()
        root.addWidget(self._slider("Edge softness", 0, 4, self._on_any, _LBL))
        self._softness = root.layout().itemAt(root.layout().count() - 1).widget()
        soft_tip = (
            "Anti-aliases the exported matte and removes the background\n"
            "colour from edge pixels. 0 gives a hard, jagged edge. 2 is right\n"
            "for most work; raise it for very soft hair."
        )
        self._softness.layout().itemAt(0).widget().setToolTip(soft_tip)

        self._cast = QCheckBox("Fix colour cast")
        self._cast.setToolTip(
            "Pulls a global colour cast out of the shot before cleaning.\n"
            "Only enable this when the tint is unwanted light rather than an\n"
            "intentional background colour - it will neutralise the backdrop."
        )
        self._cast.toggled.connect(self._on_any)
        root.addWidget(self._cast)
        root.addWidget(self._slider("Cast strength", 0, 100, self._on_any, _LBL))
        self._cast_strength = root.layout().itemAt(root.layout().count() - 1).widget()

    def _slider(self, label: str, lo: int, hi: int, slot, style: str) -> QWidget:
        """Return a labelled slider row, keeping the value label addressable."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        name = QLabel(label)
        name.setStyleSheet(style)
        name.setFixedWidth(104)
        row.addWidget(name)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.valueChanged.connect(slot)
        row.addWidget(slider, stretch=1)
        value = QLabel("0")
        value.setFixedWidth(38)
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        value.setStyleSheet(_VAL)
        row.addWidget(value)
        slider.valueChanged.connect(lambda v, w=value: w.setText(str(v)))
        # Stash so callers can reach the slider itself.
        box.slider = slider  # type: ignore[attr-defined]
        box.value_label = value  # type: ignore[attr-defined]
        return box

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_frame(self, frame) -> None:
        """Show *frame* in the preview so the eyedropper has something to read."""
        self._canvas.set_frame(frame)

    def set_profile(self, profile: ShotProfile) -> None:
        """Load every control from *profile* without emitting changes."""
        widgets = (
            self._mode,
            self._tolerance.slider,
            self._radius.slider,
            self._softness.slider,
            self._cast_strength.slider,
            self._cast,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            index = self._mode.findData(profile.spill_mode)
            self._mode.setCurrentIndex(max(0, index))
            self._tolerance.slider.setValue(profile.spill_tolerance)
            self._radius.slider.setValue(profile.spill_radius)
            self._softness.slider.setValue(profile.edge_softness)
            self._cast.setChecked(profile.cast_neutralize)
            self._cast_strength.slider.setValue(profile.cast_strength)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._pending_area_mode = profile.spill_region_mode
        self._area_mode = profile.spill_region_mode
        self._area_box = profile.spill_region
        self._area_color = profile.spill_color
        self._refresh_labels()

    def profile(self) -> ShotProfile:
        """Return the settings currently shown, as a bare profile."""
        return ShotProfile(
            spill_mode=self._mode.currentData(),
            spill_tolerance=self._tolerance.slider.value(),
            spill_radius=self._radius.slider.value(),
            edge_softness=self._softness.slider.value(),
            cast_neutralize=self._cast.isChecked(),
            cast_strength=self._cast_strength.slider.value(),
            spill_region=self._area_box,
            spill_region_mode=self._area_mode,
            spill_color=self._area_color,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    _area_mode: str = "clean"
    _area_box: tuple[int, int, int, int] | None = None
    _area_color: tuple[int, int, int] | None = None

    def _on_any(self, *_args) -> None:
        self._refresh_labels()
        self.changed.emit()

    def _on_pick_clicked(self) -> None:
        if self._mode.currentData() == "off":
            self._mode.setCurrentIndex(self._mode.findData("despill"))
        self._canvas.begin_color_pick()

    def _begin_area(self, mode: str) -> None:
        if self._mode.currentData() == "off":
            self._mode.setCurrentIndex(self._mode.findData("despill"))
        self._pending_area_mode = mode
        hint = (
            "Drag a box to CLEAN  (Esc to cancel)"
            if mode == "clean"
            else "Drag a box to PROTECT  (Esc to cancel)"
        )
        self._canvas.begin_region_pick(hint)

    def _on_canvas_color(self, red: int, green: int, blue: int) -> None:
        self._area_color = (int(red), int(green), int(blue))
        self._refresh_labels()
        self.spill_color_picked.emit(red, green, blue)
        self.changed.emit()

    def _on_canvas_region(self, x: int, y: int, w: int, h: int) -> None:
        self._canvas.end_region_pick()
        self._area_mode = self._pending_area_mode
        self._area_box = (int(x), int(y), int(w), int(h))
        self._refresh_labels()
        self.area_drawn.emit(self._area_box)
        self.changed.emit()

    def _on_canvas_region_cancel(self) -> None:
        self._canvas.end_region_pick()

    def _on_clear_area(self) -> None:
        self._area_box = None
        self._area_mode = "clean"
        self._refresh_labels()
        self.area_drawn.emit(None)
        self.changed.emit()

    def _refresh_labels(self) -> None:
        self._tolerance.value_label.setText(str(self._tolerance.slider.value()))
        self._radius.value_label.setText(f"{self._radius.slider.value()} px")
        self._softness.value_label.setText(f"{self._softness.slider.value()} px")
        self._cast_strength.value_label.setText(
            f"{self._cast_strength.slider.value()}%"
        )
        self._cast_strength.setEnabled(self._cast.isChecked())
        color = self._area_color
        if color is None:
            self._swatch.setText("-")
            self._swatch.setStyleSheet(
                "color: #8b8b94; font-size: 10px; background: transparent;"
                "border: 1px solid #3a3a40; border-radius: 3px;"
            )
        else:
            red, green, blue = color
            luma = (red * 299 + green * 587 + blue * 114) / 1000
            ink = "#101014" if luma > 140 else "#e6e6ea"
            self._swatch.setText("")
            self._swatch.setStyleSheet(
                f"background: rgb({red}, {green}, {blue}); color: {ink};"
                "border: 1px solid #3a3a40; border-radius: 3px;"
            )
        if self._area_box is None:
            self._area_label.setText("Whole frame - matte edge only")
        else:
            x, y, w, h = self._area_box
            verb = "Protect" if self._area_mode == "protect" else "Clean"
            self._area_label.setText(f"{verb} box {w}x{h} at ({x}, {y})")
