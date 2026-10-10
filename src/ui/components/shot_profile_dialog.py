"""Editor for per-shot and per-object processing profiles."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.core.profiles import ShotProfile, ShotRange
from src.hardware.profiler import ModelSize
from src.ui.components.spill_panel import SpillPanel


class ShotProfileDialog(QDialog):
    """Edit model, edge, temporal, and detail settings for one shot."""

    def __init__(
        self,
        object_name: str,
        shots: list[ShotRange],
        selected_shot: int,
        profile: ShotProfile | None = None,
        parent: QWidget | None = None,
        frame: np.ndarray | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shot Processing Profile")
        self.setModal(True)
        self.setMinimumWidth(430)
        self._object_name = object_name
        self._shots = list(shots)
        self._profile = (profile or ShotProfile()).copy()
        self._build_ui(selected_shot)
        self._spill.set_frame(frame)

    @property
    def selected_shot_id(self) -> int:
        """Return the selected shot ID."""
        value = self._shot.currentData()
        return int(value) if value is not None else 1

    @property
    def apply_to_object_default(self) -> bool:
        """Return whether the edited profile should become the object default."""
        return self._object_default.isChecked()

    def profile(self) -> ShotProfile:
        """Return the edited profile.

        The colour and edge fields are owned by the embedded
        :class:`SpillPanel`, so they are read back from it rather than rebuilt
        from separate widgets - otherwise one of the two sources would silently
        win and the other set of controls would do nothing.
        """
        model = self._model.currentData()
        colour = self._spill.profile()
        return ShotProfile(
            name=self._name.text().strip() or "Shot profile",
            model_size=model,
            edge_smoothness=self._edge.value(),
            temporal_smoothing=self._temporal.isChecked(),
            detail_preservation=self._detail.value() / 100.0,
            mask_inset=self._inset.value(),
            spill_color=colour.spill_color,
            spill_mode=colour.spill_mode,
            spill_tolerance=colour.spill_tolerance,
            spill_radius=colour.spill_radius,
            spill_region=colour.spill_region,
            spill_region_mode=colour.spill_region_mode,
            cast_neutralize=colour.cast_neutralize,
            cast_strength=colour.cast_strength,
            edge_softness=colour.edge_softness,
        )

    def _build_ui(self, selected_shot: int) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(10)
        title = QLabel(f"Processing profile · {self._object_name}")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(title)
        intro = QLabel(
            "Settings apply to the selected shot for this object. The model can use the "
            "hardware recommendation or a fixed SAM 2 tier."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9a9aa4; font-size: 11px;")
        root.addWidget(intro)

        form = QFormLayout()
        self._shot = QComboBox()
        for shot in self._shots:
            self._shot.addItem(shot.label, shot.shot_id)
        index = self._shot.findData(selected_shot)
        self._shot.setCurrentIndex(max(0, index))
        form.addRow("Shot", self._shot)
        self._name = QLineEdit(self._profile.name)
        form.addRow("Profile name", self._name)
        self._model = QComboBox()
        self._model.addItem("Auto (recommended)", None)
        for model in (
            ModelSize.SAM3_1,
            ModelSize.TINY,
            ModelSize.SMALL,
            ModelSize.BASE_PLUS,
            ModelSize.LARGE,
        ):
            label = (
                "SAM 3.1 (Object Multiplex)"
                if model is ModelSize.SAM3_1
                else model.value
            )
            self._model.addItem(label, model)
        model_index = self._model.findData(self._profile.model)
        self._model.setCurrentIndex(max(0, model_index))
        form.addRow("Model", self._model)
        self._edge = QSlider(Qt.Orientation.Horizontal)
        self._edge.setRange(0, 10)
        self._edge.setValue(self._profile.edge_smoothness)
        form.addRow("Edge smoothness", self._edge)
        self._temporal = QCheckBox("Temporal smoothing")
        self._temporal.setChecked(self._profile.temporal_smoothing)
        form.addRow("Flicker cleanup", self._temporal)
        self._detail = QSlider(Qt.Orientation.Horizontal)
        self._detail.setRange(0, 100)
        self._detail.setValue(int(self._profile.detail_preservation * 100))
        form.addRow("Detail preservation", self._detail)
        self._inset = QSlider(Qt.Orientation.Horizontal)
        self._inset.setRange(0, 10)
        self._inset.setValue(self._profile.mask_inset)
        self._inset.setToolTip(
            "Shrink the matte inward by this many pixels to trim background\n"
            "color bleeding around the subject."
        )
        form.addRow("Mask inset (px)", self._inset)
        self._object_default = QCheckBox("Also use as this object's default")
        form.addRow("", self._object_default)
        root.addLayout(form)

        self._spill = SpillPanel(self)
        self._spill.set_profile(self._profile)
        root.addWidget(self._spill)

        actions = QHBoxLayout()
        actions.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        save = QPushButton("Apply")
        save.setProperty("cssClass", "accent")
        save.clicked.connect(self.accept)
        actions.addWidget(save)
        root.addLayout(actions)


__all__ = ["ShotProfileDialog"]
