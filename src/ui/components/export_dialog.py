"""Professional export dialog inspired by Adobe Premiere Pro."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.processing.video_reader import VideoReader

# ── Format metadata ────────────────────────────────────────────────────────
FORMATS = [
    ("prores", "ProRes 4444 (MOV)", ".mov", True),
    ("qtrle", "QuickTime Animation (MOV)", ".mov", True),
    ("h264", "H.264 (MP4)", ".mp4", False),
    ("png", "PNG Sequence (RGBA)", "", False),
]

# Rough bytes-per-pixel-per-frame estimates for size forecasting
_BPP_ESTIMATE = {
    "prores": 1.2,  # ProRes 4444 ~ compressed
    "qtrle": 2.5,  # Animation codec is bulky
    "h264": 0.08,  # heavily compressed
    "png": 4.0,  # uncompressed RGBA
}

# Preset resolutions (excluding "Match Source")
_PRESETS = ["1920 x 1080", "1280 x 720", "960 x 540"]

# Solid backgrounds for chroma-keying (H.264 MP4 exports without alpha)
BACKGROUNDS = [
    ("none", "No background (black)"),
    ("Vivid Green", (0, 255, 0)),
    ("Vivid Blue", (0, 0, 255)),
    ("Magenta", (255, 0, 255)),
    ("Red", (255, 0, 0)),
    ("White", (255, 255, 255)),
    ("Black", (0, 0, 0)),
]


class ExportDialog(QDialog):
    """Modal export dialog with file settings, format, and export range."""

    def __init__(
        self,
        parent: QWidget | None = None,
        video_reader: VideoReader | None = None,
        num_masks: int = 0,
        num_frames: int = 0,
        timeline_selection: tuple[int, int] | None = None,
        objects: list[dict] | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_reader = video_reader
        self._num_masks = num_masks
        self._num_frames = num_frames or (
            video_reader.total_frames if video_reader else 0
        )
        self._timeline_selection = timeline_selection
        self._objects = objects or []
        self._current_frame = 0

        self.setWindowTitle("Export")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setStyleSheet(
            """
            background: #202024;
            QRadioButton { color: #c4c4cc; background: transparent; }
            QRadioButton::indicator {
                width: 13px; height: 13px; border: 1px solid #555560;
                border-radius: 7px; background: #1a1a1e;
            }
            QRadioButton::indicator:hover { border-color: #8b8b94; }
            QRadioButton::indicator:checked {
                border-color: #00b4f0; background: #00b4f0;
            }
            QCheckBox { color: #c4c4cc; background: transparent; }
            QCheckBox::indicator {
                width: 14px; height: 14px; border: 1px solid #555560;
                border-radius: 3px; background: #1a1a1e;
            }
            QCheckBox::indicator:checked { background: #00b4f0; border-color: #00b4f0; }
            QCheckBox:disabled { color: #3a3a42; }
            QComboBox:disabled { color: #3a3a42; background: #171719; }
            QPushButton:disabled { color: #3a3a42; }
            """
        )

        self._build_ui()
        self._apply_limits()
        self._update_estimates()

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        root.addLayout(self._section_title("FILE NAME & PATH"))

        # ── File name / path ───────────────────────────────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        grid.addWidget(self._label("File Name:"), 0, 0)
        self._ed_name = QLineEdit("output")
        self._ed_name.setMinimumWidth(260)
        grid.addWidget(self._ed_name, 0, 1)

        default_dir = Path.cwd() / "exports"
        grid.addWidget(self._label("Export Path:"), 1, 0)
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        self._ed_path = QLineEdit(str(default_dir))
        self._ed_path.setMinimumWidth(260)
        path_row.addWidget(self._ed_path, stretch=1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._on_browse)
        path_row.addWidget(btn_browse)
        grid.addLayout(path_row, 1, 1)
        root.addLayout(grid)

        root.addSpacing(6)
        root.addWidget(self._separator())

        # ── Format & Preset ────────────────────────────────────────────────
        root.addLayout(self._section_title("FORMAT & PRESET"))

        fmt_row = QGridLayout()
        fmt_row.setHorizontalSpacing(8)
        fmt_row.setVerticalSpacing(8)

        fmt_row.addWidget(self._label("Format:"), 0, 0)
        self._combo_format = QComboBox()
        for fid, label, _ext, _alpha in FORMATS:
            self._combo_format.addItem(label, fid)
        fmt_row.addWidget(self._combo_format, 0, 1)

        fmt_row.addWidget(self._label("Preset:"), 1, 0)
        self._combo_preset = QComboBox()
        self._combo_preset.addItem("Match Source")
        self._combo_preset.addItems(_PRESETS)
        fmt_row.addWidget(self._combo_preset, 1, 1)

        root.addLayout(fmt_row)

        self._combo_format.currentIndexChanged.connect(self._on_format_changed)
        self._combo_preset.currentIndexChanged.connect(self._on_preset_changed)

        root.addSpacing(6)
        root.addWidget(self._separator())

        # ── Video Settings ─────────────────────────────────────────────────
        root.addLayout(self._section_title("VIDEO SETTINGS"))

        vs = QGridLayout()
        vs.setHorizontalSpacing(8)
        vs.setVerticalSpacing(8)

        vs.addWidget(self._label("Resolution:"), 0, 0)
        self._spin_w = QSpinBox()
        self._spin_w.setRange(16, 8192)
        self._spin_w.setValue(self._source_width())
        vs.addWidget(self._spin_w, 0, 1)
        vs.addWidget(self._label("x"), 0, 2)
        self._spin_h = QSpinBox()
        self._spin_h.setRange(16, 8192)
        self._spin_h.setValue(self._source_height())
        vs.addWidget(self._spin_h, 0, 3)
        self._chk_match = QCheckBox("Match Source")
        self._chk_match.setChecked(True)
        self._chk_match.toggled.connect(self._on_match_toggled)
        vs.addWidget(self._chk_match, 0, 4)
        root.addLayout(vs)

        vs2 = QGridLayout()
        vs2.setHorizontalSpacing(8)
        vs2.setVerticalSpacing(8)
        vs2.addWidget(self._label("Frame Rate:"), 1, 0)
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1.0, 240.0)
        self._spin_fps.setDecimals(2)
        self._spin_fps.setValue(self._source_fps())
        vs2.addWidget(self._spin_fps, 1, 1)

        frame_row = QHBoxLayout()
        frame_row.addLayout(vs2)
        frame_row.addWidget(self._label("fps"), alignment=Qt.AlignmentFlag.AlignLeft)
        root.addLayout(frame_row)

        # ── Background (H.264 only — no alpha in MP4) ──────────────────────
        self._bg_widget = QWidget()
        self._bg_widget.setStyleSheet("background: transparent;")
        bg_inner = QHBoxLayout(self._bg_widget)
        bg_inner.setContentsMargins(0, 0, 0, 0)
        bg_inner.setSpacing(8)
        bg_inner.addWidget(self._label("Background:"))
        self._combo_bg = QComboBox()
        for name, _color in BACKGROUNDS:
            self._combo_bg.addItem(name, None if name == "none" else _color)
        self._combo_bg.setToolTip(
            "MP4 has no alpha channel. Pick a vibrant color as the backdrop so "
            "you can chroma-key / ultra-key the clip in your editor."
        )
        bg_inner.addWidget(self._combo_bg)
        lbl_hint = QLabel("for chroma-key use")
        lbl_hint.setStyleSheet(
            "color: #555560; font-size: 10px; background: transparent;"
        )
        bg_inner.addWidget(lbl_hint, alignment=Qt.AlignmentFlag.AlignLeft)
        bg_inner.addStretch()
        root.addWidget(self._bg_widget)
        self._bg_widget.setVisible(False)

        # ── Original audio ────────────────────────────────────────────────
        root.addSpacing(4)
        audio_row = QHBoxLayout()
        audio_row.setSpacing(8)
        self._chk_audio = QCheckBox("Include Original Audio")
        self._chk_audio.setToolTip(
            "Muxes the source video's audio track into the export "
            "(video formats only). Ignored for PNG sequences."
        )
        audio_row.addWidget(self._chk_audio)
        if self._video_reader is not None:
            src_name_lbl = QLabel(f"from: {self._video_reader.path.name}")
            src_name_lbl.setStyleSheet(
                "color: #6b6b74; font-size: 10px; background: transparent;"
            )
            audio_row.addWidget(src_name_lbl, alignment=Qt.AlignmentFlag.AlignLeft)
        audio_row.addStretch()
        root.addLayout(audio_row)

        root.addSpacing(6)
        root.addWidget(self._separator())

        # ── Objects to Export ─────────────────────────────────────────────
        root.addLayout(self._section_title("OBJECTS TO EXPORT"))

        self._rb_combined = QRadioButton("All objects in one file (combined masks)")
        self._rb_combined.setChecked(True)
        root.addWidget(self._rb_combined)

        single_row = QHBoxLayout()
        single_row.setSpacing(8)
        self._rb_single = QRadioButton("Single object:")
        single_row.addWidget(self._rb_single)
        self._combo_object = QComboBox()
        for obj in self._objects:
            self._combo_object.addItem(obj.get("name", "Object"), obj.get("id"))
        self._combo_object.setEnabled(False)
        single_row.addWidget(self._combo_object)
        single_row.addStretch()
        root.addLayout(single_row)

        self._rb_separate = QRadioButton("Separate files (one file per object)")
        root.addWidget(self._rb_separate)

        self._rb_single.toggled.connect(self._on_object_mode_changed)
        self._rb_separate.toggled.connect(self._update_estimates)

        root.addSpacing(6)
        root.addWidget(self._separator())

        # ── Export Range ───────────────────────────────────────────────────
        root.addLayout(self._section_title("EXPORT RANGE"))

        # Timeline selection (from the timeline widget)
        if self._timeline_selection is not None:
            sel_s, sel_e = self._timeline_selection
            self._rb_timeline = QRadioButton(
                f"Timeline Selection (frames {sel_s} - {sel_e})"
            )
            self._rb_timeline.setChecked(True)
        else:
            self._rb_timeline = QRadioButton("Timeline Selection")
            self._rb_timeline.setEnabled(False)
            self._rb_timeline.setToolTip(
                "Select a range on the timeline (Shift+drag) first."
            )

        self._rb_all = QRadioButton("All Masked Frames")
        if self._timeline_selection is None:
            self._rb_all.setChecked(True)
        root.addWidget(self._rb_timeline)
        root.addWidget(self._rb_all)

        range_row = QHBoxLayout()
        range_row.setSpacing(8)
        self._rb_range = QRadioButton("Frame Range:")
        range_row.addWidget(self._rb_range)
        self._spin_from = QSpinBox()
        self._spin_from.setRange(0, max(0, int(self._num_frames) - 1))
        range_row.addWidget(self._spin_from)
        range_row.addWidget(self._label("to"))
        self._spin_to = QSpinBox()
        self._spin_to.setRange(0, max(0, int(self._num_frames) - 1))
        self._spin_to.setValue(max(0, int(self._num_frames) - 1))
        range_row.addWidget(self._spin_to)
        root.addLayout(range_row)

        if self._timeline_selection is not None:
            sel_s, sel_e = self._timeline_selection
            self._spin_from.setValue(sel_s)
            self._spin_to.setValue(sel_e)

        self._rb_timeline.toggled.connect(self._update_estimates)
        self._rb_all.toggled.connect(self._update_estimates)

        root.addStretch()

        # ── Estimated size ─────────────────────────────────────────────────
        self._lbl_size = QLabel()
        self._lbl_size.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_size.setStyleSheet(
            "color: #8b8b94; font-size: 11px; background: transparent;"
        )
        root.addWidget(self._lbl_size)

        # ── Action bar ─────────────────────────────────────────────────────
        actions = QHBoxLayout()
        actions.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        actions.addWidget(btn_cancel)

        self._btn_export = QPushButton("Export")
        self._btn_export.setProperty("cssClass", "accent")
        self._btn_export.setDefault(True)
        self._btn_export.clicked.connect(self.accept)
        actions.addWidget(self._btn_export)

        root.addLayout(actions)

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #8b8b94; font-size: 11px; background: transparent;")
        lbl.setFixedWidth(90)
        return lbl

    @staticmethod
    def _section_title(text: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #8b8b94; font-size: 10px; font-weight: 600; "
            "letter-spacing: 1px; background: transparent;"
        )
        row.addWidget(lbl)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setStyleSheet("color: #2e2e33; background: #2e2e33; max-height: 1px;")
        row.addWidget(rule, stretch=1)
        return row

    @staticmethod
    def _separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #2e2e33; color: #2e2e33;")
        return sep

    # ── Source data ───────────────────────────────────────────────────────
    def _source_width(self) -> int:
        return self._video_reader.width if self._video_reader else 1920

    def _source_height(self) -> int:
        return self._video_reader.height if self._video_reader else 1080

    def _source_fps(self) -> float:
        return self._video_reader.fps if self._video_reader else 30.0

    # ── Handlers ──────────────────────────────────────────────────────────
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose Export Folder", self._ed_path.text()
        )
        if path:
            self._ed_path.setText(path)

    def _on_object_mode_changed(self) -> None:
        self._combo_object.setEnabled(self._rb_single.isChecked())
        self._update_estimates()

    def _on_format_changed(self) -> None:
        fmt = self._combo_format.currentData() or "prores"
        self._chk_audio.setEnabled(fmt != "png")
        self._bg_widget.setVisible(fmt == "h264")
        if fmt == "h264":
            # MP4 has no alpha channel — offering "No background (black)" is
            # confusing. Only solid chroma-key colors make sense here.
            for i in range(self._combo_bg.count()):
                is_none = self._combo_bg.itemData(i) is None
                self._combo_bg.model().item(i).setEnabled(not is_none)
            if self._combo_bg.currentData() is None:
                self._combo_bg.setCurrentIndex(1)  # "Vivid Green" default
        else:
            for i in range(self._combo_bg.count()):
                self._combo_bg.model().item(i).setEnabled(True)
        self._update_estimates()

    # ── Limits ────────────────────────────────────────────────────────────
    def _apply_limits(self) -> None:
        """Better Brush is free: every export option is always available."""
        self._spin_w.setMaximum(8192)
        self._spin_h.setMaximum(8192)
        self._chk_audio.setEnabled(True)
        self._chk_audio.setToolTip(
            "Muxes the source video's audio track into the export "
            "(video formats only). Ignored for PNG sequences."
        )

    def _on_preset_changed(self) -> None:
        text = self._combo_preset.currentText()
        if text == "Match Source":
            self._chk_match.setChecked(True)
        else:
            try:
                w, h = (int(p) for p in text.split(" x "))
            except ValueError:
                return
            self._chk_match.setChecked(False)
            self._spin_w.setValue(w)
            self._spin_h.setValue(h)
        self._update_estimates()

    def _on_match_toggled(self, checked: bool) -> None:
        if checked:
            self._spin_w.setValue(self._source_width())
            self._spin_h.setValue(self._source_height())
            self._spin_fps.setValue(self._source_fps())
        self._update_estimates()

    # ── Estimation ────────────────────────────────────────────────────────
    def _update_estimates(self) -> None:
        fmt = self._combo_format.currentData() or "prores"
        w, h = self._spin_w.value(), self._spin_h.value()
        self._spin_fps.value()

        if self._rb_timeline.isChecked() and self._timeline_selection is not None:
            n = self._timeline_selection[1] - self._timeline_selection[0] + 1
        elif self._rb_all.isChecked():
            n = self._num_masks or self._num_frames
        else:
            n = max(0, self._spin_to.value() - self._spin_from.value() + 1)

        if n <= 0:
            self._lbl_size.setText("Nothing to export in this range")
            return

        bpp = _BPP_ESTIMATE.get(fmt, 1.0)
        size_bytes = w * h * bpp * n
        if size_bytes > 10_000_000_000:
            text = f"Estimated size: {size_bytes / 1_000_000_000:.1f} GB"
        elif size_bytes > 10_000_000:
            text = f"Estimated size: {size_bytes / 1_000_000:.1f} MB"
        else:
            text = f"Estimated size: {size_bytes / 1_000:.1f} KB"
        self._lbl_size.setText(text)

    # ── Result ────────────────────────────────────────────────────────────
    def settings(self) -> dict:
        """Return validated export settings."""
        fmt = self._combo_format.currentData() or "prores"
        name = self._ed_name.text().strip() or "output"
        path = self._ed_path.text().strip()
        match = self._chk_match.isChecked()
        include_audio = bool(
            self._chk_audio.isChecked()
            and fmt != "png"
            and self._video_reader is not None
        )
        resolution = (self._spin_w.value(), self._spin_h.value()) if not match else None
        background = (
            self._combo_bg.currentData()
            if fmt == "h264" and self._combo_bg.currentData() is not None
            else None
        )
        if self._rb_timeline.isChecked() and self._timeline_selection is not None:
            range_mode = "range"
            start_frame, end_frame = self._timeline_selection
        elif self._rb_all.isChecked():
            range_mode = "all"
            start_frame, end_frame = 0, max(0, int(self._num_frames) - 1)
        else:
            range_mode = "range"
            start_frame = min(self._spin_from.value(), self._spin_to.value())
            end_frame = max(self._spin_from.value(), self._spin_to.value())
        if self._rb_single.isChecked():
            object_mode = "single"
        elif self._rb_separate.isChecked():
            object_mode = "separate"
        else:
            object_mode = "combined"
        return {
            "format": fmt,
            "file_name": name,
            "output_dir": Path(path),
            "fps": self._spin_fps.value(),
            "resolution": resolution,
            "range_mode": range_mode,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "include_audio": include_audio,
            "background": background,
            "object_mode": object_mode,
            "object_id": self._combo_object.currentData(),
        }
