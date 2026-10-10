"""Zoomable source-frame and current-matte quality-control inspector."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.processing.matte_qc import (
    MatteQCReport,
    analyze_matte,
    compare_matte_reports,
    render_mask_overlay,
)
from src.processing.qc_coach import (
    QCAnalysis,
    QCTipsDatabase,
    analyze_qc_image,
)


def _pixmap(image: np.ndarray) -> QPixmap:
    array = np.ascontiguousarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        return QPixmap()
    array = array[:, :, :3]
    height, width = array.shape[:2]
    qimage = QImage(
        array.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimage.copy())


class MattePreview(QWidget):
    """Display a matte overlay with wheel zoom and drag-to-pan."""

    zoom_changed = Signal(float)
    view_changed = Signal(float, int, int)
    activated = Signal(object)

    def __init__(self, image: np.ndarray | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = _pixmap(image) if image is not None else QPixmap()
        self._zoom = 1.0
        self._pan = (0, 0)
        self._drag_start: tuple[int, int] | None = None
        self.setMinimumSize(240, 180)
        self.setMouseTracking(True)
        self.setToolTip("Ctrl+wheel or wheel to zoom · drag to pan")

    def set_image(self, image: np.ndarray | None) -> None:
        self._pixmap = _pixmap(image) if image is not None else QPixmap()
        self.update()

    def set_zoom(self, value: float, notify: bool = True) -> None:
        self._zoom = max(0.25, min(8.0, float(value)))
        self.zoom_changed.emit(self._zoom)
        if notify:
            self.view_changed.emit(self._zoom, self._pan[0], self._pan[1])
        self.update()

    def set_pan(self, x: int, y: int, notify: bool = True) -> None:
        self._pan = (int(x), int(y))
        if notify:
            self.view_changed.emit(self._zoom, self._pan[0], self._pan[1])
        self.update()

    def set_view(self, zoom: float, pan: tuple[int, int], notify: bool = True) -> None:
        """Set zoom and pan together for linked previews."""
        self.set_zoom(zoom, notify=False)
        self.set_pan(pan[0], pan[1], notify=notify)

    def pan(self) -> tuple[int, int]:
        """Return the current pan offset."""
        return self._pan

    def zoom(self) -> float:
        """Return the current display zoom."""
        return self._zoom

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pixmap.isNull():
            painter.end()
            return
        base_scale = min(
            self.width() / self._pixmap.width(),
            self.height() / self._pixmap.height(),
        )
        scale = base_scale * self._zoom
        width = max(1, int(self._pixmap.width() * scale))
        height = max(1, int(self._pixmap.height() * scale))
        x = (self.width() - width) // 2 + int(self._pan[0])
        y = (self.height() - height) // 2 + int(self._pan[1])
        painter.drawPixmap(x, y, width, height, self._pixmap)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if not delta:
            return
        factor = 1.15 ** (delta / 120)
        self.set_zoom(self._zoom * factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (event.position().x(), event.position().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.activated.emit(self)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            return
        position = event.position()
        dx = int(position.x() - self._drag_start[0])
        dy = int(position.y() - self._drag_start[1])
        self._pan = (self._pan[0] + dx, self._pan[1] + dy)
        self._drag_start = (position.x(), position.y())
        self.view_changed.emit(self._zoom, self._pan[0], self._pan[1])
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)


class MatteInspectorDialog(QDialog):
    """Show the source frame, current matte, QC evidence, and comparison metrics."""

    apply_suggestion_requested = Signal(object)

    def __init__(
        self,
        frame_rgb: np.ndarray | None,
        before_mask: np.ndarray | None,
        after_mask: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
        profile: object | None = None,
        tips_database: QCTipsDatabase | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Matte QC Inspector")
        self.setModal(True)
        self.setMinimumSize(640, 480)
        self.setSizeGripEnabled(True)
        self.resize(1040, 760)
        self._frame = frame_rgb
        self._before_mask = before_mask
        self._after_mask = after_mask if after_mask is not None else before_mask
        self._before_report = analyze_matte(frame_rgb, before_mask, previous_mask)
        self._after_report = analyze_matte(frame_rgb, self._after_mask, previous_mask)
        self._comparison = compare_matte_reports(self._before_report, self._after_report)
        self._profile = profile
        self._tips_database = tips_database or QCTipsDatabase()
        self._analysis = analyze_qc_image(
            frame_rgb,
            self._after_mask,
            self._after_report,
            previous_mask,
            profile,
        )
        if self._before_report.mask.shape == self._after_report.mask.shape:
            self._delta_pixels = int(
                np.count_nonzero(self._before_report.mask != self._after_report.mask)
            )
        else:
            self._delta_pixels = 0
        self._build_ui()

    @property
    def before_report(self) -> MatteQCReport:
        """Return the report for the before matte."""
        return self._before_report

    @property
    def after_report(self) -> MatteQCReport:
        """Return the report for the after matte."""
        return self._after_report

    @property
    def comparison(self):
        """Return the before/after comparison."""
        return self._comparison

    @property
    def delta_pixels(self) -> int:
        """Return the number of pixels changed by the refinement pass."""
        return self._delta_pixels

    @property
    def masks_changed(self) -> bool:
        """Return whether the refinement changed the binary matte."""
        return self._delta_pixels > 0

    @property
    def views_linked(self) -> bool:
        """Return whether zoom and pan are mirrored between panes."""
        return self._views_linked

    @property
    def analysis(self) -> QCAnalysis:
        """Return the automated frame/matte guidance."""
        return self._analysis

    @property
    def tips_database(self) -> QCTipsDatabase:
        """Return the local question/answer database."""
        return self._tips_database

    def _source_image(self) -> np.ndarray:
        if self._frame is None:
            return np.zeros((320, 480, 3), dtype=np.uint8)
        frame = np.asarray(self._frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return np.zeros((320, 480, 3), dtype=np.uint8)
        return np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)

    def source_preview_image(self) -> np.ndarray:
        """Return the source-frame image shown in the left pane."""
        return self._source_image()

    def current_preview_image(self) -> np.ndarray:
        """Return the current-matte image shown in the right pane."""
        return self._preview_image(self._after_mask, self._after_report)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(8)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        title = QLabel("Matte QC")
        title.setStyleSheet("font-size: 17px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(title)
        intro = QLabel(
            "Source on the left · current matte + QC evidence on the right. "
            "Wheel zooms; drag pans; Link views mirrors both."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9a9aa4; font-size: 11px;")
        root.addWidget(intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Zoom"))
        self._zoom_left = QPushButton("−")
        self._zoom_right = QPushButton("+")
        self._zoom_left.clicked.connect(lambda: self._zoom_by(0.8))
        self._zoom_right.clicked.connect(lambda: self._zoom_by(1.25))
        controls.addWidget(self._zoom_left)
        controls.addWidget(self._zoom_right)
        self._zoom = QSlider(Qt.Orientation.Horizontal)
        self._zoom.setRange(25, 800)
        self._zoom.setValue(100)
        self._zoom.valueChanged.connect(self._on_zoom_changed)
        controls.addWidget(self._zoom)
        self._btn_link_views = QPushButton("Link views")
        self._btn_link_views.setCheckable(True)
        self._btn_link_views.setChecked(True)
        self._btn_link_views.setToolTip(
            "Keep zoom and panning mirrored between the source and QC views"
        )
        controls.addWidget(self._btn_link_views)
        self._btn_reset_views = QPushButton("Reset view")
        self._btn_reset_views.clicked.connect(self._reset_views)
        controls.addWidget(self._btn_reset_views)
        controls.addStretch(1)
        root.addLayout(controls)

        self._issue_checks: dict[str, QCheckBox] = {}
        checks = QHBoxLayout()
        for key, label in (
            ("hair_detail", "Hair / detail"),
            ("background_spill", "Background spill"),
            ("holes", "Holes"),
            ("edge_flicker", "Edge flicker"),
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._refresh_previews)
            self._issue_checks[key] = checkbox
            checks.addWidget(checkbox)
        checks.addStretch()
        root.addLayout(checks)

        previews = QHBoxLayout()
        self._before_preview = MattePreview(self._source_image())
        self._after_preview = MattePreview(self._preview_image(self._after_mask, self._after_report))
        self._views_linked = True
        self._active_preview = self._before_preview
        self._syncing_views = False
        self._btn_link_views.toggled.connect(self._set_views_linked)
        self._before_preview.activated.connect(self._on_preview_activated)
        self._after_preview.activated.connect(self._on_preview_activated)
        self._before_preview.view_changed.connect(
            lambda zoom, x, y: self._on_view_changed(self._before_preview, zoom, x, y)
        )
        self._after_preview.view_changed.connect(
            lambda zoom, x, y: self._on_view_changed(self._after_preview, zoom, x, y)
        )
        before_column = QVBoxLayout()
        before_column.addWidget(QLabel("Source frame"))
        before_column.addWidget(self._before_preview, stretch=1)
        after_column = QVBoxLayout()
        after_column.addWidget(QLabel("Current matte + QC overlay"))
        after_column.addWidget(self._after_preview, stretch=1)
        previews.addLayout(before_column, stretch=1)
        previews.addLayout(after_column, stretch=1)
        root.addLayout(previews, stretch=1)

        metrics = QGridLayout()
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.addWidget(QLabel("QC"), 0, 0)
        metrics.addWidget(QLabel("Raw"), 0, 1)
        metrics.addWidget(QLabel("Now"), 0, 2)
        for row, (key, label) in enumerate(
            (("hair_detail", "Hair"), ("background_spill", "Spill"), ("holes", "Holes"), ("edge_flicker", "Flicker")),
            start=1,
        ):
            metrics.addWidget(QLabel(label), row, 0)
            before = self._before_report.issue(key)
            after = self._after_report.issue(key)
            metrics.addWidget(QLabel(self._metric(before)), row, 1)
            metrics.addWidget(QLabel(self._metric(after)), row, 2)
        root.addLayout(metrics)

        delta_text = (
            f"{self._delta_pixels:,} pixel(s) changed by refinement"
            if self._delta_pixels
            else "No mask pixels changed by refinement; the QC overlay is still active"
        )
        summary = QLabel(f"{self._comparison.summary} · {delta_text}")
        summary.setStyleSheet("color: #8b8b94; font-size: 11px;")
        root.addWidget(summary)

        coach = QGroupBox("QC Coach · local analysis")
        coach.setStyleSheet(
            "QGroupBox { color: #d4d4d8; font-size: 12px; font-weight: 600; "
            "border: 1px solid #3a3a40; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        coach_layout = QVBoxLayout(coach)
        coach_layout.setSpacing(6)
        self._coach_summary = QLabel(self._analysis.summary)
        self._coach_summary.setWordWrap(True)
        self._coach_summary.setStyleSheet("color: #9a9aa4; font-size: 11px;")
        coach_layout.addWidget(self._coach_summary)
        self._coach_tips = QLabel(
            "\n".join(f"• {self._tip_summary(tip)}" for tip in self._analysis.tips[:3])
        )
        self._coach_tips.setWordWrap(True)
        self._coach_tips.setStyleSheet("color: #d0cfe0; font-size: 11px;")
        coach_layout.addWidget(self._coach_tips)
        self._coach_settings = QLabel(self._format_settings(self._analysis.settings))
        self._coach_settings.setWordWrap(True)
        self._coach_settings.setStyleSheet("color: #8b8b94; font-size: 10px;")
        self._coach_settings.setVisible(bool(self._analysis.settings))
        coach_layout.addWidget(self._coach_settings)

        question_row = QHBoxLayout()
        self._question = QLineEdit()
        self._question.setPlaceholderText("Describe what looks wrong, e.g. hair is too smooth")
        self._question.returnPressed.connect(self._ask_question)
        self._question_choice = QComboBox()
        self._question_choice.addItems(
            [
                "Hair is way too smooth and not sharp",
                "The hair is too sharp or jagged",
                "The edge flickers between frames",
                "There are holes inside the mask",
                "There is a colored fringe or spill",
            ]
        )
        self._question_choice.currentTextChanged.connect(self._question.setText)
        question_row.addWidget(self._question_choice, stretch=1)
        question_row.addWidget(self._question, stretch=2)
        ask = QPushButton("Ask")
        ask.setProperty("cssClass", "accent")
        ask.clicked.connect(self._ask_question)
        question_row.addWidget(ask)
        self._save_tip = QPushButton("Save as custom tip")
        self._save_tip.setToolTip("Store this answer in the local QC tip database")
        self._save_tip.clicked.connect(self._save_custom_tip)
        question_row.addWidget(self._save_tip)
        coach_layout.addLayout(question_row)

        self._answer = QLabel("Ask a question or choose a common complaint above.")
        self._answer.setWordWrap(True)
        self._answer.setStyleSheet("color: #a0a0a8; font-size: 11px;")
        coach_layout.addWidget(self._answer)

        self._apply_suggestion = QPushButton("Apply suggestion to current shot")
        self._apply_suggestion.setEnabled(bool(self._analysis.settings))
        self._apply_suggestion.clicked.connect(self._apply_suggestion_clicked)
        coach_layout.addWidget(self._apply_suggestion, alignment=Qt.AlignmentFlag.AlignRight)
        root.addWidget(coach)

        actions = QHBoxLayout()
        actions.addStretch()
        close = QPushButton("Close")
        close.setProperty("cssClass", "accent")
        close.clicked.connect(self.accept)
        actions.addWidget(close)
        root.addLayout(actions)

    @staticmethod
    def _metric(issue) -> str:
        return f"{issue.score:.2f} · {issue.count:,} px"

    @staticmethod
    def _format_settings(settings: dict[str, object]) -> str:
        if not settings:
            return "No automatic setting change suggested."
        parts: list[str] = []
        if "edge_smoothness" in settings:
            parts.append(f"Edge smoothness {settings['edge_smoothness']}/10")
        if "detail_preservation" in settings:
            parts.append(f"Detail preservation {float(settings['detail_preservation']):.0%}")
        if "temporal_smoothing" in settings:
            parts.append("Temporal smoothing on" if settings["temporal_smoothing"] else "Temporal smoothing off")
        if "temporal_window" in settings:
            parts.append(f"Temporal window {settings['temporal_window']}")
        return "Suggested: " + " · ".join(parts) if parts else "No automatic setting change suggested."

    @classmethod
    def _tip_summary(cls, tip) -> str:
        settings = cls._format_settings(tip.settings)
        if settings.startswith("Suggested: "):
            return f"{tip.title} — {settings.removeprefix('Suggested: ')}"
        return tip.title

    def _ask_question(self) -> None:
        question = self._question.text().strip() or self._question_choice.currentText()
        answer = self._tips_database.ask(
            question,
            report=self._after_report,
            profile=self._profile,
        )
        source = "saved locally" if answer.source != "analysis" else "automated analysis"
        self._answer.setText(f"{answer.answer}\n\nSource: {source}.")

    def _save_custom_tip(self) -> None:
        question = self._question.text().strip() or self._question_choice.currentText()
        answer = self._tips_database.ask(
            question,
            report=self._after_report,
            profile=self._profile,
        )
        key = f"custom_{len(self._tips_database.custom_tips) + 1}"
        self._tips_database.add(
            question,
            answer.answer,
            key=key,
            variants=(question,),
        )
        self._answer.setText(f"{answer.answer}\n\nSaved as a custom tip in the local database.")

    def _apply_suggestion_clicked(self) -> None:
        if self._analysis.settings:
            self.apply_suggestion_requested.emit(dict(self._analysis.settings))

    def _preview_image(self, mask, report) -> np.ndarray:
        if self._frame is None:
            return np.zeros((320, 480, 3), dtype=np.uint8)
        visible = MatteQCReport(
            issues={
                name: issue
                for name, issue in report.issues.items()
                if self._issue_checks.get(name, None)
                and self._issue_checks[name].isChecked()
            },
            shape=report.shape,
            mask=report.mask,
            previous_mask=report.previous_mask,
            overall_score=report.overall_score,
            overall_warning=report.overall_warning,
            valid=report.valid,
            empty=report.empty,
            message=report.message,
            warnings=report.warnings,
        )
        return render_mask_overlay(
            self._frame,
            mask,
            visible,
            issue_alpha=0.42 if self._any_issues_selected() else 0.0,
        )

    def _any_issues_selected(self) -> bool:
        return any(box.isChecked() for box in self._issue_checks.values())

    def _refresh_previews(self) -> None:
        self._before_preview.set_image(self._source_image())
        self._after_preview.set_image(self._preview_image(self._after_mask, self._after_report))

    def _zoom_by(self, factor: float) -> None:
        self._zoom.setValue(int(self._zoom.value() * factor))

    def _on_zoom_changed(self, value: int) -> None:
        zoom = max(0.25, value / 100.0)
        self._active_preview.set_zoom(zoom)
        self._sync_linked_view(self._active_preview)

    def _set_views_linked(self, linked: bool) -> None:
        self._views_linked = bool(linked)
        if self._views_linked:
            self._sync_linked_view(self._active_preview)

    def _on_preview_activated(self, preview) -> None:
        self._active_preview = preview
        self._zoom.blockSignals(True)
        self._zoom.setValue(int(max(25, min(800, preview.zoom() * 100))))
        self._zoom.blockSignals(False)

    def _on_view_changed(self, source, zoom: float, x: int, y: int) -> None:
        if self._syncing_views:
            return
        self._active_preview = source
        if self._views_linked:
            self._sync_linked_view(source, zoom=zoom, pan=(x, y))
        self._zoom.blockSignals(True)
        self._zoom.setValue(int(max(25, min(800, zoom * 100))))
        self._zoom.blockSignals(False)

    def _sync_linked_view(self, source, zoom: float | None = None, pan: tuple[int, int] | None = None) -> None:
        if not self._views_linked:
            return
        other = self._after_preview if source is self._before_preview else self._before_preview
        self._syncing_views = True
        try:
            other.set_view(
                source.zoom() if zoom is None else zoom,
                source.pan() if pan is None else pan,
                notify=False,
            )
        finally:
            self._syncing_views = False

    def _reset_views(self) -> None:
        self._syncing_views = True
        try:
            self._before_preview.set_view(1.0, (0, 0), notify=False)
            self._after_preview.set_view(1.0, (0, 0), notify=False)
        finally:
            self._syncing_views = False
        self._zoom.blockSignals(True)
        self._zoom.setValue(100)
        self._zoom.blockSignals(False)


__all__ = ["MatteInspectorDialog", "MattePreview"]
