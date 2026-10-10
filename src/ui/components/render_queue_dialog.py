"""Persistent batch render queue window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.export.render_queue import (
    QueueState,
    RenderPreset,
    RenderQueue,
    RenderQueueError,
)


class RenderQueueDialog(QDialog):
    """Manage persistent render jobs and start them sequentially."""

    def __init__(
        self,
        parent: QWidget | None = None,
        queue: RenderQueue | None = None,
        current_project: Path | None = None,
        current_video: Path | None = None,
        default_settings: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Render Queue")
        self.setModal(False)
        self.resize(820, 580)
        self._queue = queue or RenderQueue(
            Path.home() / ".betterbrush" / "render_queue.json"
        )
        self._current_project = Path(current_project) if current_project else None
        self._current_video = Path(current_video) if current_video else None
        self._default_settings = dict(default_settings or {})
        self._build_ui()
        self._refresh()
        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    @property
    def queue(self) -> RenderQueue:
        """Return the persistent queue service."""
        return self._queue

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)
        title = QLabel("Batch Render Queue")
        title.setStyleSheet("font-size: 17px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(title)
        intro = QLabel(
            "Queue saved .bbproj clips, run them one at a time, and retry failures without "
            "losing completed work. Jobs persist between sessions."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9a9aa4; font-size: 11px;")
        root.addWidget(intro)

        form = QFormLayout()
        self._output = QLineEdit(str(self._default_settings.get("output_dir", Path.cwd() / "exports")))
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output, stretch=1)
        output_row.addWidget(browse_output)
        form.addRow("Final destination", output_row)
        self._preset = QComboBox()
        for name in sorted(self._queue.presets):
            self._preset.addItem(name, name)
        form.addRow("Preset", self._preset)
        root.addLayout(form)

        add_row = QHBoxLayout()
        add_project = QPushButton("Add Project…")
        add_project.clicked.connect(self._add_project)
        add_row.addWidget(add_project)
        self._add_current = QPushButton("Add Current Project")
        self._add_current.setEnabled(self._current_project is not None)
        self._add_current.clicked.connect(self._add_current_project)
        add_row.addWidget(self._add_current)
        save_preset = QPushButton("Save Settings as Preset")
        save_preset.clicked.connect(self._save_preset)
        add_row.addWidget(save_preset)
        add_row.addStretch()
        root.addLayout(add_row)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._selection_changed)
        root.addWidget(self._list, stretch=1)

        detail = QLabel("Select a job to inspect its state.")
        detail.setWordWrap(True)
        detail.setStyleSheet("color: #a0a0a8; font-size: 11px;")
        root.addWidget(detail)
        self._detail = detail
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        root.addWidget(self._progress)

        controls = QHBoxLayout()
        self._start = QPushButton("Start Queue")
        self._start.setProperty("cssClass", "accent")
        self._start.clicked.connect(self._start_queue)
        controls.addWidget(self._start)
        self._pause = QPushButton("Pause")
        self._pause.clicked.connect(self._pause_job)
        controls.addWidget(self._pause)
        self._resume = QPushButton("Resume")
        self._resume.clicked.connect(self._resume_job)
        controls.addWidget(self._resume)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._cancel_job)
        controls.addWidget(self._cancel)
        self._retry = QPushButton("Retry")
        self._retry.clicked.connect(self._retry_job)
        controls.addWidget(self._retry)
        self._remove = QPushButton("Remove")
        self._remove.clicked.connect(self._remove_job)
        controls.addWidget(self._remove)
        clear = QPushButton("Clear Finished")
        clear.clicked.connect(self._clear_finished)
        controls.addWidget(clear)
        controls.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        controls.addWidget(close)
        root.addLayout(controls)

    def _refresh(self) -> None:
        selected_id = self._selected_id()
        self._list.clear()
        for job in self._queue.jobs:
            item = QListWidgetItem(
                f"{job.name}  ·  {job.state.value}  ·  {int(job.progress * 100):3d}%"
            )
            item.setData(Qt.ItemDataRole.UserRole, job.id)
            item.setToolTip(
                f"{job.project_path}\n→ {job.output_destination}\n"
                f"Preset: {job.preset_name}\nAttempts: {job.attempts}"
            )
            self._list.addItem(item)
            if job.id == selected_id:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        self._update_controls()

    def _selected_id(self) -> str | None:
        item = self._list.currentItem()
        return None if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def _selected_job(self):
        selected = self._selected_id()
        if selected is None:
            return None
        try:
            return self._queue.get_job(selected)
        except RenderQueueError:
            return None

    def _selection_changed(self, _current, _previous) -> None:
        self._update_controls()

    def _update_controls(self) -> None:
        job = self._selected_job()
        if job is None:
            self._detail.setText("Select a job to inspect its state.")
            self._progress.setValue(0)
            for button in (self._pause, self._resume, self._cancel, self._retry, self._remove):
                button.setEnabled(False)
            return
        self._detail.setText(
            f"{job.name}  ·  {job.stage}  ·  {job.completed:,}/{job.total:,} frames\n"
            f"{job.project_path} → {job.output_destination}"
            + (f"\nLast error: {job.last_error}" if job.last_error else "")
        )
        self._progress.setValue(int(job.progress * 100))
        self._pause.setEnabled(job.state in {QueueState.QUEUED, QueueState.RUNNING})
        self._resume.setEnabled(job.state is QueueState.PAUSED)
        self._cancel.setEnabled(
            job.state in {QueueState.QUEUED, QueueState.RUNNING, QueueState.PAUSED}
        )
        self._retry.setEnabled(job.state in {QueueState.FAILED, QueueState.CANCELLED})
        self._remove.setEnabled(job.state is not QueueState.RUNNING)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Final Export Destination", self._output.text())
        if path:
            self._output.setText(path)

    def _preset_value(self) -> RenderPreset:
        name = str(self._preset.currentData() or self._preset.currentText() or "Default")
        try:
            return self._queue.get_preset(name)
        except RenderQueueError:
            return RenderPreset(name=name)

    def _preset_from_settings(self, name: str) -> RenderPreset:
        settings = self._default_settings
        fmt = str(settings.get("format", "prores"))
        resolution = settings.get("resolution")
        background = settings.get("background")
        return RenderPreset(
            name=name,
            output_type="png" if fmt == "png" else "video",
            format=fmt,
            base_name=str(settings.get("file_name", "render")),
            fps=float(settings.get("fps")) if settings.get("fps") else None,
            resolution=tuple(resolution) if resolution else None,
            include_audio=bool(settings.get("include_audio", False)),
            background=tuple(background) if background else None,
            object_mode=str(settings.get("object_mode", "combined")),
            object_id=settings.get("object_id"),
            range_mode=str(settings.get("range_mode", "all")),
            start_frame=int(settings.get("start_frame", 0)),
            end_frame=int(settings.get("end_frame", 0)),
        )

    def _add_job(self, project: Path, name: str | None = None) -> None:
        try:
            self._queue.enqueue(
                project,
                Path(self._output.text().strip() or Path.cwd() / "exports"),
                self._preset_value(),
                name=name or project.stem,
                video_path=self._current_video if project == self._current_project else None,
            )
            self._refresh()
        except (OSError, RenderQueueError, ValueError) as exc:
            QMessageBox.warning(self, "Queue error", str(exc))

    def _add_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add Project to Queue",
            "",
            "Better Brush Project (*.bbproj);;All Files (*)",
        )
        if path:
            self._add_job(Path(path))

    def _add_current_project(self) -> None:
        if self._current_project is not None:
            self._add_job(self._current_project, self._current_project.stem)

    def _save_preset(self) -> None:
        name, ok = self._ask_name()
        if not ok:
            return
        try:
            preset = self._preset_from_settings(name)
            self._queue.save_preset(preset, overwrite=True)
            self._preset.clear()
            for preset_name in sorted(self._queue.presets):
                self._preset.addItem(preset_name, preset_name)
            self._preset.setCurrentIndex(self._preset.findData(name))
        except (RenderQueueError, ValueError) as exc:
            QMessageBox.warning(self, "Preset error", str(exc))

    def _ask_name(self) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name")
        return name.strip(), bool(ok and name.strip())

    def _start_queue(self) -> None:
        try:
            self._queue.start()
        except RenderQueueError as exc:
            QMessageBox.warning(self, "Queue error", str(exc))
        self._refresh()

    def _transition(self, action) -> None:
        job = self._selected_job()
        if job is None:
            return
        try:
            action(job.id)
        except (RenderQueueError, OSError) as exc:
            QMessageBox.warning(self, "Queue error", str(exc))
        self._refresh()

    def _pause_job(self) -> None:
        self._transition(self._queue.pause)

    def _resume_job(self) -> None:
        self._transition(self._queue.resume)

    def _cancel_job(self) -> None:
        self._transition(self._queue.cancel)

    def _retry_job(self) -> None:
        self._transition(self._queue.retry)

    def _remove_job(self) -> None:
        self._transition(self._queue.remove)

    def _clear_finished(self) -> None:
        self._queue.clear({QueueState.COMPLETED, QueueState.FAILED, QueueState.CANCELLED})
        self._refresh()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)


__all__ = ["RenderQueueDialog"]
