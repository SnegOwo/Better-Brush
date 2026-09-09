"""Main application window: orchestrates video, canvas, SAM 2, timeline, export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.config import AppConfig
from src.hardware import ContentProfile, HardwareDetector, SAM2Profiler
from src.models.sam2_manager import MaskResult, PointPrompt, SAM2Manager
from src.processing.frame_buffer import FrameBuffer
from src.processing.inference_worker import InferenceWorker, PropagationWorker
from src.processing.video_reader import VideoReader
from src.ui.timeline import Timeline
from src.ui.video_canvas import VideoCanvas
from src.export.exporter import Exporter

logger = logging.getLogger(__name__)


class ExportWorker(QThread):
    """Run PNG/ffmpeg export off the GUI thread."""

    finished = Signal(object)  # Path or Exception
    progress = Signal(str)

    def __init__(
        self,
        exporter: Exporter,
        kind: str,
        masks: dict[int, np.ndarray],
        parent: Optional[QWidget] = None,
        frames: Optional[dict[int, np.ndarray]] = None,
        fps: float = 30.0,
        base_name: str = "output",
    ) -> None:
        super().__init__(parent)
        self._exporter = exporter
        self._kind = kind
        self._masks = masks
        self._frames = frames or {}
        self._fps = fps
        self._base_name = base_name

    def run(self) -> None:
        try:
            self.progress.emit("Exporting PNG masks..." if self._kind == "png" else "Encoding video...")
            if self._kind == "png":
                dest = self._exporter.export_png_sequence(self._masks, base_name=self._base_name)
            else:
                dest = self._exporter.export_transparent_video(
                    self._frames, self._masks, self._fps, base_name=self._base_name
                )
            self.finished.emit(dest)
        except Exception as exc:  # noqa: BLE001
            logger.error("Export failed: %s", exc)
            self.finished.emit(exc)


class MainWindow(QMainWindow):
    """Better Brush main window."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Better Brush — SAM 2 Rotoscoping")
        self.resize(1400, 900)

        # State
        self._config = AppConfig()
        self._video_reader: Optional[VideoReader] = None
        self._frame_buffer: Optional[FrameBuffer] = None
        self._current_frame_idx = 0
        self._is_playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._points: list[PointPrompt] = []
        self._masks: dict[int, np.ndarray] = {}  # frame_idx -> bool mask
        self._keyframes: set[int] = set()
        self._sam_manager: Optional[SAM2Manager] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._propagation_worker: Optional[PropagationWorker] = None
        self._export_worker: Optional[ExportWorker] = None

        # Build UI
        self._setup_menu()
        self._setup_central_widget()
        self._setup_status_bar()

        # Hardware recommendation on startup
        self._auto_detect_hardware()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _setup_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_act = QAction("&Open Video...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_video)
        file_menu.addAction(open_act)

        export_menu = menubar.addMenu("&Export")
        export_png = QAction("Export &PNG Masks...", self)
        export_png.triggered.connect(self._on_export_png)
        export_menu.addAction(export_png)
        export_mov = QAction("Export &Transparent Video...", self)
        export_mov.triggered.connect(self._on_export_mov)
        export_menu.addAction(export_mov)

    def _setup_central_widget(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self._btn_detect = QPushButton("Auto-Detect Hardware")
        self._btn_detect.clicked.connect(self._auto_detect_hardware)
        left_layout.addWidget(self._btn_detect)

        self._lbl_hw = QLabel("Hardware: detecting...")
        self._lbl_hw.setWordWrap(True)
        left_layout.addWidget(self._lbl_hw)

        self._btn_segment = QPushButton("Segment Current Frame (Space)")
        self._btn_segment.clicked.connect(self._run_segmentation)
        left_layout.addWidget(self._btn_segment)

        self._btn_propagate = QPushButton("Propagate to End")
        self._btn_propagate.clicked.connect(self._run_propagation)
        left_layout.addWidget(self._btn_propagate)

        self._btn_clear_points = QPushButton("Clear Points")
        self._btn_clear_points.clicked.connect(self._clear_points)
        left_layout.addWidget(self._btn_clear_points)

        self._btn_clear_mask = QPushButton("Clear Mask")
        self._btn_clear_mask.clicked.connect(self._clear_mask)
        left_layout.addWidget(self._btn_clear_mask)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right: canvas + timeline
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._canvas = VideoCanvas()
        self._canvas.point_added.connect(self._on_canvas_point)
        self._canvas.frame_navigate.connect(self._on_navigate)
        self._canvas.request_segment.connect(self._run_segmentation)
        self._canvas.play_pause.connect(self._toggle_play)
        right_layout.addWidget(self._canvas, stretch=1)

        self._timeline = Timeline()
        self._timeline.frame_selected.connect(self._set_frame)
        right_layout.addWidget(self._timeline)

        splitter.addWidget(right)
        splitter.setSizes([260, 1140])
        self.setCentralWidget(splitter)

    def _setup_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ------------------------------------------------------------------ #
    # Hardware
    # ------------------------------------------------------------------ #
    def _auto_detect_hardware(self) -> None:
        self._status.showMessage("Detecting hardware...")
        QApplication.processEvents()
        try:
            profile = HardwareDetector().detect()
            rec = SAM2Profiler().recommend(profile, content=ContentProfile.GENERAL)
            self._config.recommendation = rec
            self._lbl_hw.setText(rec.user_message.replace("**", ""))
            self._status.showMessage(f"Hardware OK — {rec.model_size.value} on {rec.device}")
            logger.info("Hardware detection complete: %s", rec)
        except Exception as exc:
            logger.error("Hardware detection failed: %s", exc)
            self._status.showMessage(f"HW detection failed: {exc}")

    # ------------------------------------------------------------------ #
    # Video loading
    # ------------------------------------------------------------------ #
    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        if not path:
            return
        self._load_video(Path(path))

    def _load_video(self, path: Path) -> None:
        try:
            if self._video_reader is not None:
                self._frame_buffer.close()
            self._video_reader = VideoReader(path)
            self._frame_buffer = FrameBuffer(self._video_reader)
            self._current_frame_idx = 0
            self._masks.clear()
            self._keyframes.clear()
            self._points.clear()
            self._timeline.set_duration(self._video_reader.total_frames)
            self._timeline.set_current_frame(0)
            self._show_current_frame()
            self._status.showMessage(
                f"Loaded {path.name} — {self._video_reader.width}x{self._video_reader.height} "
                f"@ {self._video_reader.fps:.2f} fps"
            )
            # SAM 2 video state is initialised lazily by the propagation
            # worker (frame extraction + model load must not block the GUI).
        except Exception as exc:
            logger.error("Failed to load video: %s", exc)
            QMessageBox.critical(self, "Error", f"Cannot load video:\n{exc}")

    # ------------------------------------------------------------------ #
    # Frame navigation
    # ------------------------------------------------------------------ #
    def _show_current_frame(self) -> None:
        if self._frame_buffer is None:
            return
        frame = self._frame_buffer.get(self._current_frame_idx)
        if frame is not None:
            self._canvas.set_frame(frame)
            mask = self._masks.get(self._current_frame_idx)
            self._canvas.set_mask(mask)
            self._canvas.set_points(self._points)
            self._timeline.set_current_frame(self._current_frame_idx)

    def _set_frame(self, idx: int) -> None:
        self._current_frame_idx = max(0, idx)
        self._show_current_frame()

    def _on_navigate(self, delta: int) -> None:
        self._set_frame(self._current_frame_idx + delta)

    def _toggle_play(self) -> None:
        if self._is_playing:
            self._play_timer.stop()
            self._is_playing = False
            self._status.showMessage("Paused")
        else:
            if self._video_reader is None:
                return
            interval_ms = int(1000 / self._video_reader.fps) if self._video_reader.fps > 0 else 33
            self._play_timer.start(interval_ms)
            self._is_playing = True
            self._status.showMessage("Playing")

    def _on_play_tick(self) -> None:
        if self._video_reader is None:
            return
        if self._current_frame_idx + 1 >= self._video_reader.total_frames:
            self._toggle_play()
            return
        self._current_frame_idx += 1
        self._show_current_frame()

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def _on_canvas_point(self, x: int, y: int, label: int) -> None:
        self._points.append(PointPrompt(x=x, y=y, label=label))
        self._canvas.set_points(self._points)
        self._status.showMessage(f"Point added ({x},{y}) label={label}")

    def _clear_points(self) -> None:
        self._points.clear()
        self._canvas.clear_points()

    def _clear_mask(self) -> None:
        self._canvas.clear_mask()

    # ------------------------------------------------------------------ #
    # SAM 2 manager / segmentation (image predictor)
    # ------------------------------------------------------------------ #
    def _ensure_sam_manager(self) -> SAM2Manager:
        """Create the shared SAM 2 manager once (weights loaded lazily)."""
        if self._sam_manager is None:
            if self._config.recommendation is None:
                raise RuntimeError("No model recommendation available")
            self._sam_manager = SAM2Manager(self._config.recommendation)
        return self._sam_manager

    def _run_segmentation(self) -> None:
        if self._frame_buffer is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._config.recommendation is None:
            QMessageBox.warning(self, "No model", "Run hardware detection first.")
            return
        if not self._points:
            QMessageBox.warning(
                self, "No prompt", "Click on the frame (left = include, right = exclude) first."
            )
            return
        frame = self._frame_buffer.get(self._current_frame_idx)
        if frame is None:
            return
        self._status.showMessage("Segmenting...")
        self.setEnabled(False)

        self._inference_worker = InferenceWorker(
            self._config.recommendation, sam_manager=self._ensure_sam_manager(), parent=self
        )
        self._inference_worker.set_task(frame, self._points)
        self._inference_worker.progress.connect(
            lambda msg: self._status.showMessage(msg)
        )
        self._inference_worker.finished.connect(self._on_segment_finished)
        self._inference_worker.start()

    def _on_segment_finished(self, result: object) -> None:
        self.setEnabled(True)
        if isinstance(result, Exception):
            logger.error("Segmentation failed: %s", result)
            QMessageBox.critical(self, "Segmentation Error", str(result))
            self._status.showMessage("Segmentation failed")
            return
        results: list[MaskResult] = result
        if results:
            best = max(results, key=lambda m: m.score)
            self._masks[self._current_frame_idx] = best.mask
            self._keyframes.add(self._current_frame_idx)
            self._canvas.set_mask(best.mask)
            self._timeline.set_keyframes(self._keyframes)
            self._status.showMessage(f"Mask generated (score={best.score:.3f})")
        else:
            self._status.showMessage("No mask found")

    # ------------------------------------------------------------------ #
    # Propagation (video predictor)
    # ------------------------------------------------------------------ #
    def _run_propagation(self) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._config.recommendation is None:
            QMessageBox.warning(self, "No model", "Run hardware detection first.")
            return
        if not self._points and self._current_frame_idx not in self._masks:
            QMessageBox.warning(self, "No prompt", "Add points or segment current frame first.")
            return
        if self._propagation_worker is not None and self._propagation_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A propagation is already running.")
            return

        # NOTE: the heavy work (frame extraction, model load, tracking) runs
        # inside the worker thread — the GUI thread only wires signals here.
        self._propagation_worker = PropagationWorker(
            self._config.recommendation, self._ensure_sam_manager(), self
        )
        self._propagation_worker.set_task(
            str(self._video_reader.path),
            start_frame=self._current_frame_idx,
            points=list(self._points) or None,
            total_frames=self._video_reader.total_frames,
        )
        self._propagation_worker.progress.connect(self._on_worker_progress)
        self._propagation_worker.frame_ready.connect(self._on_propagation_frame_ready)
        self._propagation_worker.finished.connect(self._on_propagation_finished)
        self._status.showMessage("Propagating...")
        self.setEnabled(False)
        self._propagation_worker.start()

    def _on_worker_progress(self, stage: str, current: int, total: int) -> None:
        if total > 0:
            self._status.showMessage(f"{stage}... {current}/{total}")
        else:
            self._status.showMessage(stage)

    def _on_propagation_frame_ready(self, frame_idx: int, _obj_id: int, mask: np.ndarray) -> None:
        """Called for every propagated frame (live update)."""
        self._masks[frame_idx] = mask
        self._keyframes.add(frame_idx)
        self._timeline.set_keyframes(self._keyframes)
        if frame_idx == self._current_frame_idx:
            self._canvas.set_mask(mask)

    def _on_propagation_finished(self, result: object) -> None:
        self.setEnabled(True)
        if isinstance(result, Exception):
            logger.error("Propagation failed: %s", result)
            QMessageBox.critical(self, "Propagation Error", str(result))
            self._status.showMessage("Propagation failed")
            return
        results: list[tuple[int, int, np.ndarray, np.ndarray]] = result
        # frame_ready already stored everything; replay just in case
        for frame_idx, _obj_id, mask, _score in results:
            self._masks[frame_idx] = mask
            self._keyframes.add(frame_idx)
        self._timeline.set_keyframes(self._keyframes)
        self._show_current_frame()
        n_frames = len(set(f for f, _, _, _ in results))
        self._status.showMessage(f"Propagation complete — {n_frames} frames masked")

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _on_export_png(self) -> None:
        if not self._masks:
            QMessageBox.warning(self, "No masks", "Generate masks before exporting.")
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.warning(self, "Busy", "An export is already running.")
            return
        self._export_worker = ExportWorker(Exporter(), "png", dict(self._masks), self)
        self._export_worker.progress.connect(self._status.showMessage)
        self._export_worker.finished.connect(self._on_export_finished)
        self.setEnabled(False)
        self._export_worker.start()

    def _on_export_mov(self) -> None:
        if self._video_reader is None or self._frame_buffer is None or not self._masks:
            QMessageBox.warning(self, "No data", "Open video and generate masks first.")
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.warning(self, "Busy", "An export is already running.")
            return
        frames = {}
        for idx in sorted(self._masks.keys()):
            f = self._frame_buffer.get(idx)
            if f is not None:
                frames[idx] = f
        self._export_worker = ExportWorker(
            Exporter(),
            "video",
            dict(self._masks),
            self,
            frames=frames,
            fps=self._video_reader.fps,
            base_name="output",
        )
        self._export_worker.progress.connect(self._status.showMessage)
        self._export_worker.finished.connect(self._on_export_finished)
        self.setEnabled(False)
        self._export_worker.start()

    def _on_export_finished(self, result: object) -> None:
        self.setEnabled(True)
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Export Error", str(result))
            return
        QMessageBox.information(self, "Export OK", f"Saved to:\n{result}")

    # ------------------------------------------------------------------ #
    # Close
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: N802
        if self._propagation_worker is not None and self._propagation_worker.isRunning():
            self._propagation_worker.cancel()
            self._propagation_worker.wait(5000)
        for worker in (self._inference_worker, self._export_worker):
            if worker is not None and worker.isRunning():
                worker.wait(5000)
        if self._sam_manager is not None:
            self._sam_manager.unload()
        if self._frame_buffer is not None:
            self._frame_buffer.close()
        event.accept()
