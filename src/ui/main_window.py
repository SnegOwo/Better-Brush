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
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.config import AppConfig
from src.hardware import ContentProfile, HardwareDetector, ModelSize, SAM2Profiler
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
        self._play_start = 0
        self._syncing = False

        # Build UI
        self._setup_menu()
        self._setup_central_widget()
        self._setup_toolbar()
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
        self._btn_propagate.clicked.connect(lambda: self._run_propagation(1))
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
        self._canvas.mask_edited.connect(self._on_canvas_mask_edited)
        right_layout.addWidget(self._canvas, stretch=1)

        self._timeline = Timeline()
        self._timeline.frame_selected.connect(self._set_frame)
        right_layout.addWidget(self._timeline)

        splitter.addWidget(right)
        splitter.setSizes([260, 1140])

        central = QWidget()
        self._central_layout = QVBoxLayout(central)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self._central_layout.setSpacing(0)
        self._central_layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _setup_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ------------------------------------------------------------------ #
    # Transport / tools toolbar
    # ------------------------------------------------------------------ #
    def _setup_toolbar(self) -> None:
        tb = QWidget(self)
        lay = QHBoxLayout(tb)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        # --- Playback transport ---
        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.clicked.connect(self._toggle_play)
        lay.addWidget(self._btn_play)

        self._btn_stop = QPushButton("⏹ Stop")
        self._btn_stop.clicked.connect(self._stop_playback)
        lay.addWidget(self._btn_stop)
        lay.addSpacing(8)

        # --- Preview ---
        lay.addWidget(QLabel("Aperçu:"))
        self._combo_preview = QComboBox()
        self._combo_preview.addItem("Masque (overlay)", "mask")
        self._combo_preview.addItem("Découpe (checker)", "cutout")
        self._combo_preview.addItem("Vidéo brute", "frame")
        lay.addWidget(self._combo_preview)
        lay.addSpacing(8)

        # --- Content mode ---
        lay.addWidget(QLabel("Mode:"))
        self._combo_mode = QComboBox()
        for cp in ContentProfile:
            self._combo_mode.addItem(self._content_label(cp), cp)
        lay.addWidget(self._combo_mode)
        lay.addSpacing(8)

        # --- Model ---
        lay.addWidget(QLabel("Modèle:"))
        self._combo_model = QComboBox()
        self._combo_model.addItem("Auto (recommandé)", None)
        for size in (
            ModelSize.TINY,
            ModelSize.SMALL,
            ModelSize.BASE_PLUS,
            ModelSize.LARGE,
        ):
            self._combo_model.addItem(size.value, size)
        lay.addWidget(self._combo_model)
        lay.addSpacing(8)

        # --- Segment navigation / re-propagate ---
        btn_prev_kf = QPushButton("◀ Key")
        btn_prev_kf.clicked.connect(lambda: self._goto_keyframe(-1))
        lay.addWidget(btn_prev_kf)
        btn_next_kf = QPushButton("Key ▶")
        btn_next_kf.clicked.connect(lambda: self._goto_keyframe(1))
        lay.addWidget(btn_next_kf)

        btn_prop_back = QPushButton("Propager ←")
        btn_prop_back.clicked.connect(lambda: self._run_propagation(-1))
        lay.addWidget(btn_prop_back)
        btn_prop_fwd = QPushButton("Propager →")
        btn_prop_fwd.clicked.connect(lambda: self._run_propagation(1))
        lay.addWidget(btn_prop_fwd)
        lay.addSpacing(8)

        # --- Tools: points / brush ---
        self._btn_points = QPushButton("Point")
        self._btn_points.setCheckable(True)
        self._btn_points.setChecked(True)
        self._btn_points.clicked.connect(lambda: self._set_tool("points"))
        lay.addWidget(self._btn_points)

        self._btn_brush = QPushButton("Pinceau")
        self._btn_brush.setCheckable(True)
        self._btn_brush.clicked.connect(lambda: self._set_tool("brush"))
        lay.addWidget(self._btn_brush)

        self._chk_erase = QCheckBox("Gomme")
        lay.addWidget(self._chk_erase)

        lay.addWidget(QLabel("Taille:"))
        self._spin_brush = QSpinBox()
        self._spin_brush.setRange(1, 300)
        self._spin_brush.setValue(20)
        lay.addWidget(self._spin_brush)
        lay.addStretch()

        self._central_layout.insertWidget(0, tb)

        # Connect signals *after* the widgets exist so no handler fires
        # while the toolbar is assembled.
        self._combo_preview.currentIndexChanged.connect(self._on_preview_changed)
        self._combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._combo_model.currentIndexChanged.connect(self._on_model_changed)
        self._chk_erase.toggled.connect(self._canvas.set_brush_erase)
        self._spin_brush.valueChanged.connect(self._on_brush_size_changed)

    @staticmethod
    def _content_label(cp: ContentProfile) -> str:
        labels = {
            ContentProfile.ANIME: "Anime / AMV",
            ContentProfile.HUMAN: "Humain",
            ContentProfile.OBJECT: "Objet",
        }
        return labels[cp]

    # ------------------------------------------------------------------ #
    # Toolbar handlers
    # ------------------------------------------------------------------ #
    def _on_preview_changed(self) -> None:
        self._canvas.set_preview_mode(str(self._combo_preview.currentData() or "mask"))

    def _on_mode_changed(self) -> None:
        if self._syncing:
            return
        cp = self._combo_mode.currentData()
        if cp is None or not isinstance(cp, ContentProfile):
            return
        self._set_content_mode(cp)

    def _on_model_changed(self) -> None:
        if self._syncing:
            return
        self._apply_model_selection()

    def _set_tool(self, tool: str) -> None:
        self._canvas.set_edit_mode(tool)
        self._btn_points.setChecked(tool == "points")
        self._btn_brush.setChecked(tool == "brush")

    def _on_brush_size_changed(self, value: int) -> None:
        self._canvas.set_brush_size(value)

    def _on_canvas_mask_edited(self) -> None:
        """Store a brush-painted mask for the current frame."""
        mask = self._canvas.get_mask()
        if mask is None:
            return
        self._masks[self._current_frame_idx] = mask
        self._keyframes.add(self._current_frame_idx)
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._status.showMessage(f"Mask édité au pinceau (frame {self._current_frame_idx})")

    def _goto_keyframe(self, direction: int) -> None:
        if not self._keyframes:
            self._status.showMessage("No keyframes yet")
            return
        kfs = sorted(self._keyframes)
        if direction > 0:
            target = next((k for k in kfs if k > self._current_frame_idx), kfs[0])
        else:
            target = next((k for k in reversed(kfs) if k < self._current_frame_idx), kfs[-1])
        self._set_frame(target)

    # ------------------------------------------------------------------ #
    # Hardware
    # ------------------------------------------------------------------ #
    def _auto_detect_hardware(self) -> None:
        self._status.showMessage("Detecting hardware...")
        QApplication.processEvents()
        try:
            self._recommend_now()
            self._sync_combos()
        except Exception as exc:
            logger.error("Hardware detection failed: %s", exc)
            self._status.showMessage(f"HW detection failed: {exc}")

    def _recommend_now(self, force_model: Optional[ModelSize] = None) -> None:
        """(Re)build the SAM 2 recommendation in real time.

        Re-reads the hardware each time so the freshest *free VRAM* is used —
        the model proposed is always the most adapted to the current load.
        """
        profile = HardwareDetector().detect()
        rec = SAM2Profiler().recommend(
            profile, content=self._config.content_profile, force_model=force_model
        )
        self._config.recommendation = rec
        self._lbl_hw.setText(rec.user_message.replace("**", ""))
        self._status.showMessage(f"Hardware OK — {rec.model_size.value} on {rec.device}")
        logger.info("Hardware recommendation updated: %s", rec)

    def _selected_force_model(self) -> Optional[ModelSize]:
        data = self._combo_model.currentData() if hasattr(self, "_combo_model") else None
        return data if isinstance(data, ModelSize) else None

    def _set_content_mode(self, cp: ContentProfile) -> None:
        """Switch content mode (anime / human / object) and re-recommend."""
        self._config.content_profile = cp
        self._recommend_now(force_model=self._selected_force_model())
        self._sync_combos()
        self._status.showMessage(
            f"Mode {self._content_label(cp)} — "
            f"modèle {self._config.recommendation.model_size.value}"
        )

    def _apply_model_selection(self) -> None:
        """User picked a specific model in the combo (or back to Auto)."""
        if not hasattr(self, "_combo_model") or self._combo_model is None:
            return
        self._recommend_now(force_model=self._selected_force_model())
        self._sync_combos()

    def _sync_combos(self) -> None:
        """Align the combo boxes with the active configuration (no loops)."""
        if not hasattr(self, "_combo_mode") or getattr(self, "_syncing", True):
            return
        rec = self._config.recommendation
        if rec is None:
            return
        self._syncing = True
        try:
            for i in range(self._combo_mode.count()):
                if self._combo_mode.itemData(i) == self._config.content_profile:
                    self._combo_mode.setCurrentIndex(i)
                    break
            # If a forced model is no longer the active recommendation, reset
            # the combo back to "Auto" so the user sees the real state.
            forced = self._selected_force_model()
            if forced is not None and forced != rec.model_size:
                self._combo_model.setCurrentIndex(0)
        finally:
            self._syncing = False

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
            self._timeline.set_masked_frames(set())
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
            self._pause_playback()
            return
        if self._video_reader is None:
            self._btn_play.setChecked(False)
            return
        self._play_start = self._current_frame_idx
        interval_ms = int(1000 / self._video_reader.fps) if self._video_reader.fps > 0 else 33
        self._play_timer.start(interval_ms)
        self._is_playing = True
        self._btn_play.setText("⏸ Pause")
        self._status.showMessage("Playing")

    def _pause_playback(self) -> None:
        self._play_timer.stop()
        self._is_playing = False
        self._btn_play.setText("▶ Play")
        self._btn_play.setChecked(False)
        self._status.showMessage("Paused")

    def _stop_playback(self) -> None:
        """Stop and return to the frame where playback started."""
        self._pause_playback()
        self._set_frame(self._play_start)
        self._status.showMessage("Stopped")

    def _on_play_tick(self) -> None:
        if self._video_reader is None:
            self._pause_playback()
            return
        if self._current_frame_idx + 1 >= self._video_reader.total_frames:
            self._pause_playback()
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
            self._timeline.set_masked_frames(set(self._masks))
            self._status.showMessage(f"Mask generated (score={best.score:.3f})")
        else:
            self._status.showMessage("No mask found")

    # ------------------------------------------------------------------ #
    # Propagation (video predictor)
    # ------------------------------------------------------------------ #
    def _run_propagation(self, direction: int = 1) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._config.recommendation is None:
            QMessageBox.warning(self, "No model", "Run hardware detection first.")
            return
        if not self._points:
            QMessageBox.warning(self, "No prompt", "Add correction points on this frame first.")
            return
        if self._propagation_worker is not None and self._propagation_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A propagation is already running.")
            return

        # Partial segment: from the current frame to the next (forward) or
        # previous (backward) keyframe — leaves other segments untouched.
        start, target = self._segment_bounds(direction)
        max_frames = abs(target - start) + 1 if start != target else None

        # NOTE: the heavy work (frame extraction, model load, tracking) runs
        # inside the worker thread — the GUI thread only wires signals here.
        self._propagation_worker = PropagationWorker(
            self._config.recommendation, self._ensure_sam_manager(), self
        )
        self._propagation_worker.set_task(
            str(self._video_reader.path),
            start_frame=start,
            points=list(self._points) or None,
            total_frames=self._video_reader.total_frames,
        )
        self._propagation_worker.set_range(max_frames=max_frames, reverse=direction < 0)
        self._propagation_worker.progress.connect(self._on_worker_progress)
        self._propagation_worker.frame_ready.connect(self._on_propagation_frame_ready)
        self._propagation_worker.finished.connect(self._on_propagation_finished)
        self._status.showMessage(
            f"Propagating {'←' if direction < 0 else '→'} from {start} to {target}... "
            f"({max_frames if max_frames else 'end'} frame(s))"
        )
        self.setEnabled(False)
        self._propagation_worker.start()

    def _segment_bounds(self, direction: int) -> tuple[int, int]:
        """Return (start, target) interval for a partial re-propagation.

        Forward:  current frame → frame just before the next keyframe (or end).
        Backward: current frame → frame just after the previous keyframe (or 0).
        """
        cur = self._current_frame_idx
        n = self._video_reader.total_frames if self._video_reader else cur + 1
        kfs = sorted(self._keyframes)
        if direction >= 0:
            nxt = next((k for k in kfs if k > cur), None)
            target = (nxt - 1) if nxt is not None else n - 1
            return cur, max(target, cur)
        prv = next((k for k in reversed(kfs) if k < cur), None)
        target = (prv + 1) if prv is not None else 0
        return cur, min(target, cur)

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
        self._timeline.set_masked_frames(set(self._masks))
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
        self._timeline.set_masked_frames(set(self._masks))
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
