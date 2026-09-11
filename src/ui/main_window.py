"""Main application window: orchestrates video, canvas, SAM 2, timeline, export."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.config import AppConfig
from src.core.project import PROJECT_EXT, ObjectRecord, ProjectData, ProjectManager
from src.core.scene import OBJECT_PALETTE, SceneObject, next_object_color, next_object_name
from src.core.undo import UndoManager
from src.ui.assets import app_icon, logo_path
from src.ui.components.tutorial_dialog import TutorialDialog
from src.ui.components.welcome_dialog import WelcomeDialog
from src.hardware import ContentProfile, HardwareDetector, ModelSize, SAM2Profiler
from src.models.sam2_manager import MaskResult, PointPrompt, SAM2Manager
from src.processing.frame_buffer import FrameBuffer
from src.processing.inference_worker import InferenceWorker, PropagationWorker
from src.processing.video_reader import VideoReader
from src.ui.timeline import Timeline
from src.ui.video_canvas import VideoCanvas
from src.ui.components import CollapsibleSection, Separator
from src.ui.components.export_dialog import ExportDialog
from src.ui.components.export_progress import ExportProgressDialog
from src.ui.components.loading_overlay import LoadingOverlay
from src.ui.components.library_panel import LibraryDialog
from src.export.exporter import Exporter

logger = logging.getLogger(__name__)


class ExportWorker(QThread):
    """Run PNG/ffmpeg export off the GUI thread."""

    finished = Signal(object)  # Path or Exception
    progress = Signal(str, int, int)  # stage, processed, total

    def __init__(
        self,
        exporter: Exporter,
        kind: str,
        masks: dict[int, np.ndarray],
        parent: Optional[QWidget] = None,
        frames: Optional[dict[int, np.ndarray]] = None,
        fps: float = 30.0,
        base_name: str = "output",
        resolution: Optional[tuple[int, int]] = None,
        range_mode: str = "all",
        start_frame: int = 0,
        end_frame: int = 0,
        audio_source: Optional[Path] = None,
        background: Optional[tuple[int, int, int]] = None,
    ) -> None:
        super().__init__(parent)
        self._exporter = exporter
        self._kind = kind  # "png" | "prores" | "qtrle" | "h264"
        self._masks = masks
        self._frames = frames or {}
        self._fps = fps
        self._base_name = base_name
        self._resolution = resolution
        self._range_mode = range_mode
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._audio_source = audio_source
        self._background = background
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel.set()

    def run(self) -> None:
        try:
            masks = dict(self._masks)
            frames = dict(self._frames)
            if self._range_mode == "range":
                masks = {
                    i: m for i, m in masks.items()
                    if self._start_frame <= i <= self._end_frame
                }
                frames = {
                    i: f for i, f in frames.items()
                    if self._start_frame <= i <= self._end_frame
                }
            if not masks:
                raise RuntimeError("No masks in the selected frame range")

            if self._kind == "png":
                self.progress.emit("Writing PNG masks", 0, len(masks))
                dest = self._exporter.export_png_sequence(
                    masks,
                    base_name=self._base_name,
                    on_progress=lambda c, t: self.progress.emit("Writing PNG masks", c, t),
                    cancel_event=self._cancel,
                )
            else:
                if not frames:
                    raise RuntimeError("Source frames are unavailable for this range")
                self.progress.emit("Compositing frames", 0, len(frames))
                dest = self._exporter.export_video(
                    frames,
                    masks,
                    self._fps,
                    base_name=self._base_name,
                    fmt=self._kind,
                    resolution=self._resolution,
                    on_progress=lambda c, t: self.progress.emit("Encoding frames", c, t),
                    cancel_event=self._cancel,
                    audio_source=self._audio_source,
                    background=self._background,
                )
            self.finished.emit(dest)
        except Exception as exc:  # noqa: BLE001
            logger.error("Export failed: %s", exc)
            self.finished.emit(exc)


class MainWindow(QMainWindow):
    """Better Brush main window."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Better Brush")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        # State
        self._config = AppConfig()
        self._video_reader: Optional[VideoReader] = None
        self._frame_buffer: Optional[FrameBuffer] = None
        self._current_frame_idx = 0
        self._is_playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._points: list[PointPrompt] = []
        self._masks: dict[int, np.ndarray] = {}
        self._keyframes: set[int] = set()
        self._sam_manager: Optional[SAM2Manager] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._propagation_worker: Optional[PropagationWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._export_progress_dlg = None
        self._export_started = 0.0
        self._timeline_selection: Optional[tuple[int, int]] = None
        self._play_start = 0
        self._syncing = False
        self._project_path: Optional[Path] = None
        self._project_dirty = False
        self._welcome_shown = False
        self._undo_manager = UndoManager()
        self._brush_stroke_active = False
        self._propagating = False

        # Multi-object scene state. ``self._points/_masks/_keyframes`` are
        # aliases into the active object's data.
        self._objects: list[SceneObject] = []
        self._active_object_id: Optional[int] = None

        # Automatic save
        project_settings = ProjectManager.load_user_settings()
        self._auto_save_enabled = bool(project_settings.get("auto_save_enabled", True))
        self._auto_save_interval_s = int(project_settings.get("auto_save_interval_s", 60))
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.timeout.connect(self._on_auto_save_tick)

        # Build UI
        self._setup_menu()
        self._setup_central_widget()
        self._setup_status_bar()

        # Fresh scene with one default object (aliases set up in-place)
        self._reset_scene()

        # Baseline undo state (empty session)
        self._undo_manager.reset(self._points, self._masks, self._keyframes)
        self._update_auto_save_timer()

        # Loading overlay covers the whole central area during long tasks
        self._loading_overlay = LoadingOverlay(self.centralWidget())

        # Branding: window/taskbar icon from the bundled app icon
        icon_path = app_icon()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        # Hardware recommendation on startup
        self._auto_detect_hardware()

    # ── Drag and drop ─────────────────────────────────────────────────────
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if any(path.endswith(e) for e in (".mp4", ".mov", ".avi", ".mkv", ".webm")):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if any(path.lower().endswith(e) for e in (".mp4", ".mov", ".avi", ".mkv", ".webm")):
                if self._confirm_discard():
                    self._load_video(Path(path))
                return

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────
    def _setup_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New Project...", self)
        new_act.setShortcut(QKeySequence("Ctrl+Shift+N"))
        new_act.triggered.connect(self._on_new_project)
        file_menu.addAction(new_act)

        open_project_act = QAction("&Open Project...", self)
        open_project_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_project_act.triggered.connect(self._on_open_project)
        file_menu.addAction(open_project_act)

        save_act = QAction("&Save Project", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._on_save_project)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save Project &As...", self)
        save_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_act.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        self._act_auto_save = QAction("&Automatic Save", self)
        self._act_auto_save.setCheckable(True)
        self._act_auto_save.setChecked(self._auto_save_enabled)
        self._act_auto_save.toggled.connect(self._on_auto_save_toggled)
        file_menu.addAction(self._act_auto_save)

        file_menu.addSeparator()

        open_act = QAction("&Open Video...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_video)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        lib_act = QAction("&Media Library...", self)
        lib_act.setShortcut(QKeySequence("Ctrl+L"))
        lib_act.triggered.connect(self._show_library)
        file_menu.addAction(lib_act)

        export_menu = menubar.addMenu("&Export")
        export_act = QAction("&Export...", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self._show_export_dialog)
        export_menu.addAction(export_act)

        edit_menu = menubar.addMenu("&Edit")

        undo_act = QAction("&Undo", self)
        undo_act.setShortcut(QKeySequence("Ctrl+Z"))
        undo_act.triggered.connect(self._undo)
        edit_menu.addAction(undo_act)

        redo_act = QAction("&Redo", self)
        redo_act.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        redo_act.triggered.connect(self._redo)
        edit_menu.addAction(redo_act)

        help_menu = menubar.addMenu("&Help")
        tutorial_act = QAction("&Quick Tutorial", self)
        tutorial_act.triggered.connect(lambda: self._show_tutorial(parent=self))
        help_menu.addAction(tutorial_act)
        about_act = QAction("&About Better Brush", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

    def _setup_central_widget(self) -> None:
        central = QWidget()
        self._central_layout = QVBoxLayout(central)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self._central_layout.setSpacing(0)

        # ── Branding bar ──────────────────────────────────────────────────
        branding = QWidget()
        branding.setFixedHeight(34)
        branding.setStyleSheet("background: #1a1a1e; border-bottom: 1px solid #2e2e33;")
        bl = QHBoxLayout(branding)
        bl.setContentsMargins(14, 0, 14, 0)

        # Small logo mark next to the app name (strategic, not everywhere)
        logo_path_ = logo_path()
        if logo_path_.is_file():
            logo_lbl = QLabel()
            logo_lbl.setPixmap(
                QPixmap(str(logo_path_)).scaled(
                    18, 18,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            bl.addWidget(logo_lbl)

        lbl_app = QLabel("Better Brush")
        lbl_app.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #d4d4d8; "
            "background: transparent; border: none;"
        )
        bl.addWidget(lbl_app)
        bl.addStretch()

        self._lbl_gpu = QLabel("")
        self._lbl_gpu.setStyleSheet(
            "font-size: 10px; color: #555560; background: transparent; border: none;"
        )
        bl.addWidget(self._lbl_gpu)

        self._central_layout.addWidget(branding)

        # ── Main splitter: canvas + right panel ───────────────────────────
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(1)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background: #2e2e33; }"
            "QSplitter::handle:horizontal { width: 1px; }"
        )

        # Canvas (center, dominant)
        self._canvas = VideoCanvas()
        self._canvas.point_added.connect(self._on_canvas_point)
        self._canvas.point_removed.connect(self._on_canvas_point_removed)
        self._canvas.point_selected.connect(self._on_canvas_point_selected)
        self._canvas.frame_navigate.connect(self._on_navigate)
        self._canvas.request_segment.connect(self._run_segmentation)
        self._canvas.play_pause.connect(self._toggle_play)
        self._canvas.brush_press.connect(self._on_canvas_brush_press)
        self._canvas.mask_edited.connect(self._on_canvas_mask_edited)
        main_splitter.addWidget(self._canvas)

        # Right panel
        self._setup_right_panel()
        main_splitter.addWidget(self._right_scroll)

        main_splitter.setSizes([1100, 280])
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)

        self._central_layout.addWidget(main_splitter, stretch=1)

        # ── Bottom bar: transport + timeline ──────────────────────────────
        self._setup_bottom_bar()

        # Connect combo/signal handlers *after* all widgets are created
        self._combo_preview.currentIndexChanged.connect(self._on_preview_changed)
        self._combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._combo_model.currentIndexChanged.connect(self._on_model_changed)
        self._combo_object.currentIndexChanged.connect(self._on_object_changed)
        self._chk_erase.toggled.connect(self._canvas.set_brush_erase)
        self._spin_brush.valueChanged.connect(self._on_brush_size_changed)

        # Populate the objects dropdown once the widget tree exists
        self._refresh_objects_ui()

        self.setCentralWidget(central)

    # ── Right panel ───────────────────────────────────────────────────────
    def _setup_right_panel(self) -> None:
        self._right_scroll = QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._right_scroll.setFixedWidth(280)
        self._right_scroll.setStyleSheet(
            "QScrollArea { background: #202024; border: none; border-left: 1px solid #2e2e33; }"
            "QScrollBar { background: transparent; }"
        )

        panel = QWidget()
        panel.setStyleSheet("background: #202024;")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)

        # ── Content ───────────────────────────────────────────────────────
        sec_content = CollapsibleSection("Content")
        self._combo_mode = QComboBox()
        for cp in ContentProfile:
            self._combo_mode.addItem(self._content_label(cp), cp)
        sec_content.add_widget(self._combo_mode)
        pl.addWidget(sec_content)

        # ── AI Model ──────────────────────────────────────────────────────
        sec_model = CollapsibleSection("AI Model", expanded=False)
        self._combo_model = QComboBox()
        self._combo_model.addItem("Auto (recommended)", None)
        for size in (
            ModelSize.TINY,
            ModelSize.SMALL,
            ModelSize.BASE_PLUS,
            ModelSize.LARGE,
        ):
            self._combo_model.addItem(size.value, size)
        sec_model.add_widget(self._combo_model)
        pl.addWidget(sec_model)

        # ── Objects ───────────────────────────────────────────────────────
        sec_objects = CollapsibleSection("Objects")
        obj_row = QHBoxLayout()
        obj_row.setSpacing(4)
        self._combo_object = QComboBox()
        self._combo_object.setToolTip("Active object - choose which character to edit")
        obj_row.addWidget(self._combo_object, stretch=1)
        self._btn_add_object = QPushButton("+")
        self._btn_add_object.setFixedWidth(28)
        self._btn_add_object.setToolTip("Add object")
        self._btn_add_object.clicked.connect(self._add_object)
        obj_row.addWidget(self._btn_add_object)
        self._btn_delete_object = QPushButton("\u2212")
        self._btn_delete_object.setFixedWidth(28)
        self._btn_delete_object.setToolTip("Delete active object")
        self._btn_delete_object.clicked.connect(self._delete_active_object)
        obj_row.addWidget(self._btn_delete_object)
        sec_objects.add_layout(obj_row)
        pl.addWidget(sec_objects)

        pl.addWidget(Separator())

        # ── Tools ─────────────────────────────────────────────────────────
        sec_tools = CollapsibleSection("Tools")

        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        self._btn_points = QPushButton("Point")
        self._btn_points.setCheckable(True)
        self._btn_points.setChecked(True)
        self._btn_points.clicked.connect(lambda: self._set_tool("points"))
        tool_row.addWidget(self._btn_points)

        self._btn_brush = QPushButton("Brush")
        self._btn_brush.setCheckable(True)
        self._btn_brush.clicked.connect(lambda: self._set_tool("brush"))
        tool_row.addWidget(self._btn_brush)
        sec_tools.add_layout(tool_row)

        self._chk_erase = QCheckBox("Eraser")
        sec_tools.add_widget(self._chk_erase)

        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        lbl_size = QLabel("Size:")
        lbl_size.setStyleSheet("color: #8b8b94; font-size: 11px;")
        size_row.addWidget(lbl_size)
        self._spin_brush = QSpinBox()
        self._spin_brush.setRange(1, 300)
        self._spin_brush.setValue(20)
        size_row.addWidget(self._spin_brush, stretch=1)
        sec_tools.add_layout(size_row)

        # Tools hint: brush add/erase via Shift
        hint_nav = QLabel("Brush: paint to add · hold Shift while painting to erase")
        hint_nav.setStyleSheet("color: #7a7a84; font-size: 10px; background: transparent;")
        hint_nav.setWordWrap(True)
        sec_tools.add_widget(hint_nav)

        pl.addWidget(sec_tools)

        pl.addWidget(Separator())

        # ── Segment ───────────────────────────────────────────────────────
        sec_segment = CollapsibleSection("Segment")

        self._btn_segment = QPushButton("Segment Frame")
        self._btn_segment.clicked.connect(self._run_segmentation)
        sec_segment.add_widget(self._btn_segment)

        self._btn_clear_points = QPushButton("Clear Points")
        self._btn_clear_points.clicked.connect(self._clear_points)
        sec_segment.add_widget(self._btn_clear_points)

        self._btn_delete_point = QPushButton("Delete Selected Point")
        self._btn_delete_point.clicked.connect(self._delete_selected_point)
        sec_segment.add_widget(self._btn_delete_point)

        hint = QLabel("Left-click = include (green) · Right-click = exclude (red)")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7a7a84; font-size: 10px; background: transparent;")
        sec_segment.add_widget(hint)

        self._btn_clear_mask = QPushButton("Clear Mask")
        self._btn_clear_mask.clicked.connect(self._clear_mask)
        sec_segment.add_widget(self._btn_clear_mask)

        pl.addWidget(sec_segment)

        pl.addWidget(Separator())

        # ── Propagation ───────────────────────────────────────────────────
        sec_prop = CollapsibleSection("Propagation")

        kf_row = QHBoxLayout()
        kf_row.setSpacing(4)
        btn_prev_kf = QPushButton("\u25C0 Key")
        btn_prev_kf.clicked.connect(lambda: self._goto_keyframe(-1))
        kf_row.addWidget(btn_prev_kf)
        btn_next_kf = QPushButton("Key \u25B6")
        btn_next_kf.clicked.connect(lambda: self._goto_keyframe(1))
        kf_row.addWidget(btn_next_kf)
        sec_prop.add_layout(kf_row)

        prop_row = QHBoxLayout()
        prop_row.setSpacing(4)
        btn_prop_back = QPushButton("\u25C0 Prop")
        btn_prop_back.clicked.connect(lambda: self._run_propagation(-1))
        prop_row.addWidget(btn_prop_back)
        btn_prop_fwd = QPushButton("Prop \u25B6")
        btn_prop_fwd.clicked.connect(lambda: self._run_propagation(1))
        prop_row.addWidget(btn_prop_fwd)
        sec_prop.add_layout(prop_row)

        self._btn_propagate = QPushButton("Propagate to End")
        self._btn_propagate.clicked.connect(lambda: self._run_propagation(1))
        sec_prop.add_widget(self._btn_propagate)

        pl.addWidget(sec_prop)

        pl.addWidget(Separator())

        # ── Export ────────────────────────────────────────────────────────
        sec_export = CollapsibleSection("Export")

        self._btn_export = QPushButton("Export...")
        self._btn_export.setProperty("cssClass", "accent")
        self._btn_export.clicked.connect(self._show_export_dialog)
        sec_export.add_widget(self._btn_export)

        lbl_export_hint = QLabel("PNG alpha, ProRes 4444, H.264")
        lbl_export_hint.setStyleSheet("color: #555560; font-size: 10px; background: transparent;")
        lbl_export_hint.setWordWrap(True)
        sec_export.add_widget(lbl_export_hint)

        pl.addWidget(sec_export)

        pl.addWidget(Separator())

        # ── Hardware (collapsed) ──────────────────────────────────────────
        sec_hw = CollapsibleSection("Hardware", expanded=False)

        self._btn_detect = QPushButton("Detect Hardware")
        self._btn_detect.clicked.connect(self._auto_detect_hardware)
        sec_hw.add_widget(self._btn_detect)

        self._lbl_hw = QLabel("Hardware: detecting...")
        self._lbl_hw.setWordWrap(True)
        self._lbl_hw.setStyleSheet("color: #8b8b94; font-size: 10px; background: transparent;")
        sec_hw.add_widget(self._lbl_hw)

        pl.addWidget(sec_hw)

        pl.addStretch()

        self._right_scroll.setWidget(panel)

    # ── Bottom bar ────────────────────────────────────────────────────────
    def _setup_bottom_bar(self) -> None:
        bottom = QWidget()
        bottom.setStyleSheet("background: #171719; border-top: 1px solid #2e2e33;")
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)

        # Transport row
        transport = QWidget()
        transport.setFixedHeight(36)
        transport.setStyleSheet("background: #171719;")
        tl = QHBoxLayout(transport)
        tl.setContentsMargins(10, 0, 10, 0)
        tl.setSpacing(6)

        self._btn_play = QPushButton("Play")
        self._btn_play.setCheckable(True)
        self._btn_play.setFixedWidth(64)
        self._btn_play.clicked.connect(self._toggle_play)
        tl.addWidget(self._btn_play)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setFixedWidth(52)
        self._btn_stop.clicked.connect(self._stop_playback)
        tl.addWidget(self._btn_stop)

        tl.addSpacing(16)

        self._lbl_timecode = QLabel("00:00:00:00")
        self._lbl_timecode.setStyleSheet(
            "font-family: 'Consolas', 'SF Mono', monospace; font-size: 13px; "
            "color: #d4d4d8; background: #1a1a1e; padding: 3px 10px; "
            "border: 1px solid #2e2e33; border-radius: 3px;"
        )
        tl.addWidget(self._lbl_timecode)

        self._lbl_frame_counter = QLabel("0 / 0")
        self._lbl_frame_counter.setStyleSheet(
            "font-family: 'Consolas', 'SF Mono', monospace; font-size: 11px; "
            "color: #8b8b94; background: transparent; border: none;"
        )
        tl.addWidget(self._lbl_frame_counter)

        tl.addStretch()

        lbl_view = QLabel("View:")
        lbl_view.setStyleSheet("color: #8b8b94; font-size: 11px; background: transparent; border: none;")
        tl.addWidget(lbl_view)

        self._combo_preview = QComboBox()
        self._combo_preview.addItem("Overlay", "mask")
        self._combo_preview.addItem("Cutout", "cutout")
        self._combo_preview.addItem("Raw", "frame")
        tl.addWidget(self._combo_preview)

        bl.addWidget(transport)

        # Timeline
        self._timeline = Timeline()
        self._timeline.frame_selected.connect(self._set_frame)
        self._timeline.selection_changed.connect(self._on_timeline_selection_changed)
        bl.addWidget(self._timeline)

        self._central_layout.addWidget(bottom)

    def _setup_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ──────────────────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _content_label(cp: ContentProfile) -> str:
        labels = {
            ContentProfile.ANIME: "Anime / AMV",
            ContentProfile.HUMAN: "Human",
            ContentProfile.OBJECT: "Object",
        }
        return labels.get(cp, str(cp.value))

    def _format_timecode(self, frame_idx: int) -> str:
        fps = self._video_reader.fps if self._video_reader else 30.0
        fps = max(1.0, fps)
        total_s = frame_idx / fps
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        s = int(total_s % 60)
        f = int(frame_idx % int(round(fps)))
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    def _update_transport(self) -> None:
        if hasattr(self, "_lbl_timecode"):
            self._lbl_timecode.setText(self._format_timecode(self._current_frame_idx))
        if hasattr(self, "_lbl_frame_counter"):
            total = self._video_reader.total_frames if self._video_reader else 0
            self._lbl_frame_counter.setText(
                f"{self._current_frame_idx} / {max(0, total - 1)}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Toolbar handlers
    # ──────────────────────────────────────────────────────────────────────
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
        if tool == "brush":
            self._status.showMessage("Brush mode - paint directly on the frame")
        else:
            self._status.showMessage(
                "Points mode - Left-click = include (green) · Right-click = exclude (red)"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Scene / multi-object management
    # ──────────────────────────────────────────────────────────────────────
    def _active_object(self) -> Optional[SceneObject]:
        """The currently active object, or None if the scene is empty."""
        for obj in self._objects:
            if obj.obj_id == self._active_object_id:
                return obj
        return self._objects[0] if self._objects else None

    def _reset_scene(self) -> None:
        """Discard every object and create the default single object."""
        self._objects = []
        first = SceneObject(obj_id=1, name="Object 1", color=OBJECT_PALETTE[0])
        self._objects.append(first)
        self._active_object_id = first.obj_id
        self._points = first.points
        self._masks = first.masks
        self._keyframes = first.keyframes
        if hasattr(self, "_canvas"):
            self._canvas.set_mask_color(QColor(first.color))
        self._refresh_objects_ui()

    def _new_object(self) -> SceneObject:
        """Create and append a fresh object with the next free name/color."""
        used_ids = {obj.obj_id for obj in self._objects}
        obj_id = next(i for i in range(1, len(self._objects) + 2) if i not in used_ids)
        obj = SceneObject(
            obj_id=obj_id,
            name=next_object_name(self._objects),
            color=next_object_color(self._objects),
        )
        self._objects.append(obj)
        return obj

    def _add_object(self) -> None:
        obj = self._new_object()
        self._set_active_object(obj.obj_id)
        self._status.showMessage(f"Added object - {obj.name}")

    def _delete_active_object(self) -> None:
        if len(self._objects) <= 1:
            self._status.showMessage("Keep at least one object - all would be deleted")
            return
        obj = self._active_object()
        if obj is None:
            return
        self._objects = [o for o in self._objects if o.obj_id != obj.obj_id]
        remaining = self._objects[0]
        self._active_object_id = remaining.obj_id
        self._points = remaining.points
        self._masks = remaining.masks
        self._keyframes = remaining.keyframes
        self._canvas.set_mask_color(QColor(remaining.color))
        self._canvas.set_points(self._points)
        self._show_current_frame()
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._undo_manager.reset(self._points, self._masks, self._keyframes)
        self._refresh_objects_ui()
        self._mark_dirty()
        self._status.showMessage(f"Deleted {obj.name} - editing {remaining.name}")

    def _set_active_object(self, obj_id: int) -> None:
        """Make the object with ``obj_id`` the active one and alias its data."""
        obj = next((o for o in self._objects if o.obj_id == obj_id), None)
        if obj is None:
            return
        if obj_id == self._active_object_id and self._masks is obj.masks:
            self._refresh_objects_ui()
            return
        self._active_object_id = obj_id
        self._points = obj.points
        self._masks = obj.masks
        self._keyframes = obj.keyframes
        self._canvas.set_mask_color(QColor(obj.color))
        self._canvas.set_points(self._points)
        self._show_current_frame()
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._undo_manager.reset(self._points, self._masks, self._keyframes)
        self._refresh_objects_ui()
        self._status.showMessage(f"Editing {obj.name}")

    def _on_object_changed(self) -> None:
        idx = self._combo_object.currentIndex()
        if idx < 0:
            return
        self._set_active_object(int(self._combo_object.itemData(idx)))

    def _refresh_objects_ui(self) -> None:
        """Rebuild the objects combo to match ``self._objects``."""
        combo = getattr(self, "_combo_object", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        for obj in self._objects:
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(obj.color))
            combo.addItem(QIcon(swatch), obj.name, obj.obj_id)
        idx = next(
            (
                i
                for i in range(combo.count())
                if combo.itemData(i) == self._active_object_id
            ),
            0,
        )
        combo.setCurrentIndex(max(0, min(idx, combo.count() - 1)))
        combo.blockSignals(False)
        btn = getattr(self, "_btn_delete_object", None)
        if btn is not None:
            btn.setEnabled(len(self._objects) > 1)

    def _combined_masks(self) -> dict[int, np.ndarray]:
        """Union of every object's masks, keyed by frame index."""
        combined: dict[int, np.ndarray] = {}
        for obj in self._objects:
            for fi, m in obj.masks.items():
                if fi in combined:
                    combined[fi] = combined[fi] | m
                else:
                    combined[fi] = m
        return combined

    def _on_brush_size_changed(self, value: int) -> None:
        self._canvas.set_brush_size(value)

    def _on_canvas_brush_press(self) -> None:
        """A brush stroke is starting - flag that a push should happen on release."""
        self._brush_stroke_active = True

    def _on_canvas_mask_edited(self) -> None:
        mask = self._canvas.get_mask()
        if mask is None:
            return
        self._masks[self._current_frame_idx] = mask
        self._keyframes.add(self._current_frame_idx)
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._mark_dirty()
        if self._brush_stroke_active and not self._propagating:
            self._undo_manager.push(self._points, self._masks, self._keyframes)
            self._brush_stroke_active = False
        self._status.showMessage(f"Mask edited (frame {self._current_frame_idx})")

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

    # ──────────────────────────────────────────────────────────────────────
    # Undo / Redo
    # ──────────────────────────────────────────────────────────────────────
    def _undo(self) -> None:
        snap = self._undo_manager.undo(self._points, self._masks, self._keyframes)
        if snap is not None:
            self._restore_state(snap)
            self._status.showMessage("Undo")

    def _redo(self) -> None:
        snap = self._undo_manager.redo(self._points, self._masks, self._keyframes)
        if snap is not None:
            self._restore_state(snap)
            self._status.showMessage("Redo")

    def _restore_state(self, snap) -> None:
        self._canvas.set_points(self._points)
        if self._current_frame_idx in self._masks:
            self._canvas.set_mask(self._masks[self._current_frame_idx])
        else:
            self._canvas.clear_mask()
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._mark_dirty()

    # ──────────────────────────────────────────────────────────────────────
    # Hardware
    # ──────────────────────────────────────────────────────────────────────
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
        profile = HardwareDetector().detect()
        rec = SAM2Profiler().recommend(
            profile, content=self._config.content_profile, force_model=force_model
        )
        self._config.recommendation = rec
        self._lbl_hw.setText(rec.user_message.replace("**", ""))
        # Update GPU indicator in branding bar
        if hasattr(self, "_lbl_gpu"):
            target_gpu = next(
                (g for g in profile.gpus if g.backend.value == rec.backend.value), None
            )
            if target_gpu:
                name = target_gpu.name
                if len(name) > 28:
                    name = name[:26] + ".."
                self._lbl_gpu.setText(name)
            else:
                self._lbl_gpu.setText(rec.device.upper())
        self._status.showMessage(f"Hardware OK - {rec.model_size.value} on {rec.device}")
        logger.info("Hardware recommendation updated: %s", rec)

    def _selected_force_model(self) -> Optional[ModelSize]:
        data = self._combo_model.currentData() if hasattr(self, "_combo_model") else None
        return data if isinstance(data, ModelSize) else None

    def _set_content_mode(self, cp: ContentProfile) -> None:
        self._config.content_profile = cp
        self._recommend_now(force_model=self._selected_force_model())
        self._sync_combos()
        self._status.showMessage(
            f"Mode {self._content_label(cp)} - "
            f"model {self._config.recommendation.model_size.value}"
        )

    def _apply_model_selection(self) -> None:
        if not hasattr(self, "_combo_model") or self._combo_model is None:
            return
        self._recommend_now(force_model=self._selected_force_model())
        self._sync_combos()

    def _sync_combos(self) -> None:
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
            forced = self._selected_force_model()
            if forced is not None and forced != rec.model_size:
                self._combo_model.setCurrentIndex(0)
        finally:
            self._syncing = False

    # ──────────────────────────────────────────────────────────────────────
    # Welcome / tutorial
    # ──────────────────────────────────────────────────────────────────────
    def _show_welcome(self) -> None:
        if self._welcome_shown:
            return
        self._welcome_shown = True
        dlg = WelcomeDialog(self)
        dlg.exec()
        self._on_welcome_result(dlg.choice)
        # Quick tutorial on the very first launch
        if ProjectManager.is_first_run():
            ProjectManager.set_first_run_done()
            QTimer.singleShot(0, lambda: self._show_tutorial(parent=self))

    def show_welcome(self) -> None:
        """Public hook called by the app bootstrap to open the launcher."""
        self._show_welcome()

    def _on_welcome_result(self, choice: tuple[str, Optional[Path]]) -> None:
        kind, path = choice
        if kind == "import" and path is not None:
            self._load_video(path)
        elif kind == "open" and path is not None:
            self._open_project(path)

    def _show_tutorial(self, parent: Optional[QWidget] = None) -> None:
        TutorialDialog(parent or self).exec()

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Better Brush",
            "Better Brush v0.1.0 (beta)\n\n"
            "AI-powered rotoscoping with SAM 2.\n"
            "Import a video, prompt SAM 2 with points, propagate the mask "
            "across the timeline, refine with the brush, then export.\n\n"
            "Solo developer: Sneg — https://sneg.dev\n"
            "Sneg HQ, a company of Lugunica.",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Project files
    # ──────────────────────────────────────────────────────────────────────
    def _mark_dirty(self) -> None:
        if self._video_reader is not None:
            self._project_dirty = True

    def _confirm_discard(self) -> bool:
        """Ask before throwing away unsaved project work (Yes to discard)."""
        if not self._project_dirty or self._video_reader is None:
            return True
        ret = QMessageBox.warning(
            self,
            "Unsaved changes",
            "Discard unsaved changes to this project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ret == QMessageBox.StandardButton.Yes

    def _reset_to_blank(self) -> None:
        """Close any loaded video / project and return to a blank state."""
        self._pause_playback()
        if self._propagation_worker is not None and self._propagation_worker.isRunning():
            self._propagation_worker.cancel()
            self._propagation_worker.wait(3000)
        for worker in (self._inference_worker, self._export_worker):
            if worker is not None and worker.isRunning():
                worker.wait(3000)
        if self._frame_buffer is not None:
            self._frame_buffer.close()
        self._frame_buffer = None
        self._video_reader = None
        self._reset_scene()
        self._timeline_selection = None
        self._project_path = None
        self._project_dirty = False
        self._brush_stroke_active = False
        self._propagating = False
        self._undo_manager.reset(self._points, self._masks, self._keyframes)
        self._update_auto_save_timer()
        self._canvas.clear_points()
        self._canvas.clear_mask()
        self._canvas.set_frame(None)
        self._timeline.set_duration(1)
        self._timeline.set_current_frame(0)
        self._timeline.set_keyframes(set())
        self._timeline.set_masked_frames(set())
        self._timeline.clear_selection()
        self.setWindowTitle("Better Brush")
        self._update_transport()
        self._status.showMessage("New project - import a video or drag and drop one here")

    def _on_new_project(self) -> None:
        if not self._confirm_discard():
            return
        self._reset_to_blank()

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Better Brush Project (*.bbproj);;All Files (*)"
        )
        if not path:
            return
        self._open_project(Path(path))

    def _open_project(self, path: Path) -> None:
        try:
            data = ProjectManager.load(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read project: %s", exc)
            QMessageBox.critical(self, "Project Error", f"Cannot open project:\n{exc}")
            return
        if not self._confirm_discard():
            return

        video_path = self._resolve_video_path(data)
        if video_path is None:
            return

        self._reset_to_blank()
        try:
            self._load_video(video_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load project video: %s", exc)
            QMessageBox.critical(self, "Project Error", f"Cannot load video:\n{exc}")
            return

        # Rebuild the object scene (v2 format) or wrap a legacy single-object
        # project into one object (v1 format).
        objects: list[SceneObject] = []
        if data.objects:
            for rec in data.objects:
                objects.append(
                    SceneObject(
                        obj_id=rec.obj_id,
                        name=rec.name,
                        color=rec.color,
                        masks=dict(rec.masks),
                        points=[PointPrompt(**p) for p in rec.points],
                        keyframes=set(rec.keyframes),
                    )
                )
        else:
            objects.append(
                SceneObject(
                    obj_id=1,
                    name="Object 1",
                    color=OBJECT_PALETTE[0],
                    masks=dict(data.masks),
                    points=[PointPrompt(**p) for p in data.points],
                    keyframes=set(data.keyframes),
                )
            )
        self._objects = objects
        preferred = data.active_object
        if preferred is None or not any(o.obj_id == preferred for o in objects):
            preferred = objects[0].obj_id
        self._set_active_object(preferred)

        self._canvas.set_edit_mode("brush" if data.edit_mode == "brush" else "points")
        self._set_tool("brush" if data.edit_mode == "brush" else "points")
        self._spin_brush.setValue(data.brush_size)
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        if data.timeline_selection and len(data.timeline_selection) == 2:
            self._timeline.set_selection(
                int(data.timeline_selection[0]), int(data.timeline_selection[1])
            )
        else:
            self._timeline.clear_selection()
        self._set_frame(
            min(
                int(data.current_frame or 0),
                max(0, int(self._video_reader.total_frames) - 1),
            )
        )

        self._project_path = path
        self._project_dirty = False
        self._undo_manager.reset(self._points, self._masks, self._keyframes)
        name = data.video_name or path.stem
        self.setWindowTitle(f"Better Brush - {name} (project)")
        self._status.showMessage(
            f"Project loaded - {len(self._masks)} frame(s) masked, "
            f"{len(self._points)} point(s)"
        )

    def _resolve_video_path(self, data: ProjectData) -> Optional[Path]:
        """Return the video path, asking the user if the original is missing."""
        if data.video_path:
            p = Path(data.video_path)
            if p.is_file():
                return p
        # The original video moved or was renamed - ask for its new location.
        start = str(Path(data.video_path).parent) if data.video_path else ""
        hint = f"The video for this project was not found:\n{data.video_path}\n\nLocate it?"
        ret = QMessageBox.question(
            self,
            "Locate video",
            hint,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return None
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate Video", start, "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        return Path(path) if path else None

    def _on_save_project(self) -> None:
        if self._project_path is not None:
            self._save_project(self._project_path)
        else:
            self._on_save_project_as()

    def _on_save_project_as(self) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "Nothing to save", "Import a video first.")
            return
        start = str(self._project_path) if self._project_path else ""
        if not start:
            default = self._video_reader.path.with_suffix(".bbproj")
            start = str(default)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            start,
            "Better Brush Project (*.bbproj);;All Files (*)",
        )
        if not path:
            return
        self._save_project(Path(path))

    def _build_project_data(self) -> ProjectData:
        """Serialize the current session (including all objects)."""
        return ProjectData(
            video_path=str(self._video_reader.path),
            video_meta={
                "frame_rate": self._video_reader.fps,
                "width": self._video_reader.width,
                "height": self._video_reader.height,
                "total_frames": self._video_reader.total_frames,
            },
            points=[
                {"x": p.x, "y": p.y, "label": p.label} for p in self._points
            ],
            keyframes=sorted(self._keyframes),
            masks=dict(self._masks),
            current_frame=self._current_frame_idx,
            edit_mode=self._canvas.get_edit_mode(),
            brush_size=self._spin_brush.value(),
            timeline_selection=(
                list(self._timeline_selection) if self._timeline_selection else None
            ),
            objects=[
                ObjectRecord(
                    obj_id=obj.obj_id,
                    name=obj.name,
                    color=obj.color,
                    points=[{"x": p.x, "y": p.y, "label": p.label} for p in obj.points],
                    keyframes=sorted(obj.keyframes),
                    masks=dict(obj.masks),
                )
                for obj in self._objects
            ],
            active_object=self._active_object_id,
        )

    def _save_project(self, path: Path) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "Nothing to save", "Import a video first.")
            return
        data = self._build_project_data()
        try:
            project_path = ProjectManager.save(path, data)
            ProjectManager.add_recent(project_path, self._video_reader.path)
            self._project_path = project_path
            self._project_dirty = False
            self.setWindowTitle(
                f"Better Brush - {data.video_name} (project)"
            )
            self._status.showMessage(
                f"Project saved - {project_path.name} "
                f"({len(data.masks)} frame(s), {len(data.points)} point(s))"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save project: %s", exc)
            QMessageBox.critical(self, "Project Error", f"Cannot save project:\n{exc}")

    # ── Automatic save ────────────────────────────────────────────────────
    def _on_auto_save_toggled(self, enabled: bool) -> None:
        self._auto_save_enabled = bool(enabled)
        ProjectManager.save_user_settings({"auto_save_enabled": self._auto_save_enabled})
        self._update_auto_save_timer()
        if enabled:
            if self._video_reader is not None:
                self._perform_autosave()
            self._status.showMessage("Automatic save enabled")
        else:
            self._status.showMessage("Automatic save disabled")

    def _update_auto_save_timer(self) -> None:
        timer = getattr(self, "_auto_save_timer", None)
        if timer is None:
            return
        if self._auto_save_enabled and self._video_reader is not None:
            timer.start(max(5, int(self._auto_save_interval_s * 1000)))
        else:
            timer.stop()

    def _on_auto_save_tick(self) -> None:
        self._perform_autosave()

    def _perform_autosave(self) -> None:
        """Save a background copy of the current session without touching
        the main project file or clearing the dirty flag."""
        if self._video_reader is None or not self._project_dirty:
            return
        try:
            data = self._build_project_data()
            dest = ProjectManager.autosave_dir() / (
                f"{self._video_reader.path.stem}.autosave{PROJECT_EXT}"
            )
            ProjectManager.save(dest, data)
            self._status.showMessage(f"Autosaved - {dest.name}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Autosave failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Video loading
    # ──────────────────────────────────────────────────────────────────────
    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        if not path:
            return
        if not self._confirm_discard():
            return
        self._load_video(Path(path))

    def _load_video(self, path: Path) -> None:
        try:
            if self._video_reader is not None:
                self._frame_buffer.close()
            self._reset_scene()
            self._video_reader = VideoReader(path)
            self._frame_buffer = FrameBuffer(self._video_reader)
            self._current_frame_idx = 0
            self._canvas.clear_points()
            self._canvas.clear_mask()
            self._timeline.set_duration(self._video_reader.total_frames)
            self._timeline.set_current_frame(0)
            self._timeline.set_masked_frames(set())
            self._timeline.set_keyframes(set())
            self._timeline.clear_selection()
            self._show_current_frame()
            self._project_path = None
            self._project_dirty = False
            self._brush_stroke_active = False
            self._propagating = False
            self._undo_manager.reset(self._points, self._masks, self._keyframes)
            self._update_auto_save_timer()
            self.setWindowTitle(f"Better Brush - {path.name}")
            self._status.showMessage(
                f"Loaded {path.name} - {self._video_reader.width}x{self._video_reader.height} "
                f"@ {self._video_reader.fps:.2f} fps"
            )
        except Exception as exc:
            logger.error("Failed to load video: %s", exc)
            QMessageBox.critical(self, "Error", f"Cannot load video:\n{exc}")

    # ──────────────────────────────────────────────────────────────────────
    # Frame navigation
    # ──────────────────────────────────────────────────────────────────────
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
            self._update_transport()

    def _set_frame(self, idx: int) -> None:
        self._current_frame_idx = max(0, idx)
        self._show_current_frame()

    def _on_timeline_selection_changed(self, start: int, end: int) -> None:
        if start < 0 or end < 0 or start > end:
            self._timeline_selection = None
            return
        self._timeline_selection = (start, end)
        self._mark_dirty()

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
        self._btn_play.setText("Pause")
        self._status.showMessage("Playing")

    def _pause_playback(self) -> None:
        self._play_timer.stop()
        self._is_playing = False
        self._btn_play.setText("Play")
        self._btn_play.setChecked(False)
        self._status.showMessage("Paused")

    def _stop_playback(self) -> None:
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

    # ──────────────────────────────────────────────────────────────────────
    # Interaction
    # ──────────────────────────────────────────────────────────────────────
    def _on_canvas_point(self, x: int, y: int, label: int) -> None:
        self._points.append(PointPrompt(x=x, y=y, label=label))
        self._canvas.set_points(self._points)
        self._mark_dirty()
        self._undo_manager.push(self._points, self._masks, self._keyframes)
        if label:
            self._status.showMessage(f"Include point added ({x},{y}) - green. "
                                     "Right-click to add an exclude point.")
        else:
            self._status.showMessage(f"Exclude point added ({x},{y}) - red")

    def _on_canvas_point_removed(self, removed: PointPrompt) -> None:
        self._points[:] = [
            p for p in self._points
            if not (p.x == removed.x and p.y == removed.y and p.label == removed.label)
        ]
        self._canvas.set_points(self._points)
        self._mark_dirty()
        self._undo_manager.push(self._points, self._masks, self._keyframes)
        self._status.showMessage(f"Point deleted ({removed.x},{removed.y})")

    def _on_canvas_point_selected(self, _idx: int) -> None:
        self._status.showMessage("Point selected - press Delete or use the button to remove it")

    def _delete_selected_point(self) -> None:
        if not self._canvas.delete_selected_point():
            self._status.showMessage("Click a point to select it first")
        else:
            self._mark_dirty()

    def _clear_points(self) -> None:
        self._points.clear()
        self._canvas.clear_points()
        self._mark_dirty()
        self._undo_manager.push(self._points, self._masks, self._keyframes)

    def _clear_mask(self) -> None:
        self._masks.pop(self._current_frame_idx, None)
        self._keyframes.discard(self._current_frame_idx)
        self._canvas.clear_mask()
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._mark_dirty()
        self._undo_manager.push(self._points, self._masks, self._keyframes)
        self._status.showMessage(f"Mask cleared (frame {self._current_frame_idx})")

    # ──────────────────────────────────────────────────────────────────────
    # SAM 2 manager / segmentation (image predictor)
    # ──────────────────────────────────────────────────────────────────────
    def _ensure_sam_manager(self) -> SAM2Manager:
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
        # Brush mode: the painted mask is used as a prompt — SAM 2 adapts it
        # into a clean silhouette (like point prompts refine a region).
        if self._canvas.get_edit_mode() == "brush":
            mask = self._masks.get(self._current_frame_idx)
            if mask is None:
                self._status.showMessage(
                    "Brush mode - paint an outline on the frame, then press "
                    "Segment to make SAM 2 adapt it."
                )
                return
            if self._frame_buffer is None:
                return
            frame = self._frame_buffer.get(self._current_frame_idx)
            if frame is None:
                return
            if not self._ensure_sam_manager().is_available:
                self._status.showMessage(
                    "SAM 2 is not installed - your painted mask is used as-is."
                )
                return
            self.setEnabled(False)
            self._loading_overlay.show_message(
                "Refining mask",
                "SAM 2 is adapting your brush outline...",
            )
            self._inference_worker = InferenceWorker(
                self._config.recommendation,
                sam_manager=self._ensure_sam_manager(),
                parent=self,
            )
            self._inference_worker.set_task_mask(frame, mask)
            self._inference_worker.progress.connect(
                lambda msg: self._status.showMessage(msg)
            )
            self._inference_worker.finished.connect(self._on_segment_finished)
            self._inference_worker.start()
            return
        if not self._points:
            QMessageBox.warning(
                self,
                "No prompt",
                "Switch to the Points tool and click on the frame "
                "(left = include, right = exclude) before segmenting.",
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
        self._loading_overlay.hide_message()
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
            self._mark_dirty()
            self._canvas.set_mask(best.mask)
            self._timeline.set_keyframes(self._keyframes)
            self._timeline.set_masked_frames(set(self._masks))
            self._undo_manager.push(self._points, self._masks, self._keyframes)
            self._status.showMessage(f"Mask generated (score={best.score:.3f})")
        else:
            self._status.showMessage("No mask found")

    # ──────────────────────────────────────────────────────────────────────
    # Propagation (video predictor)
    # ──────────────────────────────────────────────────────────────────────
    def _run_propagation(self, direction: int = 1) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._config.recommendation is None:
            QMessageBox.warning(self, "No model", "Run hardware detection first.")
            return
        prompt_mask = self._masks.get(self._current_frame_idx)
        if not self._points and prompt_mask is None:
            QMessageBox.warning(
                self,
                "No prompt",
                "Add correction points (Points tool) or paint a mask "
                "(Brush tool) before propagating.",
            )
            return
        if self._propagation_worker is not None and self._propagation_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A propagation is already running.")
            return

        start, target = self._segment_bounds(direction)
        max_frames = abs(target - start) + 1 if start != target else None

        self._propagation_worker = PropagationWorker(
            self._config.recommendation, self._ensure_sam_manager(), self
        )
        self._propagation_worker.set_task(
            str(self._video_reader.path),
            start_frame=start,
            points=list(self._points) or None,
            total_frames=self._video_reader.total_frames,
            mask=prompt_mask if not self._points else None,
        )
        self._propagation_worker.set_range(max_frames=max_frames, reverse=direction < 0)
        self._propagation_worker.progress.connect(self._on_worker_progress)
        self._propagation_worker.frame_ready.connect(self._on_propagation_frame_ready)
        self._propagation_worker.finished.connect(self._on_propagation_finished)
        self._status.showMessage(
            f"Propagating {'left' if direction < 0 else 'right'} from {start} to {target}... "
            f"({max_frames if max_frames else 'end'} frame(s))"
        )
        self._propagating = True
        self.setEnabled(False)
        self._loading_overlay.show_message(
            "Propagating",
            f"Frames {max_frames if max_frames else self._video_reader.total_frames} ahead...",
        )
        self._propagation_worker.start()

    def _segment_bounds(self, direction: int) -> tuple[int, int]:
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
            self._loading_overlay.set_progress(f"Frame {current} / {total}")
        else:
            self._status.showMessage(stage)
            self._loading_overlay.set_progress(stage)

    def _on_propagation_frame_ready(self, frame_idx: int, _obj_id: int, mask: np.ndarray) -> None:
        self._masks[frame_idx] = mask
        self._keyframes.add(frame_idx)
        self._mark_dirty()
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        if frame_idx == self._current_frame_idx:
            self._canvas.set_mask(mask)

    def _on_propagation_finished(self, result: object) -> None:
        self._loading_overlay.hide_message()
        self.setEnabled(True)
        if isinstance(result, Exception):
            logger.error("Propagation failed: %s", result)
            self._propagating = False
            QMessageBox.critical(self, "Propagation Error", str(result))
            self._status.showMessage("Propagation failed")
            return
        results: list[tuple[int, int, np.ndarray, np.ndarray]] = result
        for frame_idx, _obj_id, mask, _score in results:
            self._masks[frame_idx] = mask
            self._keyframes.add(frame_idx)
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._show_current_frame()
        self._undo_manager.push(self._points, self._masks, self._keyframes)
        self._propagating = False
        n_frames = len(set(f for f, _, _, _ in results))
        self._status.showMessage(f"Propagation complete - {n_frames} frames masked")

    # ──────────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────────
    def _show_export_dialog(self) -> None:
        """Open the professional export dialog."""
        combined = self._combined_masks()
        if not combined:
            QMessageBox.warning(self, "No masks", "Generate masks before exporting.")
            return

        total_frames = self._video_reader.total_frames if self._video_reader else 0
        fallback_frames = self._frame_buffer.max_frames if self._frame_buffer else 0
        objects_payload = [
            {"id": o.obj_id, "name": o.name, "color": o.color}
            for o in self._objects
        ]
        dlg = ExportDialog(
            self,
            video_reader=self._video_reader,
            num_masks=len(combined),
            num_frames=total_frames or fallback_frames,
            timeline_selection=self._timeline_selection,
            objects=objects_payload,
        )
        dlg._current_frame = self._current_frame_idx
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._run_export_from_settings(dlg.settings())

    def _run_export_from_settings(self, settings: dict) -> None:
        """Start an export (possibly multiple files for separate-object mode)."""
        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.warning(self, "Busy", "An export is already running.")
            return

        # Immediate visual feedback between the dialog and the progress window.
        # processEvents() forces an immediate repaint: the prefetch loop below is
        # synchronous CPU work, otherwise the overlay would never be drawn.
        self._loading_overlay.show_message("Preparing export", "Reading frames…")
        QApplication.processEvents()

        fmt = settings["format"]
        base_name = settings.get("file_name", "output")
        output_dir = settings.get("output_dir", Path.cwd() / "exports")
        resolution = settings.get("resolution")
        fps = settings.get("fps", 30.0)
        range_mode = settings.get("range_mode", "all")
        start_frame = settings.get("start_frame", 0)
        end_frame = settings.get("end_frame", 0)
        include_audio = settings.get("include_audio", False)
        background = settings.get("background")
        audio_source: Optional[Path] = None
        if include_audio and fmt != "png" and self._video_reader is not None:
            audio_source = self._video_reader.path

        # Build one or more export jobs from the object mode setting.
        object_mode = settings.get("object_mode", "combined")
        object_id = settings.get("object_id")
        jobs: list[tuple[dict[int, np.ndarray], str]] = []

        if object_mode == "single" and object_id is not None:
            obj = next((o for o in self._objects if o.obj_id == object_id), None)
            if obj is not None and obj.masks:
                jobs.append((dict(obj.masks), base_name))
        elif object_mode == "separate":
            for obj in self._objects:
                if not obj.masks:
                    continue
                safe_name = obj.name.replace(" ", "_")
                jobs.append((dict(obj.masks), f"{base_name}_{safe_name}"))
        else:
            combined = self._combined_masks()
            if combined:
                jobs.append((combined, base_name))

        if not jobs:
            self._loading_overlay.hide_message()
            QMessageBox.warning(self, "No masks", "Nothing to export.")
            return

        # Pre-fetch source frames once (union of all job keys).
        all_keys: set[int] = set()
        for m, _ in jobs:
            all_keys.update(m.keys())
        frames: dict[int, np.ndarray] = {}
        if fmt != "png" and self._frame_buffer is not None:
            for idx in all_keys:
                f = self._frame_buffer.get(idx)
                if f is not None:
                    frames[idx] = f

        self._export_jobs = jobs
        self._export_fmt = fmt
        self._export_frames = frames
        self._export_resolution = resolution
        self._export_fps = fps
        self._export_range_mode = range_mode
        self._export_start_frame = start_frame
        self._export_end_frame = end_frame
        self._export_audio_source = audio_source
        self._export_background = background
        self._export_output_dir = output_dir
        self._export_results = []
        self._export_progress_dlg = ExportProgressDialog(self)
        self._export_started = time.monotonic()
        self._start_next_export_job()

    def _start_next_export_job(self) -> None:
        if not self._export_jobs:
            self._finish_exports()
            return
        self._loading_overlay.hide_message()
        masks, job_name = self._export_jobs.pop(0)
        n_total = len(masks)
        dlg = self._export_progress_dlg
        dlg.update_progress(0, n_total, stage=f"Exporting {job_name}...", elapsed=0)
        self._export_worker = ExportWorker(
            Exporter(output_dir=self._export_output_dir),
            self._export_fmt,
            masks,
            self,
            frames=self._export_frames,
            fps=self._export_fps,
            base_name=job_name,
            resolution=self._export_resolution,
            range_mode=self._export_range_mode,
            start_frame=self._export_start_frame,
            end_frame=self._export_end_frame,
            audio_source=self._export_audio_source,
            background=self._export_background,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_progress_dlg.cancel_requested.connect(self._on_cancel_export)
        self._export_progress_dlg.show()
        self.setEnabled(False)
        self._status.showMessage(f"Exporting {job_name}...")
        self._export_worker.start()

    def _on_cancel_export(self) -> None:
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.cancel()

    def _on_export_progress(self, stage: str, current: int, total: int) -> None:
        dlg = getattr(self, "_export_progress_dlg", None)
        if dlg is None:
            return
        elapsed = time.monotonic() - self._export_started
        dlg.update_progress(current, total, stage=stage, elapsed=elapsed)
        self._status.showMessage(
            f"{stage} - {current}/{total} frames - {elapsed:.0f}s elapsed"
        )

    def _on_export_finished(self, result: object) -> None:
        if isinstance(result, Exception):
            self._export_results.append(("error", str(result)))
        else:
            self._export_results.append(("ok", result))
        # Continue to the next queued job, or finish.
        if self._export_jobs:
            self._start_next_export_job()
        else:
            self._finish_exports()

    def _finish_exports(self) -> None:
        self._loading_overlay.hide_message()
        dlg = self._export_progress_dlg
        if dlg is not None:
            has_error = any(kind == "error" for kind, _ in self._export_results)
            dlg.set_finished(success=not has_error)
            dlg.close()
            self._export_progress_dlg = None
        self.setEnabled(True)
        errors = [msg for kind, msg in self._export_results if kind == "error"]
        paths = [msg for kind, msg in self._export_results if kind == "ok"]
        if errors:
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{errors[0]}")
            return
        if not paths:
            return
        # New files landed in the exports folder — refresh the Library view.
        dlg = getattr(self, "_library_dialog", None)
        if dlg is not None and hasattr(dlg, "panel"):
            dlg.panel.refresh()
        if len(paths) == 1:
            QMessageBox.information(self, "Export OK", f"Saved to:\n{paths[0]}")
        else:
            lines = "\n".join(f"- {Path(p).name}" for p in paths)
            folder = str(Path(paths[0]).parent)
            QMessageBox.information(
                self, "Export OK", f"Saved {len(paths)} files to:\n{folder}\n{lines}"
            )

    def _show_library(self) -> None:
        """Open the Media Library (File > Media Library) as a docked dialog."""
        if getattr(self, "_library_dialog", None) is None:
            self._library_dialog = LibraryDialog(self)
        self._library_dialog.panel.refresh()
        self._library_dialog.show()
        self._library_dialog.raise_()
        self._library_dialog.activateWindow()

    def _on_export_png(self) -> None:
        """Backwards-compatible: opens the export dialog preset to PNG."""
        self._show_export_dialog()

    def _on_export_mov(self) -> None:
        """Backwards-compatible: opens the export dialog."""
        self._show_export_dialog()

    # ──────────────────────────────────────────────────────────────────────
    # Close
    # ──────────────────────────────────────────────────────────────────────
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay") and self._loading_overlay.is_active():
            self._loading_overlay.reposition_overlay()

    def closeEvent(self, event) -> None:  # noqa: N802
        # Preserve unsaved work even if the user chooses to discard the project.
        if self._auto_save_enabled and self._project_dirty and self._video_reader is not None:
            self._perform_autosave()
        if not self._confirm_discard():
            event.ignore()
            return
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
