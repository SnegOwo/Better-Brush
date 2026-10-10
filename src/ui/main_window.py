"""Main application window: orchestrates video, canvas, SAM 2, timeline, export."""

from __future__ import annotations

import difflib
import logging
import os
import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core import licensing
from src.core.config import AppConfig
from src.core.constants import APP_VERSION
from src.core.history import ProjectHistory
from src.core.profiles import ShotProfile, ShotProfileMap, shot_ranges
from src.core.project import PROJECT_EXT, ObjectRecord, ProjectData, ProjectManager
from src.core.scene import (
    OBJECT_PALETTE,
    SceneObject,
    next_object_color,
    next_object_name,
)
from src.core.undo import UndoManager
from src.export.exporter import Exporter
from src.export.render_queue import RenderQueue
from src.hardware import ContentProfile, HardwareDetector, ModelSize, SAM2Profiler
from src.models.sam2_manager import MaskResult, PointPrompt, SAM2Manager
from src.processing.frame_buffer import CleanFrameStore, FrameBuffer
from src.processing.inference_worker import (
    CleanBakeWorker,
    CleanUpWorker,
    InferenceWorker,
    MattingWorker,
    PostProcessWorker,
    PropagationWorker,
    RemovalWorker,
    SceneCutWorker,
)
from src.processing.mask_refine import erode_mask
from src.processing.mask_selection import (
    choose_prompt_mask_index,
    trim_prompt_background,
)
from src.processing.matting import MATTING_MODELS, MATTING_RESOLUTIONS, MattingSettings
from src.processing.qc_coach import QCTipsDatabase
from src.processing.removal import RemovalSettings, default_model_dir
from src.processing.training_data import dataset_summary, export_clip
from src.processing.video_reader import VideoReader
from src.ui.assets import app_icon, logo_path
from src.ui.components import CollapsibleSection, DiagnosticsConsole, Separator
from src.ui.components.activation_dialog import ActivationDialog
from src.ui.components.edition import LiteNoticeDialog, ProFeaturesDialog
from src.ui.components.export_dialog import ExportDialog
from src.ui.components.export_progress import ExportProgressDialog
from src.ui.components.gpu_monitor_dialog import GpuMonitorDialog
from src.ui.components.history_dialog import HistoryDialog
from src.ui.components.library_panel import LibraryDialog
from src.ui.components.loading_overlay import LoadingOverlay
from src.ui.components.matte_inspector import MatteInspectorDialog
from src.ui.components.region_pick_dialog import RegionPickDialog
from src.ui.components.render_queue_dialog import RenderQueueDialog
from src.ui.components.shot_profile_dialog import ShotProfileDialog
from src.ui.components.tutorial_dialog import TutorialDialog
from src.ui.components.welcome_dialog import WelcomeDialog
from src.ui.timeline import Timeline
from src.ui.video_canvas import VideoCanvas
from src.utils.logging_config import log_file_path

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
        parent: QWidget | None = None,
        frames: dict[int, np.ndarray] | None = None,
        fps: float = 30.0,
        base_name: str = "output",
        resolution: tuple[int, int] | None = None,
        range_mode: str = "all",
        start_frame: int = 0,
        end_frame: int = 0,
        audio_source: Path | None = None,
        background: tuple[int, int, int] | None = None,
        alphas: dict[int, np.ndarray] | None = None,
        profile_for_frame: Callable[[int], ShotProfile] | None = None,
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
        self._alphas = dict(alphas) if alphas else {}
        self._profile_for_frame = profile_for_frame
        self._edge_softness = 0
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been requested."""
        return self._cancel.is_set()

    def _apply_shot_transforms(
        self,
        masks: dict[int, np.ndarray],
        frames: dict[int, np.ndarray],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Apply the per-shot view transforms (inset, spill) for export.

        Masks are cached un-transformed so the GUI stays adjustable, so the
        same transforms the canvas previews are applied here on copies - the
        cached masks and frames are left untouched.  A learned soft matte
        already carries its own edge falloff, so those frames skip the binary
        transforms (inset/despill/feather) and keep the matte as-is.
        """
        from src.processing.despill import apply_shot_transform

        out_masks: dict[int, np.ndarray] = {}
        out_frames: dict[int, np.ndarray] = {}
        out_alphas: dict[int, np.ndarray] = {}
        # The export is a single file, so the widest edge softness present in
        # the range is used for every frame rather than changing per shot.
        self._edge_softness = 0
        for index, mask in masks.items():
            alpha = self._alphas.get(index)
            if alpha is not None:
                out_alphas[index] = np.clip(alpha.astype(np.float32), 0.0, 1.0)
                out_masks[index] = alpha > 0.5
                frame = frames.get(index)
                if frame is not None:
                    out_frames[index] = frame
                continue
            profile = self._profile_for_frame(index)
            frame, result = apply_shot_transform(
                frames.get(index),
                mask,
                mask_inset=profile.mask_inset,
                spill_mode=profile.spill_mode,
                spill_color=profile.spill_color,
                spill_tolerance=profile.spill_tolerance,
                spill_radius=profile.spill_radius,
                spill_region=profile.spill_region,
                spill_region_mode=profile.spill_region_mode,
                cast_neutralize=profile.cast_neutralize,
                cast_strength=profile.cast_strength,
            )
            self._edge_softness = max(self._edge_softness, profile.edge_softness)
            out_masks[index] = result
            if frame is not None:
                out_frames[index] = frame
        return out_masks, out_frames, out_alphas

    def run(self) -> None:
        try:
            masks = dict(self._masks)
            frames = dict(self._frames)
            if self._range_mode == "range":
                masks = {
                    i: m
                    for i, m in masks.items()
                    if self._start_frame <= i <= self._end_frame
                }
                frames = {
                    i: f
                    for i, f in frames.items()
                    if self._start_frame <= i <= self._end_frame
                }
            if not masks:
                raise RuntimeError("No masks in the selected frame range")

            alphas: dict[int, np.ndarray] = {}
            if self._profile_for_frame is not None:
                masks, frames, alphas = self._apply_shot_transforms(masks, frames)

            if self._kind == "png":
                self.progress.emit("Writing PNG masks", 0, len(masks))
                dest = self._exporter.export_png_sequence(
                    masks,
                    base_name=self._base_name,
                    on_progress=lambda c, t: self.progress.emit(
                        "Writing PNG masks", c, t
                    ),
                    cancel_event=self._cancel,
                    alphas=alphas or None,
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
                    on_progress=lambda c, t: self.progress.emit(
                        "Encoding frames", c, t
                    ),
                    cancel_event=self._cancel,
                    audio_source=self._audio_source,
                    background=self._background,
                    edge_softness=self._edge_softness,
                    alphas=alphas or None,
                )
            self.finished.emit(dest)
        except Exception as exc:  # noqa: BLE001
            logger.error("Export failed: %s", exc)
            self.finished.emit(exc)


class MainWindow(QMainWindow):
    """Better Brush main window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._window_title_base = "Better Brush"
        self.setWindowTitle(self._window_title_base)
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        # State
        self._config = AppConfig()
        self._auto_recommendation = None
        self._video_reader: VideoReader | None = None
        self._frame_buffer: FrameBuffer | None = None
        self._current_frame_idx = 0
        self._is_playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._points: list[PointPrompt] = []
        self._masks: dict[int, np.ndarray] = {}
        #: Soft alpha mattes of the active object (learned matting).
        self._mattes: dict[int, np.ndarray] = {}
        self._keyframes: set[int] = set()
        # Frames written by propagation.  They are tracked as keyframes too,
        # but unlike a hand-placed prompt they must not act as a boundary when
        # the next shot is being chained together.
        self._propagated_frames: set[int] = set()
        self._sam_manager: SAM2Manager | None = None
        self._inference_worker: InferenceWorker | None = None
        self._propagation_worker: PropagationWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._export_progress_dlg = None
        self._export_started = 0.0
        self._timeline_selection: tuple[int, int] | None = None
        self._play_start = 0
        self._syncing = False
        self._project_path: Path | None = None
        self._project_dirty = False
        self._welcome_shown = False
        self._undo_manager = UndoManager()
        self._brush_stroke_active = False
        self._propagating = False
        self._shot_profiles = ShotProfileMap()
        self._history: ProjectHistory | None = None
        self._history_dirty_timer = QTimer(self)
        self._history_dirty_timer.setSingleShot(True)
        self._history_dirty_timer.setInterval(1500)
        self._history_dirty_timer.timeout.connect(self._record_history_checkpoint)
        self._render_queue_dialog: RenderQueueDialog | None = None
        self._gpu_monitor_dialog: GpuMonitorDialog | None = None
        self._last_export_settings: dict | None = None
        self._qc_tips_database = None

        # Multi-object scene state. ``self._points/_masks/_keyframes`` are
        # aliases into the active object's data.
        self._objects: list[SceneObject] = []
        self._active_object_id: int | None = None

        # Automatic save
        project_settings = ProjectManager.load_user_settings()
        self._auto_save_enabled = bool(project_settings.get("auto_save_enabled", True))
        self._auto_save_interval_s = int(
            project_settings.get("auto_save_interval_s", 60)
        )

        # AMV feature state
        self._scene_cuts: list[int] = []
        # Propagation never crosses a detected scene cut, and cuts are
        # always detected when a video is opened - both are fixed behaviour.
        self._cuts_stop_propagation = True
        self._cuts_detect_on_import = True
        self._smooth_strength = max(
            1, min(10, int(project_settings.get("smooth_strength", 3)))
        )
        self._post_process_enabled = bool(
            project_settings.get("post_process_enabled", self._smooth_strength > 0)
        )
        self._smooth_edges = self._post_process_enabled
        self._mask_inset = 0
        self._console_shortcut_value = str(
            project_settings.get("console_shortcut", "F7")
        )
        self._console: DiagnosticsConsole | None = None
        self._scene_cut_worker: SceneCutWorker | None = None
        self._post_worker: PostProcessWorker | None = None
        self._cleanup_worker: CleanUpWorker | None = None
        self._cleanup_plates: list = []  # list of DirtyPlate
        self._cleanup_bake_worker: CleanBakeWorker | None = None
        self._clean_frame_store = CleanFrameStore()
        # Seed used for inpainting variation ("Regenerate another seed").
        self._cleanup_seed = 0
        self._baked_seed: int | None = None
        # AI object removal (MiniMax-Remover): cleaned frames are baked into
        # their own store and composited over the cleanup result at display
        # and export time.
        self._removal_store = CleanFrameStore()
        self._removal_worker: RemovalWorker | None = None
        # Learned matting (MatAnyone / MatAnyone 2): soft mattes land in the
        # active object's ``mattes`` map and feed the canvas and the exporter.
        self._matting_worker: MattingWorker | None = None

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
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if any(
                    path.endswith(e) for e in (".mp4", ".mov", ".avi", ".mkv", ".webm")
                ):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if any(
                path.lower().endswith(e)
                for e in (".mp4", ".mov", ".avi", ".mkv", ".webm")
            ):
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

        file_menu.addSeparator()

        train_act = QAction("Save &Training Data...", self)
        train_act.setStatusTip(
            "Export this clip's frames and corrected masks as a SAM 2 "
            "fine-tuning dataset (anime / AMV)."
        )
        train_act.triggered.connect(self._on_save_training_data)
        file_menu.addAction(train_act)

        export_menu = menubar.addMenu("&Export")
        export_act = QAction("&Export...", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self._show_export_dialog)
        export_menu.addAction(export_act)
        queue_act = QAction("&Batch Render Queue...", self)
        queue_act.setShortcut(QKeySequence("Ctrl+Shift+Q"))
        queue_act.triggered.connect(self._show_render_queue)
        export_menu.addAction(queue_act)

        inspect_menu = menubar.addMenu("&Inspect")
        self._act_matte_inspector = QAction("&Matte QC Inspector", self)
        self._act_matte_inspector.setShortcuts(
            [QKeySequence("Ctrl+Shift+M"), QKeySequence("Ctrl+M")]
        )
        self._act_matte_inspector.setStatusTip(
            "Compare the source frame with the current matte and QC evidence."
        )
        self._act_matte_inspector.triggered.connect(self._show_matte_inspector)
        inspect_menu.addAction(self._act_matte_inspector)

        self._act_gpu_monitor = QAction("&Performance Monitor", self)
        self._act_gpu_monitor.setShortcut(QKeySequence("Ctrl+Shift+G"))
        self._act_gpu_monitor.setStatusTip(
            "Show live GPU utilization, VRAM, temperature, and render estimate."
        )
        self._act_gpu_monitor.triggered.connect(self._show_gpu_monitor)
        inspect_menu.addAction(self._act_gpu_monitor)

        self._act_shot_profiles = QAction("Shot &Profiles...", self)
        self._act_shot_profiles.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self._act_shot_profiles.setStatusTip(
            "Edit model and matte cleanup settings for the current shot or object."
        )
        self._act_shot_profiles.triggered.connect(self._show_shot_profiles)
        inspect_menu.addAction(self._act_shot_profiles)
        inspect_menu.addSeparator()
        self._act_history = QAction("&Version History...", self)
        self._act_history.setShortcut(QKeySequence("Ctrl+Shift+H"))
        self._act_history.triggered.connect(self._show_history)
        inspect_menu.addAction(self._act_history)

        self._act_checkpoint = QAction("Create &Checkpoint", self)
        self._act_checkpoint.setShortcut(QKeySequence("Ctrl+Alt+S"))
        self._act_checkpoint.triggered.connect(self._create_manual_checkpoint)
        inspect_menu.addAction(self._act_checkpoint)

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

        self._act_pro_features = QAction("Pro vs &Lite…", self)
        self._act_pro_features.triggered.connect(self._show_pro_features)
        help_menu.addAction(self._act_pro_features)

        self._act_activate_pro = QAction("&Activate Pro…", self)
        self._act_activate_pro.triggered.connect(self._show_activation)
        help_menu.addAction(self._act_activate_pro)

        tutorial_act = QAction("&Quick Tutorial", self)
        tutorial_act.triggered.connect(lambda: self._show_tutorial(parent=self))
        help_menu.addAction(tutorial_act)

        log_act = QAction("&Open Log Folder…", self)
        log_act.triggered.connect(self._open_log_folder)
        help_menu.addAction(log_act)

        self._act_console = QAction("&Diagnostics Console", self)
        console_shortcut = QKeySequence(self._console_shortcut_value)
        if console_shortcut.isEmpty():
            console_shortcut = QKeySequence("F7")
            self._console_shortcut_value = "F7"
        self._act_console.setShortcut(console_shortcut)
        self._act_console.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self._act_console.triggered.connect(self._show_diagnostics_console)
        help_menu.addAction(self._act_console)

        about_act = QAction("&About Better Brush", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

        self._on_edition_changed()  # sync the Pro action label on startup

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
                    18,
                    18,
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
        self._canvas.edit_mode_changed.connect(self._on_canvas_edit_mode_changed)
        self._canvas.brush_size_changed.connect(self._on_canvas_brush_size_changed)
        self._canvas.request_segment.connect(self._run_segmentation)
        self._canvas.play_pause.connect(self._toggle_play)
        self._canvas.brush_press.connect(self._on_canvas_brush_press)
        self._canvas.mask_edited.connect(self._on_canvas_mask_edited)
        main_splitter.addWidget(self._canvas)

        # Right panel
        self._setup_right_panel()
        main_splitter.addWidget(self._right_scroll)

        main_splitter.setSizes([1100, 300])
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
        self._combo_removal_res.currentIndexChanged.connect(
            self._on_removal_setting_changed
        )
        self._spin_removal_steps.valueChanged.connect(self._on_removal_setting_changed)
        self._spin_removal_grow.valueChanged.connect(self._on_removal_setting_changed)
        self._spin_removal_feather.valueChanged.connect(
            self._on_removal_setting_changed
        )
        self._chk_removal_tiling.toggled.connect(self._on_removal_setting_changed)
        self._combo_matting_model.currentIndexChanged.connect(
            self._on_matting_setting_changed
        )
        self._combo_matting_res.currentIndexChanged.connect(
            self._on_matting_setting_changed
        )
        self._spin_matting_gamma.valueChanged.connect(
            self._on_matting_setting_changed
        )
        self._spin_matting_grow.valueChanged.connect(self._on_matting_setting_changed)
        self._chk_matting_clamp.toggled.connect(self._on_matting_setting_changed)

        # Populate the objects dropdown once the widget tree exists
        self._refresh_objects_ui()

        self.setCentralWidget(central)

    # ── Right panel ───────────────────────────────────────────────────────
    def _setup_right_panel(self) -> None:
        self._right_scroll = QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._right_scroll.setMinimumWidth(260)
        self._right_scroll.setMaximumWidth(420)
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
        content_hint = QLabel(
            "Pick a content preset, then place points on the subject to begin."
        )
        content_hint.setWordWrap(True)
        content_hint.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        sec_content.add_widget(content_hint)
        pl.addWidget(sec_content)

        # ── AI Model ──────────────────────────────────────────────────────
        sec_model = CollapsibleSection("AI Model", expanded=False)
        self._combo_model = QComboBox()
        self._combo_model.addItem("Auto (recommended)", None)
        for size in (
            ModelSize.SAM3_1,
            ModelSize.TINY,
            ModelSize.SMALL,
            ModelSize.BASE_PLUS,
            ModelSize.LARGE,
            ModelSize.EFFICIENTTAM_S,
            ModelSize.EFFICIENTTAM_TI,
        ):
            if size is ModelSize.SAM3_1:
                label = "SAM 3.1 (Object Multiplex)"
            elif size is ModelSize.EFFICIENTTAM_S:
                label = "EfficientTAM S (light, weak GPUs)"
            elif size is ModelSize.EFFICIENTTAM_TI:
                label = "EfficientTAM Ti (weakest GPUs)"
            else:
                label = size.value
            self._combo_model.addItem(label, size)
        # A fine-tuned checkpoint only appears once one is installed.
        self._refresh_custom_model_entry()
        self._combo_model.setToolTip(
            "SAM 3.1 is an optional high-memory model and requires the sam3 package "
            "plus access to the gated facebook/sam3.1 checkpoint.\n"
            "EfficientTAM is a lightweight model that runs on machines that cannot "
            "host SAM 2.1 Tiny.\n"
            "A fine-tuned model appears here once you drop its checkpoint into "
            "checkpoints/custom/ (File > Save Training Data explains the workflow)."
        )
        sec_model.add_widget(self._combo_model)
        pl.addWidget(sec_model)

        # ── Objects ───────────────────────────────────────────────────────
        sec_objects = CollapsibleSection("Objects")
        obj_row = QHBoxLayout()
        obj_row.setSpacing(4)
        self._combo_object = QComboBox()
        self._combo_object.setToolTip("Active object - choose which character to edit")
        obj_row.addWidget(self._combo_object, stretch=1)
        self._btn_object_color = QPushButton()
        self._btn_object_color.setFixedSize(20, 20)
        self._btn_object_color.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_object_color.setToolTip("Change the mask color for this object")
        self._btn_object_color.clicked.connect(self._on_pick_mask_color)
        obj_row.addWidget(self._btn_object_color)
        self._btn_add_object = QPushButton("+")
        self._btn_add_object.setFixedWidth(28)
        if licensing.is_pro():
            self._btn_add_object.setToolTip("Add object — multiple subjects (Pro)")
        else:
            self._btn_add_object.setToolTip("Add object (2nd + is a Pro feature)")
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
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        self._btn_points = QPushButton("Point")
        self._btn_points.setCheckable(True)
        self._btn_points.setProperty("cssClass", "tool")
        self._btn_points.setToolTip("Point tool · include/exclude prompts")
        self._btn_points.setChecked(True)
        self._tool_group.addButton(self._btn_points)
        self._btn_points.clicked.connect(lambda: self._set_tool("points"))
        tool_row.addWidget(self._btn_points)

        self._btn_brush = QPushButton("Brush")
        self._btn_brush.setCheckable(True)
        self._btn_brush.setProperty("cssClass", "tool")
        self._btn_brush.setToolTip("Brush tool (B) · paint corrections")
        self._tool_group.addButton(self._btn_brush)
        self._btn_brush.clicked.connect(lambda: self._set_tool("brush"))
        tool_row.addWidget(self._btn_brush)
        sec_tools.add_layout(tool_row)

        self._chk_erase = QCheckBox("Eraser")
        sec_tools.add_widget(self._chk_erase)

        self._brush_size_widget = QWidget()
        size_row = QHBoxLayout(self._brush_size_widget)
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(6)
        lbl_size = QLabel("Brush size:")
        lbl_size.setStyleSheet("color: #8b8b94; font-size: 11px;")
        size_row.addWidget(lbl_size)
        self._spin_brush = QSpinBox()
        self._spin_brush.setRange(1, 500)
        self._spin_brush.setValue(20)
        size_row.addWidget(self._spin_brush, stretch=1)
        sec_tools.add_widget(self._brush_size_widget)
        self._brush_size_widget.setVisible(False)  # points is the default tool

        # Tools hint: brush add/erase via Shift
        hint_nav = QLabel("Brush: paint to add · hold Shift while painting to erase")
        hint_nav.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        hint_nav.setWordWrap(True)
        sec_tools.add_widget(hint_nav)

        pl.addWidget(sec_tools)

        pl.addWidget(Separator())

        # ── Segment ───────────────────────────────────────────────────────
        sec_segment = CollapsibleSection("Segment")

        self._btn_segment = QPushButton("Segment Frame")
        self._btn_segment.setProperty("cssClass", "primary")
        self._btn_segment.setToolTip("Create or refresh the mask on the current frame")
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

        self._chk_post_process = QCheckBox("Post-process masks")
        self._chk_post_process.setChecked(self._post_process_enabled)
        self._chk_post_process.setToolTip(
            "Runs edge cleanup and flicker smoothing after propagation.\n"
            "Turn this off to keep propagated masks unchanged."
        )
        self._chk_post_process.toggled.connect(self._on_post_process_toggled)
        sec_segment.add_widget(self._chk_post_process)

        smooth_title = QLabel("Edge smoothness")
        smooth_title.setStyleSheet(
            "color: #d4d4d8; font-size: 12px; background: transparent;"
        )
        sec_segment.add_widget(smooth_title)

        smooth_row = QHBoxLayout()
        smooth_row.setSpacing(6)
        self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(1, 10)
        self._smooth_slider.setValue(self._smooth_strength)
        self._smooth_slider.setEnabled(True)
        self._smooth_slider.setToolTip(
            "Subtle anime-safe cleanup by default. Higher values strengthen\n"
            "edge snapping while preserving thin hair and line-art details.\n"
            "Release the slider to re-apply the value to the current shot."
        )
        smooth_row.addWidget(self._smooth_slider, stretch=1)
        self._btn_apply_smooth = QPushButton("Apply")
        self._btn_apply_smooth.setToolTip(
            "Re-apply the current shot's edge smoothness to existing masks"
        )
        self._btn_apply_smooth.clicked.connect(self._on_smooth_strength_released)
        smooth_row.addWidget(self._btn_apply_smooth)
        self._lbl_smooth_val = QLabel(str(self._smooth_strength))
        self._lbl_smooth_val.setFixedWidth(18)
        self._lbl_smooth_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_smooth_val.setStyleSheet(
            "color: #8b8b94; font-size: 11px; background: transparent;"
        )
        smooth_row.addWidget(self._lbl_smooth_val)
        sec_segment.add_layout(smooth_row)
        self._smooth_slider.valueChanged.connect(self._on_smooth_strength_changed)
        self._smooth_slider.sliderReleased.connect(self._on_smooth_strength_released)

        inset_title = QLabel("Mask inset")
        inset_title.setStyleSheet(
            "color: #d4d4d8; font-size: 12px; background: transparent;"
        )
        sec_segment.add_widget(inset_title)

        inset_row = QHBoxLayout()
        inset_row.setSpacing(6)
        self._inset_slider = QSlider(Qt.Orientation.Horizontal)
        self._inset_slider.setRange(0, 10)
        self._inset_slider.setValue(self._mask_inset)
        self._inset_slider.setEnabled(True)
        self._inset_slider.setToolTip(
            "Shrink the matte inward to trim background color bleeding\n"
            "around the subject. 0 keeps the mask as-is.\n"
            "The canvas updates instantly - no re-segmentation needed."
        )
        inset_row.addWidget(self._inset_slider, stretch=1)
        self._lbl_inset_val = QLabel("0 px")
        self._lbl_inset_val.setFixedWidth(34)
        self._lbl_inset_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_inset_val.setStyleSheet(
            "color: #8b8b94; font-size: 11px; background: transparent;"
        )
        inset_row.addWidget(self._lbl_inset_val)
        sec_segment.add_layout(inset_row)
        self._inset_slider.valueChanged.connect(self._on_mask_inset_changed)

        pl.addWidget(sec_segment)

        pl.addWidget(Separator())

        # ── Propagation ───────────────────────────────────────────────────
        sec_prop = CollapsibleSection("Propagation")

        kf_row = QHBoxLayout()
        kf_row.setSpacing(4)
        btn_prev_kf = QPushButton("\u25c0 Key")
        btn_prev_kf.clicked.connect(lambda: self._goto_keyframe(-1))
        kf_row.addWidget(btn_prev_kf)
        btn_next_kf = QPushButton("Key \u25b6")
        btn_next_kf.clicked.connect(lambda: self._goto_keyframe(1))
        kf_row.addWidget(btn_next_kf)
        sec_prop.add_layout(kf_row)

        prop_row = QHBoxLayout()
        prop_row.setSpacing(4)
        btn_prop_back = QPushButton("\u25c0 Prop")
        btn_prop_back.clicked.connect(lambda: self._run_propagation(-1))
        prop_row.addWidget(btn_prop_back)
        btn_prop_fwd = QPushButton("Prop \u25b6")
        btn_prop_fwd.clicked.connect(lambda: self._run_propagation(1))
        prop_row.addWidget(btn_prop_fwd)
        sec_prop.add_layout(prop_row)

        self._btn_propagate = QPushButton("Propagate to End")
        self._btn_propagate.setProperty("cssClass", "primary")
        self._btn_propagate.setToolTip(
            "Track the active mask to the end of the current shot"
        )
        self._btn_propagate.clicked.connect(lambda: self._run_propagation(1))
        sec_prop.add_widget(self._btn_propagate)

        pl.addWidget(sec_prop)

        pl.addWidget(Separator())

        # ── Scene Cuts ────────────────────────────────────────────────────
        sec_cuts = CollapsibleSection("Scene Cuts")

        cut_row = QHBoxLayout()
        cut_row.setSpacing(4)
        self._btn_detect_cuts = QPushButton("Detect Cuts")
        self._btn_detect_cuts.clicked.connect(self._start_scene_detection)
        cut_row.addWidget(self._btn_detect_cuts, stretch=1)
        self._lbl_cut_count = QLabel("")
        self._lbl_cut_count.setFixedWidth(54)
        self._lbl_cut_count.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_cut_count.setStyleSheet("color: #8b8b94; font-size: 10px;")
        cut_row.addWidget(self._lbl_cut_count)
        sec_cuts.add_layout(cut_row)

        self._btn_clear_cuts = QPushButton("Clear Cuts")
        self._btn_clear_cuts.clicked.connect(self._clear_scene_cuts)
        sec_cuts.add_widget(self._btn_clear_cuts)

        hint_cuts = QLabel(
            "Cuts split the video into shots. Masks never spread across "
            "a cut, so each scene keeps its own edit. Auto-detected when "
            "you open a video."
        )
        hint_cuts.setWordWrap(True)
        hint_cuts.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        sec_cuts.add_widget(hint_cuts)

        pl.addWidget(sec_cuts)

        sec_profiles = CollapsibleSection("Shot Profiles", expanded=False)
        self._btn_shot_profiles = QPushButton("Edit Current Shot")
        self._btn_shot_profiles.setToolTip(
            "Use a different model and matte cleanup profile for each shot or object "
            "(Ctrl+Shift+P)."
        )
        self._btn_shot_profiles.clicked.connect(self._show_shot_profiles)
        sec_profiles.add_widget(self._btn_shot_profiles)
        self._lbl_shot_profile = QLabel("No per-shot profile")
        self._lbl_shot_profile.setWordWrap(True)
        self._lbl_shot_profile.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        sec_profiles.add_widget(self._lbl_shot_profile)
        pl.addWidget(sec_profiles)

        pl.addWidget(Separator())

        # ── Clean Up (subtitles / logos) ──────────────────────────────────
        sec_cleanup = CollapsibleSection("Clean Up")
        sec_cleanup.setToolTip(
            "Removes subtitles and static logos from the source frames.\n"
            "Best-effort: works best on flat-styled anime / AMV content.\n"
            "Applied to the preview and exports while it is active."
        )

        clean_row = QHBoxLayout()
        clean_row.setSpacing(4)
        self._btn_remove_subs = QPushButton("Remove Subtitles")
        self._btn_remove_subs.setToolTip(
            "Detect or guide removal of burned-in subtitles"
        )
        self._btn_remove_subs.clicked.connect(self._on_remove_subtitles)
        clean_row.addWidget(self._btn_remove_subs)
        self._btn_remove_logo = QPushButton("Remove Logo")
        self._btn_remove_logo.setToolTip(
            "Detect or guide removal of a logo or watermark"
        )
        self._btn_remove_logo.clicked.connect(self._on_remove_logo)
        clean_row.addWidget(self._btn_remove_logo)
        sec_cleanup.add_layout(clean_row)

        self._btn_remove_both = QPushButton("Remove Both")
        self._btn_remove_both.setToolTip("Run subtitle and logo cleanup independently")
        self._btn_remove_both.clicked.connect(self._on_remove_both)
        sec_cleanup.add_widget(self._btn_remove_both)

        self._lbl_cleanup_note = QLabel(
            "Scans the video, then patches detected text / watermarks.\n"
            "No cleanup regions detected yet."
        )
        self._lbl_cleanup_note.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        self._lbl_cleanup_note.setWordWrap(True)
        sec_cleanup.add_widget(self._lbl_cleanup_note)

        self._btn_clear_cleanup = QPushButton("Clear Cleanup")
        self._btn_clear_cleanup.clicked.connect(self._clear_cleanup)
        self._btn_clear_cleanup.setEnabled(False)
        sec_cleanup.add_widget(self._btn_clear_cleanup)

        self._btn_mask_to_end = QPushButton("Mask to End")
        self._btn_mask_to_end.setToolTip(
            "Pre-inpaints every frame of the video so scrubbing the timeline\n"
            "is smooth (shows a progress bar while it runs)."
        )
        self._btn_mask_to_end.setEnabled(False)
        self._btn_mask_to_end.clicked.connect(self._on_mask_to_end)
        sec_cleanup.add_widget(self._btn_mask_to_end)

        self._btn_regenerate = QPushButton("Regenerate")
        self._btn_regenerate.setToolTip(
            "Re-runs the inpainting with a different seed if a frame looks bad."
        )
        self._btn_regenerate.setEnabled(False)
        self._btn_regenerate.clicked.connect(self._on_regenerate)
        sec_cleanup.add_widget(self._btn_regenerate)

        self._btn_checkup = QPushButton("Check-up")
        self._btn_checkup.clicked.connect(self._show_checkup)
        sec_cleanup.add_widget(self._btn_checkup)

        pl.addWidget(sec_cleanup)

        pl.addWidget(Separator())

        # ── Object Removal (AI) ──────────────────────────────────────────
        sec_removal = CollapsibleSection("Object Removal", expanded=False)
        sec_removal.setToolTip(
            "AI removes whatever the masks cover and rebuilds the background.\n"
            "Mask the subject (points or brush), then run removal.\n"
            "Needs a CUDA-capable NVIDIA GPU; the model (~2.6 GB) "
            "downloads on first use."
        )

        removal_hint = QLabel(
            "Masks the subject, then MiniMax-Remover paints it out "
            "of the video. Runs over every masked frame."
        )
        removal_hint.setWordWrap(True)
        removal_hint.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        sec_removal.add_widget(removal_hint)

        self._combo_removal_res = QComboBox()
        for res in (352, 480, 720, 1080):
            self._combo_removal_res.addItem(f"{res}p", res)
        res_row = QHBoxLayout()
        res_row.setSpacing(6)
        lbl_res = QLabel("Resolution:")
        lbl_res.setStyleSheet("color: #8b8b94; font-size: 11px;")
        res_row.addWidget(lbl_res)
        res_row.addWidget(self._combo_removal_res, stretch=1)
        sec_removal.add_layout(res_row)

        steps_row = QHBoxLayout()
        steps_row.setSpacing(6)
        lbl_steps = QLabel("Quality steps:")
        lbl_steps.setStyleSheet("color: #8b8b94; font-size: 11px;")
        steps_row.addWidget(lbl_steps)
        self._spin_removal_steps = QSpinBox()
        self._spin_removal_steps.setRange(4, 12)
        self._spin_removal_steps.setValue(6)
        self._spin_removal_steps.setToolTip(
            "Diffusion steps - more steps look better and take longer."
        )
        steps_row.addWidget(self._spin_removal_steps, stretch=1)
        sec_removal.add_layout(steps_row)

        grow_row = QHBoxLayout()
        grow_row.setSpacing(6)
        lbl_grow = QLabel("Mask grow:")
        lbl_grow.setStyleSheet("color: #8b8b94; font-size: 11px;")
        grow_row.addWidget(lbl_grow)
        self._spin_removal_grow = QSpinBox()
        self._spin_removal_grow.setRange(0, 64)
        self._spin_removal_grow.setValue(21)
        self._spin_removal_grow.setToolTip(
            "How far the removal region is expanded (px) so the model sees\n"
            "context around the subject and the patch never seams."
        )
        grow_row.addWidget(self._spin_removal_grow, stretch=1)
        sec_removal.add_layout(grow_row)

        feather_row = QHBoxLayout()
        feather_row.setSpacing(6)
        lbl_feather = QLabel("Edge blend:")
        lbl_feather.setStyleSheet("color: #8b8b94; font-size: 11px;")
        feather_row.addWidget(lbl_feather)
        self._spin_removal_feather = QSpinBox()
        self._spin_removal_feather.setRange(0, 64)
        self._spin_removal_feather.setValue(21)
        self._spin_removal_feather.setToolTip(
            "Feather width (px) blending the rebuilt area into the\n"
            "untouched footage around it."
        )
        feather_row.addWidget(self._spin_removal_feather, stretch=1)
        sec_removal.add_layout(feather_row)

        self._chk_removal_tiling = QCheckBox("Low-VRAM mode")
        self._chk_removal_tiling.setChecked(True)
        self._chk_removal_tiling.setToolTip(
            "Tiled VAE processing: slower, but fits 8 GB laptop GPUs.\n"
            "Leave on unless you have a big card and want maximum speed."
        )
        sec_removal.add_widget(self._chk_removal_tiling)

        self._btn_remove_object = QPushButton("Remove Object")
        self._btn_remove_object.setProperty("cssClass", "primary")
        self._btn_remove_object.setToolTip(
            "Run MiniMax-Remover over every masked frame (progress + cancel)."
        )
        self._btn_remove_object.clicked.connect(self._on_remove_object)
        sec_removal.add_widget(self._btn_remove_object)

        self._btn_clear_removal = QPushButton("Clear Removal")
        self._btn_clear_removal.setToolTip("Drop the removal result and restore the original frames")
        self._btn_clear_removal.setEnabled(False)
        self._btn_clear_removal.clicked.connect(self._clear_removal)
        sec_removal.add_widget(self._btn_clear_removal)

        self._lbl_removal_note = QLabel("No removal result yet.")
        self._lbl_removal_note.setWordWrap(True)
        self._lbl_removal_note.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        sec_removal.add_widget(self._lbl_removal_note)

        pl.addWidget(sec_removal)

        pl.addWidget(Separator())

        # ── Matting (soft alpha) ──────────────────────────────────────────
        sec_matte = CollapsibleSection("Matting", expanded=False)
        sec_matte.setToolTip(
            "Turns the binary mask into a soft alpha matte (hair, motion blur,\n"
            "partial coverage) with a learned video matting model.\n"
            "Mask the subject first - the mask seeds the matting model.\n"
            "The model (~135 MB) downloads on first use."
        )

        matte_hint = QLabel(
            "MatAnyone / MatAnyone 2 build a soft matte around the subject "
            "and track it through the clip. The mask stays editable - the "
            "matte only replaces its alpha."
        )
        matte_hint.setWordWrap(True)
        matte_hint.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        sec_matte.add_widget(matte_hint)

        self._combo_matting_model = QComboBox()
        for model in MATTING_MODELS:
            label = "MatAnyone" if model == "matanyone" else "MatAnyone 2"
            self._combo_matting_model.addItem(label, model)
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        lbl_mmodel = QLabel("Model:")
        lbl_mmodel.setStyleSheet("color: #8b8b94; font-size: 11px;")
        model_row.addWidget(lbl_mmodel)
        model_row.addWidget(self._combo_matting_model, stretch=1)
        sec_matte.add_layout(model_row)

        matte_res_row = QHBoxLayout()
        matte_res_row.setSpacing(6)
        lbl_mres = QLabel("Resolution:")
        lbl_mres.setStyleSheet("color: #8b8b94; font-size: 11px;")
        matte_res_row.addWidget(lbl_mres)
        self._combo_matting_res = QComboBox()
        for res in MATTING_RESOLUTIONS:
            self._combo_matting_res.addItem(f"{res}p", res)
        self._combo_matting_res.setToolTip(
            "Internal matting size (longest side). 480 is the model's\n"
            "training resolution and ~2x faster; raise it for finals\n"
            "when you need the finest hair detail."
        )
        matte_res_row.addWidget(self._combo_matting_res, stretch=1)
        sec_matte.add_layout(matte_res_row)

        gamma_row = QHBoxLayout()
        gamma_row.setSpacing(6)
        lbl_gamma = QLabel("Edge gamma:")
        lbl_gamma.setStyleSheet("color: #8b8b94; font-size: 11px;")
        gamma_row.addWidget(lbl_gamma)
        self._spin_matting_gamma = QDoubleSpinBox()
        self._spin_matting_gamma.setRange(0.1, 10.0)
        self._spin_matting_gamma.setSingleStep(0.1)
        self._spin_matting_gamma.setValue(1.0)
        self._spin_matting_gamma.setToolTip(
            "Gamma applied to the matte: < 1 tightens the edge (less halo),\n"
            "> 1 opens it up (more partial coverage)."
        )
        gamma_row.addWidget(self._spin_matting_gamma, stretch=1)
        sec_matte.add_layout(gamma_row)

        mgrow_row = QHBoxLayout()
        mgrow_row.setSpacing(6)
        lbl_mgrow = QLabel("Matte grow:")
        lbl_mgrow.setStyleSheet("color: #8b8b94; font-size: 11px;")
        mgrow_row.addWidget(lbl_mgrow)
        self._spin_matting_grow = QSpinBox()
        self._spin_matting_grow.setRange(-20, 20)
        self._spin_matting_grow.setValue(0)
        self._spin_matting_grow.setToolTip(
            "Grow (+) or shrink (-) the matte by a few pixels.\n"
            "Useful when the matte clips hair or keeps a background fringe."
        )
        mgrow_row.addWidget(self._spin_matting_grow, stretch=1)
        sec_matte.add_layout(mgrow_row)

        self._chk_matting_clamp = QCheckBox('Keep matte inside the mask')
        self._chk_matting_clamp.setChecked(True)
        self._chk_matting_clamp.setToolTip(
            'Constrains the soft matte to the segmentation mask plus a\n'
            'small margin. Off = the network may matte background it finds\n'
            'plausible (hair, cloth, scenery).'
        )
        sec_matte.add_widget(self._chk_matting_clamp)

        self._btn_run_matting = QPushButton("Run Matting")
        self._btn_run_matting.setProperty("cssClass", "primary")
        self._btn_run_matting.setToolTip(
            "Produce a soft alpha matte for every masked frame (progress + cancel)."
        )
        self._btn_run_matting.clicked.connect(self._on_run_matting)
        sec_matte.add_widget(self._btn_run_matting)

        self._btn_clear_matting = QPushButton("Clear Matting")
        self._btn_clear_matting.setToolTip(
            "Drop the soft mattes and go back to the binary masks"
        )
        self._btn_clear_matting.setEnabled(False)
        self._btn_clear_matting.clicked.connect(self._on_clear_matting)
        sec_matte.add_widget(self._btn_clear_matting)

        self._lbl_matte_note = QLabel("No matte yet.")
        self._lbl_matte_note.setWordWrap(True)
        self._lbl_matte_note.setStyleSheet(
            "color: #7a7a84; font-size: 10px; background: transparent;"
        )
        sec_matte.add_widget(self._lbl_matte_note)

        pl.addWidget(sec_matte)

        pl.addWidget(Separator())

        # ── Export ────────────────────────────────────────────────────────
        sec_export = CollapsibleSection("Export")

        self._btn_export = QPushButton("Export...")
        self._btn_export.setProperty("cssClass", "accent")
        self._btn_export.setToolTip("Open export settings (Ctrl+E)")
        self._btn_export.clicked.connect(self._show_export_dialog)
        sec_export.add_widget(self._btn_export)

        lbl_export_hint = QLabel("PNG alpha, ProRes 4444, H.264")
        lbl_export_hint.setStyleSheet(
            "color: #555560; font-size: 10px; background: transparent;"
        )
        lbl_export_hint.setWordWrap(True)
        sec_export.add_widget(lbl_export_hint)

        pl.addWidget(sec_export)

        pl.addWidget(Separator())

        # ── Hardware (collapsed) ──────────────────────────────────────────
        sec_hw = CollapsibleSection("Hardware", expanded=False)

        self._btn_detect = QPushButton("Detect Hardware")
        self._btn_detect.clicked.connect(self._auto_detect_hardware)
        sec_hw.add_widget(self._btn_detect)

        self._btn_gpu_monitor = QPushButton("Open Performance Monitor")
        self._btn_gpu_monitor.setToolTip(
            "Live GPU utilization, VRAM, temperature, render estimate, and model safety "
            "(Ctrl+Shift+G)."
        )
        self._btn_gpu_monitor.clicked.connect(self._show_gpu_monitor)
        sec_hw.add_widget(self._btn_gpu_monitor)

        self._lbl_hw = QLabel("Hardware: detecting...")
        self._lbl_hw.setWordWrap(True)
        self._lbl_hw.setStyleSheet(
            "color: #8b8b94; font-size: 10px; background: transparent;"
        )
        sec_hw.add_widget(self._lbl_hw)

        pl.addWidget(sec_hw)

        pl.addStretch()

        self._right_scroll.setWidget(panel)

    def _refresh_custom_model_entry(self) -> None:
        """Show the fine-tuned model in the dropdown when one is installed."""
        from src.models.checkpoint_manager import CheckpointManager

        try:
            custom = CheckpointManager().custom_model()
        except Exception:  # noqa: BLE001 - no custom model installed
            return
        self._combo_model.addItem(custom.label, ModelSize.CUSTOM)

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
        self._btn_play.setProperty("cssClass", "transport")
        self._btn_play.setCheckable(True)
        self._btn_play.setToolTip("Play / pause (Space or P)")
        self._btn_play.setFixedWidth(64)
        self._btn_play.clicked.connect(self._toggle_play)
        tl.addWidget(self._btn_play)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setProperty("cssClass", "transport")
        self._btn_stop.setToolTip("Stop playback and return to the play start")
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
        lbl_view.setStyleSheet(
            "color: #8b8b94; font-size: 11px; background: transparent; border: none;"
        )
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

        self._lbl_context = QLabel("No video")
        self._lbl_context.setStyleSheet(
            "color: #8b8b94; font-size: 10px; background: transparent; "
            "border: none; padding-right: 8px;"
        )
        self._status.addPermanentWidget(self._lbl_context)

        # Edition upsell pill (Lite only) — hidden entirely for Pro.
        self._btn_edition = QPushButton(
            f"Lite (free) · Pro €{licensing.PRO_PRICE_EUR} at {licensing.PRO_URL}"
        )
        self._btn_edition.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_edition.setStyleSheet(
            "QPushButton { background: #26262b; border: 1px solid #3a3a42; "
            "border-radius: 9px; color: #ffb454; font-size: 10px; padding: 2px 8px; }"
            "QPushButton:hover { border-color: #4a9eff; color: #ffd18a; }"
        )
        self._btn_edition.setToolTip(
            "You are on the free Lite edition. See what Pro adds."
        )
        self._btn_edition.clicked.connect(self._show_pro_features)
        self._status.addPermanentWidget(self._btn_edition)
        self._on_edition_changed()  # hide the pill entirely for Pro users

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

    def _set_window_title(self, base: str) -> None:
        self._window_title_base = base
        self._refresh_window_title()

    def _refresh_window_title(self) -> None:
        marker = " *" if self._project_dirty and self._video_reader is not None else ""
        self.setWindowTitle(f"{self._window_title_base}{marker}")

    def _refresh_context(self) -> None:
        if not hasattr(self, "_lbl_context"):
            return
        if self._video_reader is None:
            self._lbl_context.setText("No video")
            return
        active = self._active_object()
        object_name = active.name if active is not None else "No object"
        if self._timeline_selection is None:
            range_text = "Full sequence"
        else:
            start, end = self._timeline_selection
            range_text = f"In {start:04d}-{end:04d} ({end - start + 1}f)"
        parts = [object_name, range_text]
        if self._project_dirty:
            parts.append("Unsaved")
        self._lbl_context.setText("  |  ".join(parts))

    def _format_timecode(self, frame_idx: int) -> str:
        fps = self._video_reader.fps if self._video_reader else 30.0
        fps = max(1.0, fps)
        total_s = frame_idx / fps
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        s = int(total_s % 60)
        f = int(frame_idx % round(fps))
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    def _update_transport(self) -> None:
        if hasattr(self, "_lbl_timecode"):
            self._lbl_timecode.setText(self._format_timecode(self._current_frame_idx))
        if hasattr(self, "_lbl_frame_counter"):
            total = self._video_reader.total_frames if self._video_reader else 0
            self._lbl_frame_counter.setText(
                f"{self._current_frame_idx} / {max(0, total - 1)}"
            )
        self._refresh_context()

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
        self._sync_tool_controls(tool)
        if tool == "brush":
            self._status.showMessage("Brush mode - paint directly on the frame")
        else:
            self._status.showMessage(
                "Points mode - Left-click = include (green) · Right-click = exclude (red)"
            )

    def _sync_tool_controls(self, tool: str) -> None:
        self._btn_points.setChecked(tool == "points")
        self._btn_brush.setChecked(tool == "brush")
        self._brush_size_widget.setVisible(tool == "brush")

    def _on_canvas_edit_mode_changed(self, tool: str) -> None:
        self._sync_tool_controls(tool)

    def _on_canvas_brush_size_changed(self, value: int) -> None:
        if self._spin_brush.value() == value:
            return
        self._spin_brush.blockSignals(True)
        self._spin_brush.setValue(value)
        self._spin_brush.blockSignals(False)

    # ──────────────────────────────────────────────────────────────────────
    # AMV feature toggles (persisted in user settings)
    # ──────────────────────────────────────────────────────────────────────
    def _on_post_process_toggled(self, enabled: bool) -> None:
        self._post_process_enabled = bool(enabled)
        self._smooth_edges = self._post_process_enabled
        self._smooth_slider.setEnabled(True)
        ProjectManager.save_user_settings(
            {"post_process_enabled": self._post_process_enabled}
        )
        state = "enabled" if self._post_process_enabled else "disabled"
        self._status.showMessage(f"Post-processing {state}")

    def _on_smooth_strength_changed(self, value: int) -> None:
        self._smooth_strength = max(1, min(10, int(value)))
        self._lbl_smooth_val.setText(str(self._smooth_strength))
        self._shot_profiles.default.edge_smoothness = self._smooth_strength
        active = self._active_object()
        current = self._active_profile()
        current.edge_smoothness = self._smooth_strength
        self._shot_profiles.set_for_shot(
            active.obj_id if active is not None else None,
            self._current_shot_id(),
            current,
        )
        self._update_shot_profile_label()
        ProjectManager.save_user_settings({"smooth_strength": self._smooth_strength})

    def _on_smooth_strength_released(self) -> None:
        if (
            self._video_reader is None
            or not self._masks
            or not self._post_process_enabled
        ):
            return
        if not self.isVisible():
            return
        self._status.showMessage(
            "Applying the new edge smoothness to the current shot…"
        )
        self._run_postprocess()

    def _on_mask_inset_changed(self, value: int) -> None:
        self._mask_inset = max(0, min(10, int(value)))
        self._lbl_inset_val.setText(f"{self._mask_inset} px")
        self._shot_profiles.default.mask_inset = self._mask_inset
        active = self._active_object()
        current = self._active_profile()
        current.mask_inset = self._mask_inset
        self._shot_profiles.set_for_shot(
            active.obj_id if active is not None else None,
            self._current_shot_id(),
            current,
        )
        self._update_shot_profile_label()
        self._mark_dirty()
        # The inset is a view transform over the cached mask, so the canvas can
        # be repainted straight away without touching the SAM session.
        self._show_current_frame()

    def _sync_inset_controls(self, shot_id: int) -> None:
        """Point the inset slider at the stored value for a shot."""
        active = self._active_object()
        profile = self._shot_profiles.get_for_shot(
            active.obj_id if active is not None else None, shot_id
        )
        self._mask_inset = profile.mask_inset
        self._inset_slider.blockSignals(True)
        self._inset_slider.setValue(self._mask_inset)
        self._inset_slider.blockSignals(False)
        self._lbl_inset_val.setText(f"{self._mask_inset} px")

    # ------------------------------------------------------------------ #
    # Spill cleaner / edge finishing
    #
    # The controls live in the Shot Profile dialog (see SpillPanel) so the
    # sidebar stays short; these helpers only resolve the per-shot settings and
    # push a live preview while the dialog is open.
    # ------------------------------------------------------------------ #
    def _spill_settings_for_frame(
        self, index: int
    ) -> tuple[
        str, tuple[int, int, int], int, int, tuple[int, int, int, int] | None, str
    ]:
        """Return the effective spill settings for a frame's shot."""
        profile = self._profile_for_frame(index)
        return (
            profile.spill_mode,
            profile.spill_color or (0, 0, 0),
            profile.spill_tolerance,
            profile.spill_radius,
            profile.spill_region,
            profile.spill_region_mode,
        )

    def _profile_for_frame(self, index: int) -> ShotProfile:
        """Return the effective per-shot profile for a frame."""
        active = self._active_object()
        return self._shot_profiles.get_for_frame(
            index,
            self._total_frames(),
            self._scene_cuts,
            active.obj_id if active is not None else None,
            active.profile if active is not None else None,
        )

    def _apply_spill_preview(
        self, frame_rgb: np.ndarray, mask: np.ndarray | None, index: int
    ) -> np.ndarray:
        """Apply the shot's cast fix then despill to a frame, preview only.

        The cast fix runs even when the edge cleaner is off, because a tinted
        light is a problem in its own right and the two settings are
        independent. Neither ever rewrites the cached frame.
        """
        mode, color, tolerance, radius, region, region_mode = (
            self._spill_settings_for_frame(index)
        )
        profile = self._profile_for_frame(index)
        out = frame_rgb
        if profile.cast_neutralize:
            from src.processing.despill import neutralize_cast

            out = neutralize_cast(out, mask, region, region_mode, profile.cast_strength)
        if mode != "despill":
            return out
        from src.processing.despill import despill_frame

        return despill_frame(out, mask, color, tolerance, radius, region, region_mode)

    def _display_mask(self, index: int) -> np.ndarray | None:
        """Return the cached mask for a frame with the shot's transforms applied.

        Masks are stored raw so the values stay adjustable without
        re-segmenting; the inset erosion and the alpha-key fade both happen
        here, at consumption time.  A learned soft matte replaces the binary
        mask's alpha entirely (it already carries its own edge falloff).
        """
        matte = self._matting_alpha(index)
        if matte is not None:
            return matte
        mask = self._masks.get(index)
        if mask is None:
            return None
        mask = erode_mask(mask, self._mask_inset)
        mode, color, tolerance, radius, region, region_mode = (
            self._spill_settings_for_frame(index)
        )
        if mode == "alpha_key" and mask.any():
            frame = self._frame_buffer.get(index) if self._frame_buffer else None
            if frame is not None:
                from src.processing.despill import despill_alpha

                mask = despill_alpha(
                    mask, frame, color, tolerance, radius, region, region_mode
                )
        return mask

    def _on_spill_panel_changed(self, profile: ShotProfile) -> None:
        """Persist edits coming from the Shot Profile dialog's colour panel."""
        active = self._active_object()
        self._shot_profiles.default.spill_mode = profile.spill_mode
        self._shot_profiles.default.spill_tolerance = profile.spill_tolerance
        self._shot_profiles.default.spill_radius = profile.spill_radius
        self._shot_profiles.default.spill_region = profile.spill_region
        self._shot_profiles.default.spill_region_mode = profile.spill_region_mode
        self._shot_profiles.default.cast_neutralize = profile.cast_neutralize
        self._shot_profiles.default.cast_strength = profile.cast_strength
        self._shot_profiles.default.edge_softness = profile.edge_softness
        if profile.spill_color is not None:
            self._shot_profiles.default.spill_color = profile.spill_color
        self._shot_profiles.set_for_shot(
            active.obj_id if active is not None else None,
            self._current_shot_id(),
            profile,
        )
        self._update_shot_profile_label()
        self._mark_dirty()
        self._show_current_frame()

    def _total_frames(self) -> int:
        if self._video_reader is not None:
            return int(self._video_reader.total_frames)
        return 1

    def _current_shot_id(self) -> int:
        for shot in shot_ranges(self._total_frames(), self._scene_cuts):
            if shot.start_frame <= self._current_frame_idx <= shot.end_frame:
                return shot.shot_id
        return 1

    def _active_profile(self) -> ShotProfile:
        active = self._active_object()
        return self._shot_profiles.get_for_frame(
            self._current_frame_idx,
            self._total_frames(),
            self._scene_cuts,
            active.obj_id if active is not None else None,
            active.profile if active is not None else None,
        )

    def _update_shot_profile_label(self) -> None:
        label = getattr(self, "_lbl_shot_profile", None)
        if label is None:
            return
        if self._video_reader is None:
            label.setText("Open a video to configure shot profiles")
            return
        active = self._active_object()
        profile = self._active_profile()
        model = profile.model.value if profile.model is not None else "Auto model"
        spill = ""
        if profile.spill_active:
            red, green, blue = profile.spill_color
            label_text = "despill" if profile.spill_mode == "despill" else "alpha key"
            area = "whole frame" if profile.spill_region is None else "boxed"
            spill = (
                f"\nspill {label_text} rgb({red},{green},{blue})"
                f" · tol {profile.spill_tolerance} · {area}"
            )
        label.setText(
            f"{active.name if active else 'Object'} · Shot {self._current_shot_id()}\n"
            f"{model} · edge {profile.edge_smoothness}/10 · "
            f"detail {profile.detail_preservation:.0%} · "
            f"inset {profile.mask_inset}px" + spill
        )

    def _show_shot_profiles(self) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        active = self._active_object()
        shots = shot_ranges(self._total_frames(), self._scene_cuts)
        frame = (
            self._frame_buffer.get(self._current_frame_idx)
            if self._frame_buffer
            else None
        )
        dialog = ShotProfileDialog(
            active.name if active is not None else "Object",
            shots,
            self._current_shot_id(),
            self._active_profile(),
            parent=self,
            frame=frame,
        )
        # Preview edits land on the current shot immediately so the panel is
        # live rather than a form you fill in blind and only see after Apply.
        dialog._spill.changed.connect(
            lambda: self._on_spill_panel_changed(dialog.profile())
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        profile = dialog.profile()
        shot_id = dialog.selected_shot_id
        self._shot_profiles.set_for_shot(
            active.obj_id if active is not None else None,
            shot_id,
            profile,
        )
        if dialog.apply_to_object_default and active is not None:
            active.profile = profile.copy()
        self._sync_inset_controls(self._current_shot_id())
        self._sync_removal_controls()
        self._sync_matting_controls()
        self._update_shot_profile_label()
        if profile.model is not None:
            self._recommend_now(force_model=profile.model)
        self._mark_dirty()
        self._status.showMessage(f"Shot {shot_id} processing profile updated")

    # ── Scene cuts ───────────────────────────────────────────────────────
    def _update_cut_label(self) -> None:
        n = len(self._scene_cuts)
        self._lbl_cut_count.setText(f"{n} cut{'s' if n != 1 else ''}")

    def _clear_scene_cuts(self) -> None:
        self._scene_cuts = []
        self._timeline.set_scene_cuts([])
        self._update_cut_label()
        self._update_shot_profile_label()
        self._mark_dirty()

    def _start_scene_detection(self) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._scene_cut_worker is not None and self._scene_cut_worker.isRunning():
            return
        self._btn_detect_cuts.setEnabled(False)
        self._status.showMessage("Detecting scene cuts…")
        # NOTE: we intentionally do NOT disable the whole window here so
        # that the Clean Up (subtitles / logos) buttons stay usable while
        # the background scan runs.  The loading overlay already blocks
        # canvas interaction; the scene worker only touches the timeline.
        self._loading_overlay.show_message("Scene cuts", "Scanning the video for cuts…")
        self._scene_cut_worker = SceneCutWorker(
            str(self._video_reader.path), parent=self
        )
        self._scene_cut_worker.finished.connect(self._on_scene_cut_finished)
        self._scene_cut_worker.start()

    def _on_scene_cut_finished(self, result: object) -> None:
        self._loading_overlay.hide_message()
        self._btn_detect_cuts.setEnabled(True)
        if isinstance(result, Exception):
            self._status.showMessage("Scene cut detection failed")
            return
        self._scene_cuts = [int(c) for c in result]
        self._timeline.set_scene_cuts(self._scene_cuts)
        self._update_cut_label()
        self._update_shot_profile_label()
        self._mark_dirty()
        self._status.showMessage(
            f"Scene cut detection complete - {len(self._scene_cuts)} cut(s) found"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Clean Up (subtitles / logos)
    # ──────────────────────────────────────────────────────────────────────
    def _on_remove_subtitles(self) -> None:
        self._cleanup_with_guide("subtitles", "Removing subtitles…")

    def _on_remove_logo(self) -> None:
        self._cleanup_with_guide("logo", "Removing logo…")

    def _on_remove_both(self) -> None:
        self._cleanup_with_guide("both", "Removing subtitles and logos…")

    def _cleanup_with_guide(self, target: str, note: str) -> None:
        """Run cleanup, letting the user guide the detector with a rectangle.

        The user draws a box over the subtitles / logo on the current frame;
        the plate is then built directly from that region (reliable even for
        always-on burned-in overlays the heuristic cannot find on its own).
        They may also choose plain auto-detection or cancel.
        """
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._cleanup_worker is not None and self._cleanup_worker.isRunning():
            return

        frame = self._frame_buffer.get(self._current_frame_idx)
        kind = target if target in ("subtitles", "logo") else "subtitles"
        dlg = RegionPickDialog(self, frame, kind)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._status.showMessage("Cleanup cancelled")
            return

        if dlg.choice == "auto" or dlg.region is None:
            self._start_cleanup_detection(target, note)
            return
        if dlg.choice == "region":
            self._start_cleanup_detection(target, note, region=dlg.region)
            return
        self._status.showMessage("Cleanup cancelled")

    def _start_cleanup_detection(
        self,
        target: str,
        note: str,
        region: tuple[int, int, int, int] | None = None,
    ) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._cleanup_worker is not None and self._cleanup_worker.isRunning():
            return

        self._btn_remove_subs.setEnabled(False)
        self._btn_remove_logo.setEnabled(False)
        self._btn_remove_both.setEnabled(False)
        self._lbl_cleanup_note.setText(f"Scanning… {note}")
        self._loading_overlay.show_message(note)

        kwargs: dict = {}
        if region is not None:
            if target == "logo":
                kwargs["logo_rect"] = region
            else:  # "subtitles" and "both": guide the subtitle band
                kwargs["subtitle_rect"] = region

        self._cleanup_worker = CleanUpWorker(
            self._video_reader.path, target=target, parent=self, **kwargs
        )
        self._cleanup_worker.progress.connect(self._on_cleanup_progress)
        self._cleanup_worker.finished.connect(self._on_cleanup_finished)
        self._cleanup_worker.start()

    def _on_cleanup_progress(self, msg: str, current: int, total: int) -> None:
        if total > 0:
            self._status.showMessage(f"{msg} {current}/{total}")
        else:
            self._status.showMessage(msg)

    def _on_cleanup_finished(self, result: object) -> None:
        self._loading_overlay.hide_message()
        self._btn_remove_subs.setEnabled(True)
        self._btn_remove_logo.setEnabled(True)
        self._btn_remove_both.setEnabled(True)
        self._status.showMessage("Cleanup detection finished")

        if isinstance(result, Exception):
            logger.error("Cleanup detection failed: %s", result)
            QMessageBox.critical(self, "Clean up error", str(result))
            return

        from src.processing.plates import DirtyPlate

        plates = [
            p
            for p in (result.subtitle_plate, result.logo_plate)
            if isinstance(p, DirtyPlate)
        ]
        if not plates:
            self._lbl_cleanup_note.setText(
                "Nothing detected automatically.\n"
                "Use Remove Subtitles / Remove Logo again and drag a box over\n"
                "the text or watermark to guide the cleanup."
            )
            self._status.showMessage("No cleanup regions detected")
            return

        for plate in plates:
            if plate.region not in [p.region for p in self._cleanup_plates]:
                self._cleanup_plates.append(plate)

        count = len(self._cleanup_plates)
        self._btn_clear_cleanup.setEnabled(True)
        self._set_bake_buttons_enabled(True)
        labels = []
        if result.subtitle_plate is not None:
            labels.append("subtitle band")
        if result.logo_plate is not None:
            labels.append("logo")
        self._lbl_cleanup_note.setText(
            f"Cleaning {count} region(s): {', '.join(labels) or 'none'}.\n"
            f"Preview and exports are patched."
        )
        self._status.showMessage(
            f"Detected {', '.join(labels) or 'nothing'} - patching preview"
        )
        self._refresh_current_frame_display()

    def _clear_cleanup(self) -> None:
        self._cleanup_plates.clear()
        self._btn_clear_cleanup.setEnabled(False)
        self._lbl_cleanup_note.setText("No cleanup regions detected yet.")
        self._status.showMessage("Clean up cleared")
        self._refresh_current_frame_display()
        self._clean_frame_store.clear()
        self._baked_seed = None
        self._set_bake_buttons_enabled(True)

    def _apply_cleanup(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Patch a frame with the active cleanup plates (no-op when none)."""
        return self._apply_clean_frame(frame_rgb, seed=self._cleanup_seed)

    def _apply_clean_frame(self, frame_rgb: np.ndarray, seed: int = 0) -> np.ndarray:
        """Patch a frame with the active cleanup plates at a given ``seed``."""
        if not self._cleanup_plates:
            return frame_rgb
        from src.processing.plates import apply_clean

        return apply_clean(frame_rgb, self._cleanup_plates, seed=seed)

    # ── "Mask to End" bake lifecycle ─────────────────────────────────────
    def _on_mask_to_end(self) -> None:
        """Pre-inpaint every frame into the baked store so scrubbing is smooth.

        Without this the timeline re-runs the live repair every frame while
        scrubbing, which is chunky.  Baking all frames up-front (with a
        progress bar + Cancel) means playback and scrubbing afterwards just
        reads pre-cleaned frames from :attr:`_clean_frame_store`.
        """
        if self._video_reader is None or not self._cleanup_plates:
            return
        if (
            self._cleanup_bake_worker is not None
            and self._cleanup_bake_worker.isRunning()
        ):
            return
        # Bumped seed always re-bakes (the store is keyed to that seed).
        self._baked_seed = None
        self._set_bake_buttons_enabled(False)

        worker = CleanBakeWorker(
            self._video_reader.path,
            list(self._cleanup_plates),
            seed=self._cleanup_seed,
            store=self._clean_frame_store,
            parent=self,
        )
        self._cleanup_bake_worker = worker
        self._loading_overlay.show_message(
            "Baking cleanup…",
            "Warming the timeline so scrubbing is smooth.",
        )
        self._loading_overlay.set_progress_bar(self._video_reader.total_frames)
        self._loading_overlay.enable_cancel(True)
        self._loading_overlay.cancel_requested.connect(self._on_cancel_cleanup_bake)

        worker.progress.connect(self._on_cleanup_bake_progress)
        worker.finished.connect(self._on_cleanup_bake_finished)
        worker.start()

    def _on_regenerate(self) -> None:
        """Re-inpaint with a fresh seed (varied result if the last one was bad).

        A different (non-zero, reproducible) seed re-runs the whole bake so
        the new variation is applied everywhere, not just the current frame.
        """
        if not self._cleanup_plates:
            return
        # Loop a handful of smaller seeds so each click gives a visibly
        # different result while staying deterministic for export / scrubbing.
        self._cleanup_seed = (self._cleanup_seed + 1) % 5
        self._clean_frame_store.clear()
        self._status.showMessage(
            f"Regenerating cleanup with seed {self._cleanup_seed + 1}…"
        )
        self._on_mask_to_end()

    def _on_cleanup_bake_progress(self, msg: str, current: int, total: int) -> None:
        self._loading_overlay.set_progress_value(current)
        if current % 8 == 0 or current == total:
            self._loading_overlay.set_progress(f"Baked {current} / {total} frames")

    def _on_cleanup_bake_finished(self, result: object) -> None:
        self._loading_overlay.cancel_requested.disconnect(self._on_cancel_cleanup_bake)
        self._loading_overlay.enable_cancel(False)
        self._loading_overlay.hide_message()

        if self._cleanup_bake_worker is not None:
            self._cleanup_bake_worker = None

        if isinstance(result, Exception):
            self._status.showMessage("Cleanup bake failed")
            self._set_bake_buttons_enabled(True)
            return
        if result == "cancelled":
            self._status.showMessage("Cleanup bake cancelled")
            self._set_bake_buttons_enabled(True)
            return

        self._baked_seed = self._cleanup_seed
        self._set_bake_buttons_enabled(True)
        self._refresh_current_frame_display()
        self._status.showMessage(
            f"Masked to end - {self._clean_frame_store.count} frame(s) pre-cleaned "
            f"(seed {self._cleanup_seed})"
        )

    def _on_cancel_cleanup_bake(self) -> None:
        """Cancel button pressed on the baking overlay."""
        if (
            self._cleanup_bake_worker is not None
            and self._cleanup_bake_worker.isRunning()
        ):
            self._cleanup_bake_worker.cancel()
            self._status.showMessage("Cancelling cleanup bake…")

    def _set_bake_buttons_enabled(self, enabled: bool) -> None:
        self._btn_mask_to_end.setEnabled(enabled and bool(self._cleanup_plates))
        self._btn_regenerate.setEnabled(enabled and bool(self._cleanup_plates))

    def _invalidate_baked_cleanup(self) -> None:
        """Clear the baked store (e.g. plates changed / video reloaded)."""
        self._clean_frame_store.clear()
        self._baked_seed = None
        if (
            self._cleanup_bake_worker is not None
            and self._cleanup_bake_worker.isRunning()
        ):
            self._cleanup_bake_worker.cancel()
        self._set_bake_buttons_enabled(True)

    def _show_checkup(self) -> None:
        """Open the clean-up check-up dialog (engine + region status)."""
        from src.ui.components.check_up_dialog import CheckUpDialog

        CheckUpDialog(list(self._cleanup_plates), parent=self).exec()

    # ── Object removal (MiniMax-Remover) ──────────────────────────────────
    def _on_removal_setting_changed(self, *_args: object) -> None:
        """Persist the removal controls into the current shot's profile."""
        values = {
            "removal_resolution": int(self._combo_removal_res.currentData()),
            "removal_steps": int(self._spin_removal_steps.value()),
            "removal_grow": int(self._spin_removal_grow.value()),
            "removal_feather": int(self._spin_removal_feather.value()),
            "removal_vae_tiling": bool(self._chk_removal_tiling.isChecked()),
        }
        self._apply_shot_setting(values)

    # ------------------------------------------------------------------ #
    # Learned matting (MatAnyone / MatAnyone 2)
    # ------------------------------------------------------------------ #
    def _on_matting_setting_changed(self, *_args: object) -> None:
        """Persist the matting controls into the current shot's profile."""
        values = {
            "matting_model": self._combo_matting_model.currentData(),
            "matting_resolution": int(self._combo_matting_res.currentData()),
            "matting_gamma": float(self._spin_matting_gamma.value()),
            "matting_grow": int(self._spin_matting_grow.value()),
            "matting_clamp": bool(self._chk_matting_clamp.isChecked()),
        }
        self._apply_shot_setting(values)

    def _apply_shot_setting(self, values: dict) -> None:
        """Write *values* into the default profile and the current shot."""
        for key, value in values.items():
            setattr(self._shot_profiles.default, key, value)
        active = self._active_object()
        current = self._active_profile()
        for key, value in values.items():
            setattr(current, key, value)
        self._shot_profiles.set_for_shot(
            active.obj_id if active is not None else None,
            self._current_shot_id(),
            current,
        )
        self._mark_dirty()

    def _sync_removal_controls(self) -> None:
        """Point the removal controls at the current shot's stored values."""
        profile = self._active_profile()
        self._combo_removal_res.blockSignals(True)
        self._spin_removal_steps.blockSignals(True)
        self._spin_removal_grow.blockSignals(True)
        self._spin_removal_feather.blockSignals(True)
        self._chk_removal_tiling.blockSignals(True)
        index = self._combo_removal_res.findData(profile.removal_resolution)
        self._combo_removal_res.setCurrentIndex(max(0, index))
        self._spin_removal_steps.setValue(profile.removal_steps)
        self._spin_removal_grow.setValue(profile.removal_grow)
        self._spin_removal_feather.setValue(profile.removal_feather)
        self._chk_removal_tiling.setChecked(profile.removal_vae_tiling)
        self._combo_removal_res.blockSignals(False)
        self._spin_removal_steps.blockSignals(False)
        self._spin_removal_grow.blockSignals(False)
        self._spin_removal_feather.blockSignals(False)
        self._chk_removal_tiling.blockSignals(False)

    def _sync_matting_controls(self) -> None:
        """Point the matting controls at the current shot's stored values."""
        profile = self._active_profile()
        # "off" means "never configured yet": show the default model so the
        # control and the effective settings never disagree.
        model = (
            profile.matting_model
            if profile.matting_model in MATTING_MODELS
            else MATTING_MODELS[-1]
        )
        self._combo_matting_model.blockSignals(True)
        self._combo_matting_res.blockSignals(True)
        self._spin_matting_gamma.blockSignals(True)
        self._spin_matting_grow.blockSignals(True)
        index = self._combo_matting_model.findData(model)
        self._combo_matting_model.setCurrentIndex(max(0, index))
        res_index = self._combo_matting_res.findData(profile.matting_resolution)
        self._combo_matting_res.setCurrentIndex(max(0, res_index))
        self._spin_matting_gamma.setValue(profile.matting_gamma)
        self._spin_matting_grow.setValue(profile.matting_grow)
        self._combo_matting_model.blockSignals(False)
        self._combo_matting_res.blockSignals(False)
        self._spin_matting_gamma.blockSignals(False)
        self._spin_matting_grow.blockSignals(False)

    def _matting_settings(self) -> MattingSettings:
        """Build matting settings from the active shot's profile."""
        profile = self._active_profile()
        return MattingSettings(
            model=profile.matting_model,
            resolution=profile.matting_resolution,
            gamma=profile.matting_gamma,
            grow=profile.matting_grow,
        )

    def _on_run_matting(self) -> None:
        """Run learned matting for the active object (or all, combined)."""
        if self._video_reader is None or self._frame_buffer is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._matting_worker is not None and self._matting_worker.isRunning():
            self._status.showMessage("Matting already running...")
            return
        profile = self._active_profile()
        settings = self._matting_settings()
        target = self._active_object()
        if profile.matting_combine_objects and len(self._objects) > 1:
            indices = sorted(
                {
                    idx
                    for obj in self._objects
                    for idx, mask in obj.masks.items()
                    if mask is not None and np.any(mask)
                }
            )
        else:
            if target is None or not target.masks:
                QMessageBox.information(
                    self,
                    "No masks",
                    "Segment or propagate a mask first - the mask seeds the matte.",
                )
                return
            indices = sorted(target.masks)
        if not indices:
            QMessageBox.information(
                self,
                "No masks",
                "Segment or propagate a mask first - the mask seeds the matte.",
            )
            return

        # Only the first masked frame, the frames the user hand-placed
        # prompts on (prompt frames + brush edits) and the first frame of
        # every shot re-seed the tracker - propagated frames are matted by
        # propagation. Seeding on every masked frame would re-run the clip
        # around each of them, which is quadratic in clip length (hours on
        # a long AMV cut).
        hand_placed = self._keyframes - self._propagated_frames
        cut_starts = set(self._scene_cuts)
        prompt_frames = [
            i
            for i in indices
            if i == indices[0] or i in hand_placed or i in cut_starts
        ]

        if profile.matting_combine_objects and len(self._objects) > 1:
            prompt_masks = {
                idx: self._combined_mask_at(idx) for idx in prompt_frames
            }
        else:
            assert target is not None
            prompt_masks = {idx: target.masks[idx] for idx in prompt_frames}

        self._loading_overlay.show_message(
            "Matting",
            "Building the soft matte…",
        )
        self._loading_overlay.set_progress_bar(len(indices))
        self._loading_overlay.enable_cancel(True)
        self._loading_overlay.cancel_requested.connect(self._on_cancel_matting)

        self._matting_worker = MattingWorker(
            str(self._video_reader.path),
            indices,
            prompt_masks,
            settings,
            device="cuda",
            model=settings.model,
            mask_getter=self._combined_mask_at if profile.matting_clamp else None,
            parent=self,
        )
        self._matting_worker.progress.connect(self._on_matting_progress)
        self._matting_worker.alpha_ready.connect(self._on_matting_alpha_ready)
        self._matting_worker.finished.connect(self._on_matting_finished)
        self._matting_worker.start()

    def _on_matting_progress(self, stage: str, current: int, total: int) -> None:
        self._loading_overlay.set_progress_value(current)
        if current % 8 == 0 or current == total:
            self._loading_overlay.set_progress(
                f"Matted {current} / {total} frames"
            )

    def _on_matting_alpha_ready(self, index: int, alpha) -> None:
        """Store a finished soft matte on the active object."""
        target = self._active_object()
        if target is None:
            return
        self._mattes[index] = alpha
        if self._current_frame_idx == index:
            self._refresh_current_frame_display()

    def _on_matting_finished(self, result: object) -> None:
        self._loading_overlay.cancel_requested.disconnect(self._on_cancel_matting)
        self._loading_overlay.enable_cancel(False)
        self._loading_overlay.hide_message()
        self._matting_worker = None
        if isinstance(result, Exception):
            logger.error("Matting failed: %s", result)
            self._lbl_matte_note.setText(f"Matting failed: {result}")
            self._status.showMessage("Matting failed")
            return
        if result == "cancelled":
            self._lbl_matte_note.setText("Matting cancelled.")
            self._status.showMessage("Matting cancelled")
            return
        count = len(self._mattes)
        self._btn_clear_matting.setEnabled(count > 0)
        self._lbl_matte_note.setText(
            f"Soft matte on {count} frame(s).\n"
            "The canvas and exports use the soft alpha."
        )
        self._refresh_current_frame_display()
        self._status.showMessage(f"Matting done - {count} frame(s) matted")

    def _on_cancel_matting(self) -> None:
        """Cancel button pressed on the matting overlay."""
        if self._matting_worker is not None and self._matting_worker.isRunning():
            self._matting_worker.cancel()
            self._status.showMessage("Cancelling matting…")

    def _on_clear_matting(self) -> None:
        """Drop the soft mattes and fall back to the binary masks."""
        self._mattes.clear()
        self._btn_clear_matting.setEnabled(False)
        self._lbl_matte_note.setText("No matte yet.")
        self._status.showMessage("Matting cleared")
        self._mark_dirty()
        self._refresh_current_frame_display()

    def _matting_alpha(self, index: int) -> np.ndarray | None:
        """The soft matte for *index*, or None when the frame is binary."""
        return self._mattes.get(index)

    def _removal_settings(self) -> RemovalSettings:
        """Build removal settings from the active shot's profile."""
        from src.processing.removal import RemovalSettings

        profile = self._active_profile()
        return RemovalSettings(
            resolution=profile.removal_resolution,
            steps=profile.removal_steps,
            mask_grow=profile.removal_grow,
            feather=profile.removal_feather,
            vae_tiling=profile.removal_vae_tiling,
        )

    def _removal_indices(self) -> list[int]:
        """Frames whose combined mask (all objects) is non-empty."""
        frames: set[int] = set()
        for obj in self._objects:
            for idx, mask in obj.masks.items():
                if mask is not None and np.any(mask):
                    frames.add(idx)
        return sorted(frames)

    def _combined_mask_at(self, idx: int) -> np.ndarray:
        """Union of every object's mask on *idx* (bool HxW)."""
        combined: np.ndarray | None = None
        for obj in self._objects:
            mask = obj.masks.get(idx)
            if mask is None:
                continue
            combined = mask.astype(bool).copy() if combined is None else np.logical_or(
                combined, mask
            )
        if combined is None:
            height = self._video_reader.height if self._video_reader else 0
            width = self._video_reader.width if self._video_reader else 0
            return np.zeros((height, width), dtype=bool)
        return combined

    def _on_remove_object(self) -> None:
        """Run AI object removal over every masked frame."""
        if self._video_reader is None or self._frame_buffer is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._removal_worker is not None and self._removal_worker.isRunning():
            self._status.showMessage("Removal already running...")
            return
        indices = self._removal_indices()
        if not indices:
            QMessageBox.information(
                self,
                "No masks",
                "Segment or propagate a mask first — removal rebuilds "
                "whatever the mask covers.",
            )
            return
        if not torch.cuda.is_available():
            QMessageBox.warning(
                self,
                "GPU required",
                "Object removal needs a CUDA-capable NVIDIA GPU.\n"
                "The model runs in half precision on the GPU.",
            )
            return

        device = "cuda"
        settings = self._removal_settings()

        self._loading_overlay.show_message(
            "Removing object",
            "MiniMax-Remover is painting the subject out…",
        )
        self._loading_overlay.set_progress_bar(len(indices))
        self._loading_overlay.enable_cancel(True)
        self._loading_overlay.cancel_requested.connect(self._on_cancel_removal)

        self._removal_worker = RemovalWorker(
            str(self._video_reader.path),
            indices,
            self._combined_mask_at,
            settings,
            model_dir=str(default_model_dir()),
            device=device,
            cleanup_plates=list(self._cleanup_plates),
            cleanup_seed=self._cleanup_seed,
            store=self._removal_store,
            parent=self,
        )
        self._removal_worker.progress.connect(self._on_removal_progress)
        self._removal_worker.finished.connect(self._on_removal_finished)
        self._removal_worker.start()

    def _on_removal_progress(self, stage: str, current: int, total: int) -> None:
        self._loading_overlay.set_progress_value(current)
        if current % 8 == 0 or current == total:
            self._loading_overlay.set_progress(
                f"Removed {current} / {total} frames"
            )

    def _on_removal_finished(self, result: object) -> None:
        self._loading_overlay.cancel_requested.disconnect(self._on_cancel_removal)
        self._loading_overlay.enable_cancel(False)
        self._loading_overlay.hide_message()
        self._removal_worker = None
        if isinstance(result, Exception):
            logger.error("Object removal failed: %s", result)
            self._lbl_removal_note.setText(
                f"Removal failed: {result}"
            )
            self._status.showMessage("Object removal failed")
            return
        if result == "cancelled":
            self._lbl_removal_note.setText("Removal cancelled.")
            self._status.showMessage("Removal cancelled")
            return
        count = self._removal_store.count
        self._btn_clear_removal.setEnabled(count > 0)
        self._lbl_removal_note.setText(
            f"Removed the subject on {count} frame(s).\n"
            "Preview and exports show the rebuilt footage."
        )
        self._refresh_current_frame_display()
        self._status.showMessage(f"Removal done - {count} frame(s) rebuilt")

    def _on_cancel_removal(self) -> None:
        """Cancel button pressed on the removal overlay."""
        if self._removal_worker is not None and self._removal_worker.isRunning():
            self._removal_worker.cancel()
            self._status.showMessage("Cancelling removal…")

    def _clear_removal(self) -> None:
        """Drop the removal result and restore the original frames."""
        self._removal_store.clear()
        self._btn_clear_removal.setEnabled(False)
        self._lbl_removal_note.setText("No removal result yet.")
        self._status.showMessage("Removal cleared")
        self._refresh_current_frame_display()

    def _apply_removed_frame(self, frame_rgb: np.ndarray, idx: int) -> np.ndarray:
        """Replace *frame* with its removed version when one was baked."""
        removed = self._removal_store.get(idx)
        return removed if removed is not None else frame_rgb

    def _show_matte_inspector(self) -> None:
        """Open the zoomable before/after matte QC inspector."""
        if self._video_reader is None or self._frame_buffer is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        mask = self._masks.get(self._current_frame_idx)
        frame = self._frame_buffer.get(self._current_frame_idx)
        if frame is None or mask is None:
            QMessageBox.warning(self, "No mask", "Create a mask on this frame first.")
            return
        profile = self._active_profile()
        from src.processing.mask_refine import refine_mask

        after = refine_mask(
            frame,
            mask,
            smooth_strength=profile.edge_smoothness,
            detail_preservation=profile.detail_preservation,
        )
        previous = None
        if self._current_frame_idx > 0:
            previous = self._masks.get(self._current_frame_idx - 1)
        if self._qc_tips_database is None:
            self._qc_tips_database = QCTipsDatabase()
        dialog = MatteInspectorDialog(
            frame,
            mask,
            after,
            previous,
            profile=profile,
            tips_database=self._qc_tips_database,
            parent=self,
        )
        dialog.apply_suggestion_requested.connect(self._apply_qc_suggestion)
        dialog.exec()

    def _apply_qc_suggestion(self, settings: dict) -> None:
        active = self._active_object()
        if not settings:
            return
        current = self._active_profile()
        values = current.to_dict()
        for key in ("edge_smoothness", "temporal_smoothing", "detail_preservation"):
            if key in settings:
                values[key] = settings[key]
        profile = ShotProfile.from_dict(values)
        self._shot_profiles.set_for_shot(
            active.obj_id if active is not None else None,
            self._current_shot_id(),
            profile,
        )
        self._update_shot_profile_label()
        self._mark_dirty()
        self._status.showMessage("Applied QC Coach suggestion to the current shot")

    # ──────────────────────────────────────────────────────────────────────
    # Scene / multi-object management
    # ──────────────────────────────────────────────────────────────────────
    def _active_object(self) -> SceneObject | None:
        """The currently active object, or None if the scene is empty."""
        for obj in self._objects:
            if obj.obj_id == self._active_object_id:
                return obj
        return self._objects[0] if self._objects else None

    def _reset_scene(self) -> None:
        """Discard every object and create the default single object."""
        self._objects = []
        self._shot_profiles = ShotProfileMap()
        first = SceneObject(obj_id=1, name="Object 1", color=OBJECT_PALETTE[0])
        self._objects.append(first)
        self._active_object_id = first.obj_id
        self._points = first.points
        self._masks = first.masks
        self._mattes = first.mattes
        self._keyframes = first.keyframes
        self._propagated_frames = set()
        if hasattr(self, "_canvas"):
            self._canvas.set_mask_color(QColor(first.color))
        self._refresh_objects_ui()
        self._update_shot_profile_label()

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
        if not licensing.is_pro() and len(self._objects) >= 1:
            QMessageBox.information(
                self,
                "Pro feature",
                "Editing more than one object is a Better Brush Pro feature.\n\n"
                f"Upgrade €{licensing.PRO_PRICE_EUR} (one-time) at "
                f"{licensing.PRO_URL} to cut out several subjects at once.",
            )
            return
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
        self._mattes = remaining.mattes
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
        self._mattes = obj.mattes
        self._keyframes = obj.keyframes
        self._propagated_frames = set()
        self._canvas.set_mask_color(QColor(obj.color))
        self._canvas.set_points(self._points)
        self._show_current_frame()
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._undo_manager.reset(self._points, self._masks, self._keyframes)
        self._refresh_objects_ui()
        self._update_shot_profile_label()
        self._status.showMessage(f"Editing {obj.name}")

    def _on_object_changed(self) -> None:
        idx = self._combo_object.currentIndex()
        if idx < 0:
            return
        self._set_active_object(int(self._combo_object.itemData(idx)))

    def _on_pick_mask_color(self) -> None:
        """Let the user pick a new overlay color for the active object."""
        obj = self._active_object()
        if obj is None:
            return
        picked = QColorDialog.getColor(QColor(obj.color), self, "Mask color")
        if not picked.isValid():
            return
        obj.color = picked.name()
        self._canvas.set_mask_color(picked)
        self._refresh_objects_ui()
        self._mark_dirty()
        self._status.showMessage(f"Mask color for {obj.name} changed")

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
        color_btn = getattr(self, "_btn_object_color", None)
        if color_btn is not None:
            active = self._active_object()
            fill = QColor(active.color) if active else QColor("#00b4f0")
            color_btn.setStyleSheet(
                f"background-color: {fill.name()}; border-radius: 4px;"
                "border: 1px solid #555560;"
            )
        self._refresh_context()

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

    def _combined_mattes(self) -> dict[int, np.ndarray] | None:
        """Element-wise max of every object's soft mattes (None when none)."""
        combined: dict[int, np.ndarray] = {}
        for obj in self._objects:
            for fi, alpha in obj.mattes.items():
                if fi in combined:
                    combined[fi] = np.maximum(combined[fi], alpha)
                else:
                    combined[fi] = alpha.astype(np.float32)
        return combined or None

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
            target = next(
                (k for k in reversed(kfs) if k < self._current_frame_idx), kfs[-1]
            )
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
        except Exception as exc:  # noqa: BLE001 - best-effort auto-detect, logged below
            logger.error("Hardware detection failed: %s", exc)
            self._status.showMessage(f"HW detection failed: {exc}")

    def _recommend_now(self, force_model: ModelSize | None = None) -> None:
        profile = HardwareDetector().detect()
        rec = SAM2Profiler().recommend(
            profile, content=self._config.content_profile, force_model=force_model
        )
        if force_model is None:
            self._auto_recommendation = rec
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
        self._status.showMessage(
            f"Hardware OK - {rec.model_size.value} on {rec.device}"
        )
        logger.info("Hardware recommendation updated: %s", rec)

    def _show_gpu_monitor(self) -> None:
        if (
            self._gpu_monitor_dialog is not None
            and not self._gpu_monitor_dialog.isVisible()
        ):
            self._gpu_monitor_dialog = None
        if self._gpu_monitor_dialog is None:
            rec = self._config.recommendation
            model = rec.model_size.value if rec is not None else ModelSize.TINY.value
            self._gpu_monitor_dialog = GpuMonitorDialog(
                self,
                model=model,
                frame_count=self._total_frames()
                if self._video_reader is not None
                else 0,
                fps=self._video_reader.fps if self._video_reader is not None else 30.0,
                width=self._video_reader.width
                if self._video_reader is not None
                else 1920,
                height=self._video_reader.height
                if self._video_reader is not None
                else 1080,
            )
        self._gpu_monitor_dialog.show()
        self._gpu_monitor_dialog.raise_()
        self._gpu_monitor_dialog.activateWindow()

    def _selected_force_model(self) -> ModelSize | None:
        data = (
            self._combo_model.currentData() if hasattr(self, "_combo_model") else None
        )
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
        self._maybe_show_lite_notice()
        # Quick tutorial on the very first launch
        if ProjectManager.is_first_run():
            ProjectManager.set_first_run_done()
            QTimer.singleShot(0, lambda: self._show_tutorial(parent=self))

    def _maybe_show_lite_notice(self) -> None:
        """One-time WinRAR-style Lite notice; never repeats (even after reset)."""
        if licensing.is_pro():
            return
        settings = ProjectManager.load_user_settings()
        if settings.get("lite_notice_shown"):
            return
        settings["lite_notice_shown"] = True
        ProjectManager.save_user_settings(settings)
        LiteNoticeDialog(self).exec()

    def show_welcome(self) -> None:
        """Public hook called by the app bootstrap to open the launcher."""
        self._show_welcome()

    def _on_welcome_result(self, choice: tuple[str, Path | None]) -> None:
        kind, path = choice
        if kind == "import" and path is not None:
            self._load_video(path)
        elif kind == "open" and path is not None:
            self._open_project(path)

    def _show_pro_features(self) -> None:
        """Comparison of what is included in Lite vs Pro (Help → Pro vs Lite)."""
        dlg = ProFeaturesDialog(self)
        dlg.exec()

    def _show_activation(self) -> None:
        """Open the Activate Pro window (Help → Activate Pro…)."""
        dlg = getattr(self, "_activation_dlg", None)
        if dlg is None:
            dlg = ActivationDialog(self)
            self._activation_dlg = dlg
            dlg.changed.connect(self._on_edition_changed)
        dlg._refresh()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_edition_changed(self) -> None:
        """Edition changed — refresh Pro-sensitive UI state."""
        self._act_activate_pro.setText(
            "Activate Pro…" if not licensing.is_pro() else "Pro edition — licensed"
        )
        btn = getattr(self, "_btn_edition", None)
        if btn is not None:
            btn.setVisible(not licensing.is_pro())
        dlg = getattr(self, "_library_dialog", None)
        if dlg is not None and hasattr(dlg, "panel"):
            dlg.panel._apply_edition()

    def _show_tutorial(self, parent: QWidget | None = None) -> None:
        dialog = TutorialDialog(parent or self)
        dialog.action_requested.connect(
            lambda action: QTimer.singleShot(
                0, lambda: self._handle_tutorial_action(action)
            )
        )
        dialog.exec()

    def _handle_tutorial_action(self, action: str) -> None:
        handlers = {
            "open_video": self._on_open_video,
            "segment": self._run_segmentation,
            "propagate": lambda: self._run_propagation(1),
            "export": self._show_export_dialog,
            "library": self._show_library,
            "matte_qc": self._show_matte_inspector,
            "shot_profiles": self._show_shot_profiles,
            "gpu_monitor": self._show_gpu_monitor,
            "queue": self._show_render_queue,
            "history": self._show_history,
        }
        handler = handlers.get(action)
        if handler is not None:
            handler()

    def _open_log_folder(self) -> None:
        log_dir = log_file_path().parent
        log_dir.mkdir(parents=True, exist_ok=True)
        if not log_file_path().exists():
            logger.info("Creating log file for diagnostics")
        try:
            os.startfile(str(log_dir))
        except OSError as exc:
            logger.warning("Could not open log folder: %s", exc)
            QMessageBox.information(
                self, "Log folder", f"Logs are written to:\n{log_file_path()}"
            )

    def _show_diagnostics_console(self) -> None:
        if self._console is None:
            self._console = DiagnosticsConsole(self)
            self._console.command_submitted.connect(self._run_diagnostics_command)
        self._console.show()
        self._console.raise_()
        self._console.activateWindow()

    def _run_diagnostics_command(self, command: str) -> None:
        if self._console is None:
            return
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            self._console.append_error(f"could not parse command: {exc}")
            return
        if not parts:
            return
        name = parts[0].lower()
        handlers = {
            "help": self._console_help,
            "list": self._console_help,
            "commands": self._console_help,
            "logs": self._console_logs,
            "gpu": self._console_gpu,
            "system": self._console_system,
            "biggest": self._console_biggest,
            "status": self._console_status,
            "settings": self._console_settings,
            "version": self._console_version,
            "shortcut": self._console_shortcut,
            "clear": self._console_clear,
        }
        handler = handlers.get(name)
        if handler is None:
            suggestion = difflib.get_close_matches(name, handlers, n=1)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            self._console.append_error(
                f"unknown command '{name}'.{hint} Type help or list for commands."
            )
            return
        self._console.append_output(f"Running {name}...")
        try:
            result = handler(parts[1:])
        except Exception as exc:  # noqa: BLE001 - console must not crash the app
            self._console.append_error(str(exc))
            return
        if result is not None:
            self._console.append_output(result)

    @staticmethod
    def _console_help(_args: list[str]) -> str:
        return (
            "Commands:\n"
            "  help / list       Show this command list\n"
            "  logs [lines]      Show recent application log lines (default 80)\n"
            "  gpu               Show GPU, VRAM, and live utilization when available\n"
            "  system            Show OS, CPU, RAM, Python, and PyTorch details\n"
            "  biggest           Show the largest model that fits current GPU memory\n"
            "  status            Show the current Better Brush session\n"
            "  settings          Show safe user settings\n"
            "  version           Show the application version\n"
            "  shortcut [keys]   Show or change the console shortcut (for example F8)\n"
            "  clear             Clear this console"
        )

    def _console_logs(self, args: list[str]) -> str:
        count = 80
        if args:
            try:
                count = max(1, min(500, int(args[0])))
            except ValueError as exc:
                raise ValueError("logs expects a line count") from exc
        path = log_file_path()
        if not path.is_file():
            return f"Log file not found: {path}"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-count:]) or "Log file is empty."

    def _console_gpu(self, _args: list[str]) -> str:
        profile = HardwareDetector().detect()
        lines = [
            f"Backend: {profile.best_backend.value}",
            f"Device: {profile.recommended_device}",
        ]
        if not profile.gpus:
            lines.append("GPU: none detected")
            return "\n".join(lines)
        for gpu in profile.gpus:
            usage = self._console_gpu_usage(gpu.index)
            lines.append(
                f"GPU {gpu.index}: {gpu.name} | {gpu.backend.value} | "
                f"VRAM {gpu.available_vram_mb:,}/{gpu.total_vram_mb:,} MB"
            )
            if usage:
                lines.append(f"  {usage}")
        return "\n".join(lines)

    @staticmethod
    def _console_gpu_usage(index: int) -> str:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used = max(0, int(memory.total) - int(memory.free))
                return (
                    f"Live: GPU {rates.gpu}% | VRAM {used // (1024 * 1024):,}/"
                    f"{int(memory.total) // (1024 * 1024):,} MB"
                )
            finally:
                pynvml.nvmlShutdown()
        except Exception as exc:  # noqa: BLE001 - usage is optional
            return f"Live usage unavailable: {exc}"

    def _console_system(self, _args: list[str]) -> str:
        profile = HardwareDetector().detect()
        return "\n".join(
            [
                f"OS: {profile.os_name}",
                f"Python: {profile.python_version}",
                f"PyTorch: {profile.torch_version}",
                f"CPU: {profile.cpu_name}",
                f"CPU cores: {profile.cpu_count_physical} physical / {profile.cpu_count_logical} logical",
                f"RAM: {profile.ram_available_mb:,}/{profile.ram_total_mb:,} MB available",
                f"Backend: {profile.best_backend.value}",
                f"Device: {profile.recommended_device}",
            ]
        )

    def _console_biggest(self, _args: list[str]) -> str:
        profile = HardwareDetector().detect()
        model = SAM2Profiler().largest_model_for_profile(profile)
        if model is None:
            return "No GPU model tier fits the detected free VRAM."
        gpu = max(
            (gpu for gpu in profile.gpus if gpu.total_vram_mb > 0),
            key=lambda item: item.available_vram_mb or item.total_vram_mb,
        )
        return "\n".join(
            [
                f"Largest tier that fits: {model.value}",
                f"GPU: {gpu.name} | free VRAM: {gpu.available_vram_mb:,} MB",
                "This is a capacity estimate, not the recommended setting.",
            ]
        )

    def _console_status(self, _args: list[str]) -> str:
        if self._video_reader is None:
            video = "none"
            frame = "0 / 0"
        else:
            video = self._video_reader.path.name
            frame = f"{self._current_frame_idx} / {max(0, self._video_reader.total_frames - 1)}"
        active = self._active_object()
        recommendation = self._config.recommendation
        return "\n".join(
            [
                f"Better Brush: {APP_VERSION}",
                f"Video: {video}",
                f"Frame: {frame}",
                f"Object: {active.name if active is not None else 'none'}",
                f"Masks: {len(self._masks)} | keyframes: {len(self._keyframes)}",
                f"Tool: {self._canvas.get_edit_mode()}",
                f"Post-process: {'on' if self._post_process_enabled else 'off'} | smoothness: {self._smooth_strength}",
                f"Shot profile: {self._active_profile().name} | model: {self._active_profile().model.value if self._active_profile().model else 'auto'}",
                f"Model: {recommendation.model_size.value if recommendation is not None else 'not selected'}",
                f"Console shortcut: {self._act_console.shortcut().toString()}",
            ]
        )

    def _console_settings(self, _args: list[str]) -> str:
        settings = ProjectManager.load_user_settings()
        safe_keys = (
            "auto_save_enabled",
            "auto_save_interval_s",
            "post_process_enabled",
            "smooth_strength",
            "console_shortcut",
        )
        return "\n".join(f"{key}: {settings.get(key)}" for key in safe_keys)

    @staticmethod
    def _console_version(_args: list[str]) -> str:
        return f"Better Brush {APP_VERSION}"

    def _console_shortcut(self, args: list[str]) -> str:
        if not args:
            return f"Console shortcut: {self._act_console.shortcut().toString()}"
        if len(args) != 1:
            raise ValueError("shortcut expects one key sequence, for example F8")
        candidate = QKeySequence(args[0])
        if candidate.isEmpty():
            raise ValueError(f"invalid shortcut: {args[0]}")
        self._act_console.setShortcut(candidate)
        self._act_console.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self._console_shortcut_value = candidate.toString()
        ProjectManager.save_user_settings(
            {"console_shortcut": self._console_shortcut_value}
        )
        return f"Console shortcut changed to {self._console_shortcut_value}."

    def _console_clear(self, _args: list[str]) -> None:
        if self._console is not None:
            self._console.clear_output()

    def _on_about(self) -> None:
        if licensing.is_pro():
            edition = "Pro (licensed to %s)" % (licensing.current_licensee() or "you")
        else:
            edition = (
                f"Lite (free, forever) — one-time €{licensing.PRO_PRICE_EUR} upgrade at {licensing.PRO_URL} unlocks "
                "multi-object, 4K, audio & no watermark. See Help → Pro vs Lite."
            )
        QMessageBox.about(
            self,
            "About Better Brush",
            f"Better Brush v{APP_VERSION} (beta)\n\n"
            "AI-powered rotoscoping with SAM 2.\n"
            "Import a video, prompt SAM 2 with points, propagate the mask "
            "across the timeline, refine with the brush, then export.\n\n"
            f"Edition: {edition}\n"
            "Proprietary software — © 2026 Sneg, Sneg HQ, a company of Lugunica.\n"
            "See the LICENSE file for the full EULA.\n\n"
            "Solo developer: Sneg — https://sneg.dev",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Project files
    # ──────────────────────────────────────────────────────────────────────
    def _mark_dirty(self) -> None:
        if self._video_reader is not None:
            self._project_dirty = True
            self._refresh_window_title()
            self._refresh_context()
            if hasattr(self, "_history_dirty_timer"):
                self._history_dirty_timer.start()

    def _get_project_history(self) -> ProjectHistory | None:
        if self._video_reader is None:
            return None
        if self._history is None or (
            self._project_path is not None
            and self._history.project_path != self._project_path
        ):
            if self._project_path is not None:
                self._history = ProjectHistory(self._project_path)
            else:
                self._history = ProjectHistory(
                    path=ProjectManager.settings_dir()
                    / "history"
                    / f"{self._video_reader.path.stem}.history"
                )
        return self._history

    def _record_history_checkpoint(self) -> None:
        if self._video_reader is None or not self._project_dirty:
            return
        history = self._get_project_history()
        if history is None:
            return
        try:
            entry = history.create_checkpoint(self._build_project_data(), "Autosave")
            self._status.showMessage(f"Checkpoint v{entry.version} saved")
        except Exception as exc:  # noqa: BLE001
            logger.warning("History checkpoint failed: %s", exc)

    def _create_manual_checkpoint(self) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        from PySide6.QtWidgets import QInputDialog

        label, accepted = QInputDialog.getText(
            self,
            "Create Checkpoint",
            "Checkpoint name",
            text="Manual checkpoint",
        )
        if not accepted:
            return
        history = self._get_project_history()
        if history is None:
            return
        try:
            entry = history.create_checkpoint(
                self._build_project_data(),
                label.strip() or "Manual checkpoint",
            )
            self._status.showMessage(f"Checkpoint v{entry.version} created")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Checkpoint Error", str(exc))

    def _show_history(self) -> None:
        if self._video_reader is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._project_dirty:
            self._record_history_checkpoint()
        history = self._get_project_history()
        if history is None:
            return
        dialog = HistoryDialog(history, parent=self)
        dialog.restore_requested.connect(self._restore_history_version)
        dialog.exec()

    def _restore_history_version(self, version: int) -> None:
        history = self._get_project_history()
        if history is None:
            return
        entry = history.get_entry(version)
        if entry is None:
            QMessageBox.warning(
                self, "History", "That checkpoint is no longer available."
            )
            return
        original_path = self._project_path
        self._open_project(entry.path)
        if self._project_path == entry.path:
            self._project_path = original_path
            self._project_dirty = True
            self._set_window_title(
                f"Better Brush - {self._video_reader.path.name} (restored)"
                if self._video_reader is not None
                else "Better Brush"
            )
            self._mark_dirty()
            self._status.showMessage(f"Restored checkpoint v{version}")

    def _confirm_discard(self) -> bool:
        """Ask what to do with unsaved project work.

        Returns True when it is safe to continue — the work was either saved
        or explicitly discarded.  The dialog offers Save (which opens the
        Save-As flow; the pending action only continues when the save
        actually completed), Discard and Cancel.
        """
        if not self._project_dirty or self._video_reader is None:
            return True
        choice = self._ask_unsaved_changes()
        if choice == "discard":
            return True
        if choice == "save":
            return self._save_project_as_confirmed()
        return False

    def _ask_unsaved_changes(self) -> str:
        """Show the Save / Discard / Cancel dialog; return the choice.

        Returns ``"save"``, ``"discard"`` or ``"cancel"``.  Kept separate
        from :meth:`_confirm_discard` so the decision logic can be exercised
        without a modal dialog.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText("Save changes to this project before continuing?")
        save_button = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_button = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_button)
        box.exec()
        if box.clickedButton() is discard_button:
            return "discard"
        if box.clickedButton() is save_button:
            return "save"
        return "cancel"

    def _save_project_as_confirmed(self) -> bool:
        """Run the Save-As flow; True once the project is safely on disk."""
        if self._video_reader is None:
            QMessageBox.warning(self, "Nothing to save", "Import a video first.")
            return False
        default = self._video_reader.path.with_suffix(".bbproj")
        start = str(self._project_path) if self._project_path else str(default)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            start,
            "Better Brush Project (*.bbproj);;All Files (*)",
        )
        if not path:
            # The file dialog was cancelled: keep the user in the project.
            return False
        self._save_project(Path(path))
        return not self._project_dirty

    def _reset_to_blank(self) -> None:
        """Close any loaded video / project and return to a blank state."""
        self._pause_playback()
        if (
            self._propagation_worker is not None
            and self._propagation_worker.isRunning()
        ):
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
        self._history = None
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
        self._set_window_title("Better Brush")
        self._update_transport()
        self._status.showMessage(
            "New project - import a video or drag and drop one here"
        )

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
        self._history = None

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
                        mattes=dict(getattr(rec, "mattes", {}) or {}),
                        points=[PointPrompt(**p) for p in rec.points],
                        keyframes=set(rec.keyframes),
                        profile=rec.profile,
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
        self._shot_profiles = ShotProfileMap(
            default=data.default_profile,
            records=data.shot_profiles,
        )
        self._smooth_strength = max(
            1, min(10, self._shot_profiles.default.edge_smoothness)
        )
        self._smooth_slider.blockSignals(True)
        self._smooth_slider.setValue(self._smooth_strength)
        self._smooth_slider.blockSignals(False)
        self._lbl_smooth_val.setText(str(self._smooth_strength))
        self._mask_inset = self._shot_profiles.default.mask_inset
        self._inset_slider.blockSignals(True)
        self._inset_slider.setValue(self._mask_inset)
        self._inset_slider.blockSignals(False)
        self._lbl_inset_val.setText(f"{self._mask_inset} px")
        self._update_shot_profile_label()

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
        self._scene_cuts = [int(c) for c in data.scene_cuts]
        self._timeline.set_scene_cuts(self._scene_cuts)
        # A project stores a keyframe for every masked frame, so the stored set
        # is treated as propagation output; a fresh prompt placed by the user
        # is what marks a real boundary.
        self._propagated_frames = set(self._keyframes)
        self._update_cut_label()
        self._update_shot_profile_label()
        self._undo_manager.reset(self._points, self._masks, self._keyframes)
        name = data.video_name or path.stem
        self._set_window_title(f"Better Brush - {name} (project)")
        self._refresh_context()
        self._status.showMessage(
            f"Project loaded - {len(self._masks)} frame(s) masked, "
            f"{len(self._points)} point(s)"
        )

    def _resolve_video_path(self, data: ProjectData) -> Path | None:
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
            self,
            "Locate Video",
            start,
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)",
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
                "fps": self._video_reader.fps,
                "width": self._video_reader.width,
                "height": self._video_reader.height,
                "total_frames": self._video_reader.total_frames,
            },
            points=[{"x": p.x, "y": p.y, "label": p.label} for p in self._points],
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
                    mattes=dict(obj.mattes),
                    profile=obj.profile,
                )
                for obj in self._objects
            ],
            active_object=self._active_object_id,
            scene_cuts=list(self._scene_cuts),
            default_profile=self._shot_profiles.default,
            shot_profiles=self._shot_profiles.to_records(
                self._total_frames(),
                self._scene_cuts,
            ),
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
            self._set_window_title(f"Better Brush - {data.video_name} (project)")
            self._refresh_context()
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
        ProjectManager.save_user_settings(
            {"auto_save_enabled": self._auto_save_enabled}
        )
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
            self,
            "Open Video",
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)",
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
            self._clear_cleanup()  # plates are tied to the previous video
            self._clear_removal()  # removal results are tied to the previous video
            self._scene_cuts = []
            self._timeline.set_scene_cuts([])
            self._update_cut_label()
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
            self._set_window_title(f"Better Brush - {path.name}")
            self._refresh_context()
            self._status.showMessage(
                f"Loaded {path.name} - {self._video_reader.width}x{self._video_reader.height} "
                f"@ {self._video_reader.fps:.2f} fps"
            )
            if self._cuts_detect_on_import:
                self._start_scene_detection()
        except Exception as exc:  # noqa: BLE001 - shown to the user in the dialog below
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
            frame = self._apply_cleanup(frame)
            frame = self._draw_cleanup_regions(frame)
            frame = self._apply_removed_frame(frame, self._current_frame_idx)
            mask = self._display_mask(self._current_frame_idx)
            # Despill preview is a consumption-time transform too, so the cached
            # frame on disk is never rewritten.
            frame = self._apply_spill_preview(frame, mask, self._current_frame_idx)
            self._canvas.set_frame(frame)
            self._canvas.set_mask(mask)
            self._canvas.set_points(self._points)
            self._timeline.set_current_frame(self._current_frame_idx)
            self._update_transport()

    def _draw_cleanup_regions(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Outline active cleanup regions so the user sees what is removed."""
        if not self._cleanup_plates:
            return frame_rgb
        import cv2

        out = frame_rgb.copy()
        for plate in self._cleanup_plates:
            x, y, w, h = plate.region
            color = (230, 120, 60) if plate.mode == "logo" else (60, 180, 230)
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, 2)
        return out

    def _refresh_current_frame_display(self) -> None:
        """Repaint the current frame (e.g. after cleanup plates change)."""
        self._show_current_frame()

    def _set_frame(self, idx: int) -> None:
        total = self._video_reader.total_frames if self._video_reader else 0
        upper = max(0, total - 1)
        self._current_frame_idx = max(0, min(int(idx), upper))
        self._show_current_frame()
        self._update_shot_profile_label()

    def _on_timeline_selection_changed(self, start: int, end: int) -> None:
        if start < 0 or end < 0 or start > end:
            self._timeline_selection = None
            self._refresh_context()
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
        interval_ms = (
            int(1000 / self._video_reader.fps) if self._video_reader.fps > 0 else 33
        )
        self._play_timer.start(interval_ms)
        self._is_playing = True
        self._btn_play.setChecked(True)
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
            self._status.showMessage(
                f"Include point added ({x},{y}) - green. "
                "Right-click to add an exclude point."
            )
        else:
            self._status.showMessage(f"Exclude point added ({x},{y}) - red")

    def _on_canvas_point_removed(self, removed: PointPrompt) -> None:
        self._points[:] = [
            p
            for p in self._points
            if not (p.x == removed.x and p.y == removed.y and p.label == removed.label)
        ]
        self._canvas.set_points(self._points)
        self._mark_dirty()
        self._undo_manager.push(self._points, self._masks, self._keyframes)
        self._status.showMessage(f"Point deleted ({removed.x},{removed.y})")

    def _on_canvas_point_selected(self, _idx: int) -> None:
        self._status.showMessage(
            "Point selected - press Delete or use the button to remove it"
        )

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
        self._mattes.pop(self._current_frame_idx, None)
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
    def _recommendation_for_active_profile(self):
        recommendation = self._config.recommendation
        if recommendation is None:
            return None
        model = self._active_profile().model
        if model is None:
            return self._auto_recommendation or recommendation
        try:
            profile = HardwareDetector().detect()
            return SAM2Profiler().recommend(
                profile,
                content=self._config.content_profile,
                force_model=model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build shot model recommendation: %s", exc)
            return recommendation

    def _ensure_sam_manager(self, recommendation=None) -> SAM2Manager:
        selected = recommendation or self._recommendation_for_active_profile()
        if selected is None:
            raise RuntimeError("No model recommendation available")
        if self._sam_manager is not None:
            current = self._sam_manager.rec
            if (
                current.model_size != selected.model_size
                or current.device != selected.device
            ):
                self._sam_manager.unload()
                self._sam_manager = None
        if self._sam_manager is None:
            self._sam_manager = SAM2Manager(selected)
        return self._sam_manager

    def _run_segmentation(self) -> None:
        if self._frame_buffer is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._config.recommendation is None:
            QMessageBox.warning(self, "No model", "Run hardware detection first.")
            return
        recommendation = self._recommendation_for_active_profile()
        if recommendation is None:
            QMessageBox.warning(self, "No model", "Run hardware detection first.")
            return
        # A mask painted on this frame is the source of truth: Segment
        # refines it instead of replacing it with a bare point prompt, which
        # is how brush corrections used to vanish on the next click.
        existing_mask = self._masks.get(self._current_frame_idx)
        mask = existing_mask
        if mask is None and self._canvas.get_edit_mode() == "brush":
            self._status.showMessage(
                "Brush mode - paint an outline on the frame, then press "
                "Segment to make SAM 2 adapt it."
            )
            return
        if mask is not None:
            if self._frame_buffer is None:
                return
            frame = self._frame_buffer.get(self._current_frame_idx)
            if frame is None:
                return
            if not self._ensure_sam_manager(recommendation).is_available:
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
                recommendation,
                sam_manager=self._ensure_sam_manager(recommendation),
                parent=self,
            )
            # The user's points ride along: without them the box derived
            # from a stale (smaller) mask confines SAM 2 to that extent and
            # the segmentation silently loses the subject.
            self._inference_worker.set_task_mask(
                frame, mask, points=list(self._points)
            )
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
        self._loading_overlay.show_message(
            "Segmenting", "SAM 2 is cutting out the subject…"
        )

        self._inference_worker = InferenceWorker(
            recommendation,
            sam_manager=self._ensure_sam_manager(recommendation),
            parent=self,
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
            masks = [item.mask for item in results]
            scores = [item.score for item in results]
            index = choose_prompt_mask_index(masks, scores, self._points)
            best = results[index]
            selected_mask = best.mask
            if self._frame_buffer is not None:
                frame = self._frame_buffer.get(self._current_frame_idx)
                if frame is not None:
                    selected_mask = trim_prompt_background(
                        frame,
                        selected_mask,
                        self._points,
                    )
            self._masks[self._current_frame_idx] = selected_mask
            self._keyframes.add(self._current_frame_idx)
            self._mark_dirty()
            self._canvas.set_mask(selected_mask)
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
        recommendation = self._recommendation_for_active_profile()
        if recommendation is None:
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
        if (
            self._propagation_worker is not None
            and self._propagation_worker.isRunning()
        ):
            QMessageBox.warning(self, "Busy", "A propagation is already running.")
            return

        start, target = self._segment_bounds(direction)
        segments = self._shot_segments(direction)
        n_shots = len(segments)
        to_track = sum(abs(last - first) + 1 for first, last in segments)

        self._propagation_worker = PropagationWorker(
            recommendation,
            self._ensure_sam_manager(recommendation),
            self,
        )
        self._propagation_worker.set_task(
            str(self._video_reader.path),
            start_frame=start,
            points=list(self._points) or None,
            total_frames=self._video_reader.total_frames,
            # Both are passed: a painted mask survives alongside points
            # instead of being discarded.
            mask=prompt_mask,
        )
        self._propagation_worker.set_segments(segments)
        self._propagation_worker.set_range(reverse=direction < 0)
        self._propagation_worker.progress.connect(self._on_worker_progress)
        self._propagation_worker.frame_ready.connect(self._on_propagation_frame_ready)
        self._propagation_worker.finished.connect(self._on_propagation_finished)
        where = "left" if direction < 0 else "right"
        shot_note = f" across {n_shots} shot(s)" if n_shots > 1 else ""
        self._status.showMessage(
            f"Propagating {where} from {start} to {target}... "
            f"({to_track} frame(s){shot_note})"
        )
        self._propagating = True
        self.setEnabled(False)
        self._loading_overlay.show_message(
            "Propagating",
            f"{to_track} frame(s){shot_note}...",
        )
        self._propagation_worker.start()

    def _segment_bounds(self, direction: int) -> tuple[int, int]:
        cur = self._current_frame_idx
        n = self._video_reader.total_frames if self._video_reader else cur + 1
        kfs = sorted(self._keyframes)
        if direction >= 0:
            nxt = next((k for k in kfs if k > cur), None)
            target = (nxt - 1) if nxt is not None else n - 1
            if self._cuts_stop_propagation and self._scene_cuts:
                cut = next((c for c in self._scene_cuts if c > cur), None)
                if cut is not None:
                    target = min(target, cut - 1)
            return cur, max(target, cur)
        prv = next((k for k in reversed(kfs) if k < cur), None)
        target = (prv + 1) if prv is not None else 0
        if self._cuts_stop_propagation and self._scene_cuts:
            cut = next((c for c in reversed(self._scene_cuts) if c < cur), None)
            if cut is not None:
                target = max(target, cut + 1)
        return cur, min(target, cur)

    def _shot_segments(self, direction: int) -> list[tuple[int, int]]:
        """Contiguous propagation segments covering every shot of the clip.

        :meth:`_segment_bounds` deliberately stops at the first detected cut so
        that a single tracking run never spans a camera change.  Propagation
        used to stop there as well, which left every following scene without a
        mask; since the exporter walks the frames that actually own a mask,
        "everything" and "selection" both silently produced the opening scene
        only.  Each shot is therefore returned as its own segment and
        :class:`PropagationWorker` chains them, re-seeding every shot after the
        first from the last mask of the previous one.

        Returns:
            ``(first, last)`` frame pairs in propagation order, contiguous and
            non-overlapping.  A single pair means one uninterrupted run.
        """
        start, first_end = self._segment_bounds(direction)
        total = self._video_reader.total_frames if self._video_reader else start + 1
        cuts = (
            sorted({int(c) for c in self._scene_cuts if 0 < int(c) < total})
            if self._cuts_stop_propagation
            else []
        )
        # Only hand-placed prompts are boundaries: frames written by
        # propagation are keyframes as well, and treating them as boundaries
        # would shred the clip into thousands of one-frame segments.
        stops = self._keyframes - self._propagated_frames

        if direction >= 0:
            if first_end <= start:
                # Nothing resolved ahead (the dense-keyframe case, which
                # ``_run_propagation`` used to read as "track to the end"), so
                # run out to the end of the shot holding the current frame.
                first_end = next((c - 1 for c in cuts if c > start), total - 1)
            segments = [(start, first_end)]
            edges = sorted(
                {c for c in cuts if c > first_end}
                | {k for k in stops if first_end < k < total}
            )
            for i, edge in enumerate(edges):
                last = edges[i + 1] - 1 if i + 1 < len(edges) else total - 1
                segments.append((edge, last))
            return segments

        # Backward: the first segment never leaves the shot holding the current
        # frame, whose first frame is the highest cut at or below it.
        below = [c for c in cuts if c <= start]
        floor = below[-1] if below else 0
        prv = next(
            (k for k in sorted(self._keyframes, reverse=True) if k < start), None
        )
        if prv is not None and prv >= floor:
            floor = prv + 1
        segments = [(floor, start)]
        upper = floor - 1
        edges = sorted(
            {c for c in cuts if 0 < c < floor} | {k for k in stops if 0 < k < floor},
            reverse=True,
        )
        for edge in edges:
            if edge > upper:
                continue
            segments.append((edge, upper))
            upper = edge - 1
        if upper >= 0:
            segments.append((0, upper))
        return segments

    def _on_worker_progress(self, stage: str, current: int, total: int) -> None:
        if total > 0:
            self._status.showMessage(f"{stage}... {current}/{total}")
            self._loading_overlay.set_progress(f"Frame {current} / {total}")
        else:
            self._status.showMessage(stage)
            self._loading_overlay.set_progress(stage)

    def _on_propagation_frame_ready(
        self, frame_idx: int, _obj_id: int, mask: np.ndarray
    ) -> None:
        self._masks[frame_idx] = mask
        self._mattes.pop(frame_idx, None)  # the binary mask changed
        self._keyframes.add(frame_idx)
        self._propagated_frames.add(frame_idx)
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
            self._mattes.pop(frame_idx, None)  # the binary mask changed
            self._keyframes.add(frame_idx)
            self._propagated_frames.add(frame_idx)
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._show_current_frame()
        self._undo_manager.push(self._points, self._masks, self._keyframes)
        self._propagating = False
        n_frames = len({f for f, _, _, _ in results})
        self._status.showMessage(f"Propagation complete - {n_frames} frames masked")
        if not self._post_process_enabled:
            return
        self._run_postprocess()

    def _run_postprocess(self) -> None:
        if not self._post_process_enabled:
            return
        if not self._masks or self._frame_buffer is None:
            return
        if self._post_worker is not None and self._post_worker.isRunning():
            return
        if not self.isVisible():
            return
        active = self._active_object()
        object_id = active.obj_id if active is not None else None
        object_default = active.profile if active is not None else None

        def profile_for_frame(index: int) -> tuple[float, bool, float, int, int]:
            selected = self._shot_profiles.get_for_frame(
                index,
                self._total_frames(),
                self._scene_cuts,
                object_id,
                object_default,
            )
            return (
                selected.edge_smoothness,
                selected.temporal_smoothing,
                selected.detail_preservation,
                1 if selected.detail_preservation >= 0.8 else 2,
                selected.mask_inset,
            )

        self.setEnabled(False)
        self._loading_overlay.show_message(
            "Post-processing",
            "Refining mask edges / smoothing temporally…",
        )
        self._post_worker = PostProcessWorker(
            lambda idx: self._frame_buffer.get(idx) if self._frame_buffer else None,
            dict(self._masks),
            refine_edges=self._post_process_enabled,
            temporal_smooth=self._post_process_enabled,
            smooth_strength=self._smooth_strength,
            detail_preservation=self._shot_profiles.default.detail_preservation,
            temporal_window=1,
            profile_resolver=profile_for_frame,
            prompt_points=list(self._points),
            parent=self,
        )
        self._post_worker.progress.connect(self._on_worker_progress)
        self._post_worker.finished.connect(self._on_postprocess_finished)
        self._post_worker.start()

    def _on_postprocess_finished(self, result: object) -> None:
        self._loading_overlay.hide_message()
        self.setEnabled(True)
        if isinstance(result, Exception):
            self._status.showMessage("Post-processing failed")
            return
        self._masks.clear()
        self._masks.update(result)
        self._keyframes.clear()
        self._keyframes.update(i for i in result)
        self._propagated_frames = set(result)
        self._timeline.set_keyframes(self._keyframes)
        self._timeline.set_masked_frames(set(self._masks))
        self._show_current_frame()
        self._undo_manager.push(self._points, self._masks, self._keyframes)
        self._mark_dirty()
        self._status.showMessage(
            f"Post-processing complete - {len(self._masks)} masks refined"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────────
    def _show_render_queue(self) -> None:
        """Open the persistent multi-project render queue."""
        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.warning(
                self, "Busy", "Finish the current export before starting the queue."
            )
            return
        if (
            self._project_path is None
            and self._video_reader is not None
            and self._project_dirty
        ):
            answer = QMessageBox.question(
                self,
                "Save project first",
                "Save this project before adding it to the batch queue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._on_save_project_as()
        try:
            queue = RenderQueue(ProjectManager.settings_dir() / "render_queue.json")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Queue Error", f"Cannot open render queue:\n{exc}"
            )
            return
        if (
            self._render_queue_dialog is not None
            and not self._render_queue_dialog.isVisible()
        ):
            self._render_queue_dialog = None
        if self._render_queue_dialog is None:
            self._render_queue_dialog = RenderQueueDialog(
                self,
                queue=queue,
                current_project=self._project_path,
                current_video=self._video_reader.path if self._video_reader else None,
                default_settings=self._last_export_settings,
            )
        self._render_queue_dialog.show()
        self._render_queue_dialog.raise_()
        self._render_queue_dialog.activateWindow()

    def _show_export_dialog(self) -> None:
        """Open the professional export dialog."""
        combined = self._combined_masks()
        if not combined:
            QMessageBox.warning(self, "No masks", "Generate masks before exporting.")
            return

        total_frames = self._video_reader.total_frames if self._video_reader else 0
        fallback_frames = self._frame_buffer.max_frames if self._frame_buffer else 0
        objects_payload = [
            {"id": o.obj_id, "name": o.name, "color": o.color} for o in self._objects
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
        self._last_export_settings = dict(dlg.settings())
        self._run_export_from_settings(self._last_export_settings)

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
        audio_source: Path | None = None
        if include_audio and fmt != "png" and self._video_reader is not None:
            audio_source = self._video_reader.path

        # Build one or more export jobs from the object mode setting.
        object_mode = settings.get("object_mode", "combined")
        object_id = settings.get("object_id")
        jobs: list[tuple[dict[int, np.ndarray], dict[int, np.ndarray] | None, str]] = []

        if object_mode == "single" and object_id is not None:
            obj = next((o for o in self._objects if o.obj_id == object_id), None)
            if obj is not None and obj.masks:
                jobs.append((dict(obj.masks), dict(obj.mattes) or None, base_name))
        elif object_mode == "separate":
            for obj in self._objects:
                if not obj.masks:
                    continue
                safe_name = obj.name.replace(" ", "_")
                jobs.append(
                    (dict(obj.masks), dict(obj.mattes) or None, f"{base_name}_{safe_name}")
                )
        else:
            combined = self._combined_masks()
            if combined:
                jobs.append((combined, self._combined_mattes(), base_name))

        if not jobs:
            self._loading_overlay.hide_message()
            QMessageBox.warning(self, "No masks", "Nothing to export.")
            return

        # Pre-fetch source frames once (union of all job keys).
        all_keys: set[int] = set()
        for m, _alphas, _name in jobs:
            all_keys.update(m.keys())
        frames: dict[int, np.ndarray] = {}
        if fmt != "png" and self._frame_buffer is not None:
            for idx in all_keys:
                f = self._frame_buffer.get(idx)
                if f is not None:
                    frames[idx] = f

        # Apply cleanup plates (subtitles / logos) to exported frames.
        if self._cleanup_plates and frames:
            from src.processing.plates import apply_clean

            patches = self._cleanup_plates
            frames = {idx: apply_clean(f, patches) for idx, f in frames.items()}
            logger.info("Applying cleanup plates to %d export frame(s)", len(frames))

        # AI object removal: the rebuilt frames replace their cleaned
        # originals (frames outside the removal region are untouched).
        if self._removal_store.count:
            frames = {
                idx: self._apply_removed_frame(f, idx) for idx, f in frames.items()
            }
            logger.info("Applying removal result to %d export frame(s)", len(frames))

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
        self._export_progress_dlg.cancel_requested.connect(self._on_cancel_export)
        self._export_started = time.monotonic()
        self._start_next_export_job()

    def _start_next_export_job(self) -> None:
        if not self._export_jobs:
            self._finish_exports()
            return
        self._loading_overlay.hide_message()
        masks, alphas, job_name = self._export_jobs.pop(0)
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
            alphas=alphas,
            profile_for_frame=self._profile_for_frame,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        dlg.show()
        self.setEnabled(False)
        # setEnabled(False) cascades into child widgets, so the dialog - and
        # with it the Cancel button - has to be re-enabled explicitly or the
        # button silently stops responding.
        dlg.setEnabled(True)
        self._status.showMessage(f"Exporting {job_name}...")
        self._export_worker.start()

    def _on_cancel_export(self) -> None:
        """Stop the running export and discard any job still queued."""
        self._export_jobs = []
        if self._export_worker is not None:
            # Unconditional: the worker may still be spinning up, and setting
            # the flag on a finished one is harmless.
            self._export_worker.cancel()
            self._status.showMessage("Cancelling export...")

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
        cancelled = self._export_worker is not None and self._export_worker.cancelled
        if cancelled:
            self._export_results.append(("cancelled", ""))
        elif isinstance(result, Exception):
            self._export_results.append(("error", str(result)))
        else:
            self._export_results.append(("ok", result))
        # Continue to the next queued job, or finish.
        if self._export_jobs and not cancelled:
            self._start_next_export_job()
        else:
            self._finish_exports()

    def _finish_exports(self) -> None:
        self._loading_overlay.hide_message()
        dlg = self._export_progress_dlg
        cancelled = any(kind == "cancelled" for kind, _ in self._export_results)
        if dlg is not None:
            has_error = any(kind == "error" for kind, _ in self._export_results)
            dlg.set_finished(success=not has_error, cancelled=cancelled)
            dlg.close()
            self._export_progress_dlg = None
        self.setEnabled(True)
        errors = [msg for kind, msg in self._export_results if kind == "error"]
        paths = [msg for kind, msg in self._export_results if kind == "ok"]
        if errors:
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{errors[0]}")
            return
        if cancelled:
            self._status.showMessage("Export cancelled")
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

    # ── Fine-tuning data export ───────────────────────────────────────────
    def _on_save_training_data(self) -> None:
        """Export this clip's masks as a SAM 2 fine-tuning dataset."""
        if self._video_reader is None or self._frame_buffer is None:
            QMessageBox.warning(self, "Nothing to export", "Import a video first.")
            return
        masked = [obj for obj in self._objects if obj.masks]
        if not masked:
            QMessageBox.warning(
                self,
                "No masks",
                "Segment or propagate a mask first - the corrected masks "
                "are what the fine-tune learns from.",
            )
            return
        dataset_root = QFileDialog.getExistingDirectory(
            self,
            "Choose the dataset folder (created when missing)",
            "",
        )
        if not dataset_root:
            return

        clip_name = (
            self._project_path.stem
            if self._project_path
            else self._video_reader.path.stem
        )
        try:
            self._loading_overlay.show_message(
                "Exporting training data",
                "Writing frames and corrected masks...",
            )
            self._loading_overlay.set_progress_bar(
                sum(len(obj.masks) for obj in masked)
            )
            self._loading_overlay.enable_cancel(False)
            result = export_clip(
                Path(dataset_root),
                clip_name,
                self._frame_buffer.get,
                [dict(obj.masks) for obj in masked],
                on_progress=lambda cur, tot: (
                    self._loading_overlay.set_progress_value(cur),
                    self._loading_overlay.set_progress(
                        f"Wrote {cur} / {tot} frames"
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Training data export failed: %s", exc)
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        finally:
            self._loading_overlay.enable_cancel(False)
            self._loading_overlay.hide_message()

        summary = dataset_summary(result.dataset_root)
        self._status.showMessage(
            f"Training data saved - {result.frames} frame(s), "
            f"{result.objects} object(s) in {result.clip}"
        )
        QMessageBox.information(
            self,
            "Training Data Saved",
            f"Clip: {result.clip}\n"
            f"Frames exported: {result.frames}\n"
            f"Objects exported: {result.objects}\n\n"
            f"Dataset now holds {summary['clips']} clip(s), "
            f"{summary['frames']} frame(s) total.\n\n"
            "Fine-tune it with tools/sam2_training/anime_amv.yaml "
            "(see docs/FINE-TUNING.md), then drop the resulting checkpoint "
            "into checkpoints/custom/ and pick it in the AI Model list.",
        )
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
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay") and self._loading_overlay.is_active():
            self._loading_overlay.reposition_overlay()

    def closeEvent(self, event) -> None:
        self._history_dirty_timer.stop()
        if self._gpu_monitor_dialog is not None:
            self._gpu_monitor_dialog.close()
        if self._render_queue_dialog is not None:
            self._render_queue_dialog.close()
        # Preserve unsaved work even if the user chooses to discard the project.
        if (
            self._auto_save_enabled
            and self._project_dirty
            and self._video_reader is not None
        ):
            self._perform_autosave()
        if not self._confirm_discard():
            event.ignore()
            return
        if (
            self._propagation_worker is not None
            and self._propagation_worker.isRunning()
        ):
            self._propagation_worker.cancel()
            self._propagation_worker.wait(5000)
        for worker in (
            self._inference_worker,
            self._export_worker,
            self._scene_cut_worker,
            self._post_worker,
            self._cleanup_worker,
        ):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(5000)
        if self._sam_manager is not None:
            # A failed inference can poison the CUDA allocator; closing must
            # never surface a traceback to the user.
            try:
                self._sam_manager.unload()
            except Exception as exc:  # noqa: BLE001
                logger.debug("SAM unload during close failed (%s)", exc)
        if self._frame_buffer is not None:
            self._frame_buffer.close()
        event.accept()
