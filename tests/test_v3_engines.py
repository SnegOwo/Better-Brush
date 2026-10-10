"""Unit tests for the V3 engines: EfficientTAM plumbing, MiniMax-Remover and
MatAnyone settings/helpers, matte persistence and the soft-alpha export path.

No GPU or model weights are required: everything here is either pure Python or
uses fake inputs. GPU-only code paths are covered by the smoke tests in
tools/ and the integration tests.

Run with:
    python -m pytest tests/test_v3_engines.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.core.project import matte_from_b64, matte_to_b64
from src.export.exporter import Exporter
from src.hardware.profiler import ModelSize, SAM2Profiler
from src.models import checkpoint_manager as ckpt_mod
from src.processing.frame_buffer import CleanFrameStore
from src.processing.matting import (
    MattingSettings,
    _fit_resolution,
    _grow_alpha,
    _install_topk_guard,
)
from src.processing.removal import RemovalSettings
from src.processing.removal import _fit_resolution as _fit_resolution_16


# ── EfficientTAM plumbing ────────────────────────────────────────────────
def test_efficienttam_sizes_exist_with_thresholds_below_tiny() -> None:
    profiler = SAM2Profiler()
    thresholds = profiler._VRAM_THRESHOLDS
    assert thresholds[ModelSize.EFFICIENTTAM_S] < thresholds[ModelSize.TINY]
    assert thresholds[ModelSize.EFFICIENTTAM_TI] < thresholds[ModelSize.EFFICIENTTAM_S]
    assert ModelSize.EFFICIENTTAM_S in profiler._CHECKPOINT_MAP
    assert ModelSize.EFFICIENTTAM_TI in profiler._CHECKPOINT_MAP
    assert "efficient-track-anything" in profiler._url_for(ModelSize.EFFICIENTTAM_S)


def test_checkpoint_manager_downloads_efficienttam_from_hf(tmp_path) -> None:
    manager = ckpt_mod.CheckpointManager(checkpoints_dir=tmp_path)
    url = ckpt_mod._CHECKPOINT_URLS[ModelSize.EFFICIENTTAM_S]
    assert "efficienttam_s_512x512.pt" in url
    assert url.startswith("https://huggingface.co/")
    # The vendored config resolves inside the package, not over HTTP.
    config = manager.get_config_path(ModelSize.EFFICIENTTAM_S)
    assert config.is_file()
    assert "efficienttam_s_512x512.yaml" in config.name


def test_weak_gpu_falls_back_to_light_model() -> None:
    from src.hardware.detector import Backend, GPUInfo, HardwareProfile

    def profile(free: int) -> HardwareProfile:
        return HardwareProfile(
            cpu_name="CPU",
            cpu_count_physical=4,
            cpu_count_logical=8,
            ram_total_mb=16_384,
            ram_available_mb=8_192,
            gpus=[
                GPUInfo(
                    index=0,
                    name="Weak GPU",
                    total_vram_mb=free,
                    available_vram_mb=free,
                    backend=Backend.CUDA,
                    compute_capability=(6, 1),
                )
            ],
            best_backend=Backend.CUDA,
            recommended_device="cuda:0",
        )

    profiler = SAM2Profiler()
    assert (
        profiler.recommend(profile(1_000)).model_size is ModelSize.EFFICIENTTAM_TI
    )
    assert (
        profiler.recommend(profile(2_400)).model_size is ModelSize.EFFICIENTTAM_TI
    )
    assert (
        profiler.recommend(profile(3_000)).model_size is ModelSize.EFFICIENTTAM_S
    )


# ── Removal settings ─────────────────────────────────────────────────────
def test_removal_settings_snaps_and_clamps() -> None:
    settings = RemovalSettings(
        resolution=500, steps=99, mask_grow=999, feather=-5, seed=7
    )
    assert settings.resolution == 480  # nearest offered value
    assert settings.steps == 12
    assert settings.mask_grow == 64
    assert settings.feather == 0
    assert settings.seed == 7
    assert RemovalSettings(resolution=1000).resolution == 1080


def test_removal_fit_resolution_aligns_to_16() -> None:
    h, w = _fit_resolution_16(720, 1280, 480)
    assert h % 16 == 0 and w % 16 == 0
    assert max(h, w) <= 480
    assert h >= 64 and w >= 64


# ── Matting settings and helpers ─────────────────────────────────────────
def test_matting_settings_snaps_and_clamps() -> None:
    settings = MattingSettings(model="nope", resolution=950, gamma=50, grow=-99)
    assert settings.model == "matanyone2"  # unknown -> default
    assert settings.resolution == 1080
    assert settings.gamma == 10.0
    assert settings.grow == -20
    assert MattingSettings(model="matanyone").model == "matanyone"
    assert MattingSettings(resolution=120).resolution == 352  # smallest offered


def test_matting_fit_resolution_keeps_aspect() -> None:
    h, w = _fit_resolution(1080, 1920, 720)
    assert max(h, w) == 720
    assert abs((h / w) - (1080 / 1920)) < 0.05


def test_grow_alpha_grows_and_shrinks_softly() -> None:
    alpha = np.zeros((64, 64), np.float32)
    alpha[24:40, 24:40] = 1.0
    grown = _grow_alpha(alpha, 4)
    shrunk = _grow_alpha(alpha, -4)
    assert grown.sum() > alpha.sum()
    assert shrunk.sum() < alpha.sum()
    assert grown.max() <= 1.0 and grown.min() >= 0.0
    assert np.array_equal(_grow_alpha(alpha, 0), alpha)


def test_topk_guard_clamps_to_available_entries() -> None:
    """top-k over fewer entries than k must clamp instead of raising."""
    import types

    from src.vendor.matanyone2.model.utils import memory_utils

    module = types.SimpleNamespace()
    module.do_softmax = memory_utils.do_softmax
    _install_topk_guard(module)
    similarity = torch.randn(1, 3, 16)  # only 3 memory entries, top_k=30
    affinity, _usage = module.do_softmax(
        similarity, top_k=30, return_usage=True
    )
    # The softmax normalises over the memory entries (dim=1).
    assert torch.allclose(affinity.sum(dim=1), torch.ones(1, 16), atol=1e-4)
    # Idempotent: installing twice does not wrap twice.
    _install_topk_guard(module)


# ── Matte persistence ────────────────────────────────────────────────────
def test_matte_roundtrip_keeps_soft_values() -> None:
    alpha = np.zeros((32, 48), np.float32)
    alpha[4:12, 6:20] = 0.5
    alpha[20:28, 30:44] = 1.0
    restored = matte_from_b64(matte_to_b64(alpha))
    assert restored.shape == alpha.shape
    assert restored.dtype == np.float32
    assert np.allclose(restored, alpha, atol=1 / 255)


def test_project_serializes_mattes(tmp_path) -> None:
    from src.core.project import ObjectRecord, ProjectData, ProjectManager

    alpha = np.zeros((16, 16), np.float32)
    alpha[2:8, 2:8] = 0.75
    data = ProjectData(
        video_path="clip.mp4",
        objects=[
            ObjectRecord(
                obj_id=1,
                name="Object 1",
                color="#00b4f0",
                masks={0: np.zeros((16, 16), bool)},
                mattes={0: alpha},
            )
        ],
    )
    path = tmp_path / "mattes.bbproj"
    ProjectManager.save(path, data)
    loaded = ProjectManager.load(path)
    assert sorted(loaded.objects[0].mattes) == [0]
    assert np.allclose(loaded.objects[0].mattes[0], alpha, atol=1 / 255)


# ── Export uses the soft alpha ───────────────────────────────────────────
def test_exporter_uses_soft_alpha(tmp_path) -> None:
    frame = np.full((32, 48, 3), 90, np.uint8)
    alpha = np.zeros((32, 48), np.float32)
    alpha[8:24, 8:32] = 0.5
    exporter = Exporter(output_dir=tmp_path)
    dest = exporter.export_video(
        {0: frame},
        {0: alpha > 0.5},
        24.0,
        base_name="soft_alpha",
        fmt="h264",
        alphas={0: alpha},
    )
    assert Path(dest).is_file()
    assert Path(dest).suffix == ".mp4"

    png_dir = exporter.export_png_sequence(
        {0: alpha > 0.5}, base_name="soft_alpha_png", alphas={0: alpha}
    )
    from PIL import Image

    saved = np.asarray(Image.open(Path(png_dir) / "soft_alpha_png_000000.png"))
    # The alpha channel must carry the soft coverage, not a hard 0/255.
    assert int(saved[16, 16, 3]) in range(115, 135)
    assert int(saved[0, 0, 3]) == 0


def test_removal_store_roundtrip() -> None:
    store = CleanFrameStore()
    frame = np.full((8, 8, 3), 123, np.uint8)
    store.set(3, frame)
    assert store.contains(3)
    assert store.get(3)[0, 0, 0] == 123
    store.clear()
    assert store.get(3) is None


# ── Close / discard dialog ──────────────────────────────────────────────
def test_close_dialog_offers_save_and_only_saves_when_confirmed(
    tmp_path, monkeypatch
) -> None:
    """Declining to discard must open Save-As, and only a completed save lets
    the pending action (close, new project, open...) continue."""
    from PySide6.QtWidgets import QApplication, QFileDialog

    from src.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._video_reader = _StubVideoReader(tmp_path / "clip.mp4")
    answer = {"choice": "cancel"}
    monkeypatch.setattr(window, "_ask_unsaved_changes", lambda: answer["choice"])
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
    )

    # Clean session: no dialog at all.
    window._project_dirty = False
    assert window._confirm_discard() is True

    # Dirty session: "Discard" continues without saving.
    window._project_dirty = True
    answer["choice"] = "discard"
    assert window._confirm_discard() is True
    assert window._project_dirty is True  # still dirty, just discarded

    # "Cancel" aborts whatever was pending.
    answer["choice"] = "cancel"
    assert window._confirm_discard() is False

    # "Save" opens Save-As; cancelling the file dialog keeps the user put.
    answer["choice"] = "save"
    assert window._confirm_discard() is False
    assert window._project_dirty is True

    # "Save" with a real path writes the project and continues.
    destination = tmp_path / "rescued.bbproj"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(destination), "")),
    )
    assert window._confirm_discard() is True
    assert destination.is_file()
    assert window._project_dirty is False


class _StubVideoReader:
    """Minimal VideoReader stand-in for project-data building."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fps = 24.0
        self.width = 64
        self.height = 48
        self.total_frames = 4


def test_matting_seeds_only_where_the_user_asked(monkeypatch) -> None:
    """Propagated frames must not re-seed the matting tracker.

    Seeding on every masked frame would re-run the clip around each of them
    (quadratic in clip length). Only the first frame, hand-placed prompts
    and shot starts re-seed.
    """
    from PySide6.QtWidgets import QApplication

    from src.core.profiles import ShotProfile
    from src.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._video_reader = _StubVideoReader(Path("clip.mp4"))
    window._frame_buffer = object()

    obj = window._active_object()
    for i in range(5):
        obj.masks[i] = np.zeros((20, 20), bool)
    window._masks = obj.masks
    window._keyframes = set(range(5))
    window._propagated_frames = {0, 1, 3, 4}  # 2 was hand-placed
    window._scene_cuts = [4]

    captured: dict = {}
    monkeypatch.setattr(window, "_loading_overlay", _NoopOverlay(), raising=False)
    monkeypatch.setattr(
        window, "_active_profile", lambda: ShotProfile(matting_model="matanyone2")
    )
    monkeypatch.setattr(window, "_active_object", lambda: obj)

    class FakeMattingWorker:
        progress = _NoopSignal()
        alpha_ready = _NoopSignal()
        finished = _NoopSignal()

        def __init__(self, video_path, indices, prompt_masks, settings, **kwargs):
            captured["indices"] = list(indices)
            captured["prompts"] = sorted(prompt_masks)

        def start(self):
            pass

    import src.ui.main_window as mw

    monkeypatch.setattr(mw, "MattingWorker", FakeMattingWorker)
    window._on_run_matting()
    assert captured["prompts"] == [0, 2, 4]
    assert captured["indices"] == [0, 1, 2, 3, 4]


class _NoopOverlay:
    def show_message(self, *a, **k):
        pass

    def set_progress_bar(self, *a, **k):
        pass

    def set_progress(self, *a, **k):
        pass

    def set_progress_value(self, *a, **k):
        pass

    def enable_cancel(self, *a, **k):
        pass

    def hide_message(self, *a, **k):
        pass

    @property
    def cancel_requested(self):
        return _NoopSignal()


class _NoopSignal:
    def connect(self, *a, **k):
        pass

    def disconnect(self, *a, **k):
        pass
