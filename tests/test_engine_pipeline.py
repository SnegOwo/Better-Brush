"""End-to-end pipeline plumbing with a stubbed SAM 2 manager.

These exercise frame extraction, the auto-subject gate, propagation collection,
streaming export and the ffmpeg-free fallback — everything except the GPU.
"""

from __future__ import annotations

import io
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.engine import auto_subject, pipeline
from src.engine.auto_subject import SubjectSelection
from src.engine.schemas import JobCancelled, JobRequest
from tests.test_integration import create_synthetic_video


class FakeManager:
    """Stands in for SAM2Manager: one centred square, tracked to every frame."""

    is_available = True

    def __init__(self, _recommendation) -> None:
        self.count = 0
        self.frames_dir = None

    def load_image_predictor(self):
        ...

    def load_video_predictor(self):
        ...

    def init_video(self, frames_dir, progress_cb=None):
        self.frames_dir = Path(frames_dir)
        files = sorted(self.frames_dir.glob("*.jpg"))
        self.count = len(files)
        sample = cv2.imread(str(files[0]))
        self.h, self.w = sample.shape[:2]

    def add_mask_to_video(self, frame_idx, obj_id, mask):
        self.key = frame_idx

    def propagate_in_video(
        self,
        start_frame_idx=0,
        max_frames=None,
        reverse=False,
        on_frame=None,
        cancel_event=None,
    ):
        order = list(range(start_frame_idx, self.count))
        if reverse:
            order = list(range(start_frame_idx - 1, -1, -1))
        out = []
        for i in order:
            mask = np.zeros((self.h, self.w), dtype=bool)
            mask[self.h // 4 : 3 * self.h // 4, self.w // 4 : 3 * self.w // 4] = True
            if on_frame is not None:
                on_frame(i, 1, mask)
            out.append((i, 1, mask, np.array([1.0])))
        return out

    def reset_video_state(self):
        ...


def _patch_runtime(monkeypatch, selection=None):
    monkeypatch.setattr(
        "src.engine.model_setup.build_recommendation",
        lambda **kw: SimpleNamespace(
            model_size=SimpleNamespace(value="fake_model")
        ),
    )
    monkeypatch.setattr(
        "src.models.sam2_manager.SAM2Manager", FakeManager
    )
    if selection is None:
        selection = SubjectSelection(
            mask=np.ones((64, 64), dtype=bool),
            confidence=0.91,
            box=(16, 16, 32, 32),
            reasons=["fake"],
        )
    monkeypatch.setattr(auto_subject, "find_subject", lambda *a, **k: selection)
    monkeypatch.setattr(auto_subject, "detect_face_center", lambda frame: None)


@pytest.fixture
def synth_video(tmp_path) -> Path:
    path = tmp_path / "clip.mp4"
    create_synthetic_video(path, frames=10, size=(256, 256))
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("OpenCV could not write the synthetic video")
    return path


def _request(video: Path, out_dir: Path) -> JobRequest:
    return JobRequest.from_dict(
        {
            "input_path": str(video),
            "output_dir": str(out_dir),
            "fps": 10.0,
            "quality": "fast",
        }
    )


def test_png_fallback_when_ffmpeg_missing(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )

    result = pipeline.run_job(_request(synth_video, tmp_path / "out"))

    assert result.output_kind == "png_sequence"
    out = Path(result.output_path)
    assert out.is_dir()
    assert len(list(out.glob("*.png"))) == 10
    assert result.frame_count == 10
    assert result.preview_path is not None
    assert Path(result.preview_path).is_file()
    assert result.qc["state"] == "ok"
    assert result.subject is not None
    assert result.subject.confidence > 0


def test_end_to_end_with_real_exporters(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch)

    result = pipeline.run_job(_request(synth_video, tmp_path / "out"))

    assert result.frame_count == 10
    assert result.frame_in == 0
    assert result.frame_out == 9
    output = Path(result.output_path)
    if result.output_kind == "mov_alpha":
        assert output.is_file()
        assert output.stat().st_size > 0
    else:
        assert output.is_dir()
        assert len(list(output.glob("*.png"))) == 10


def test_no_character_is_reported(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch, selection=SubjectSelection(confidence=0.1))
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.run_job(_request(synth_video, tmp_path / "out"))
    assert exc.value.kind == "no_character"


def test_multiple_subjects_is_reported(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch, selection=SubjectSelection(multiple=True, confidence=0.8))
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.run_job(_request(synth_video, tmp_path / "out"))
    assert exc.value.kind == "multiple_subjects"


def test_missing_input_is_unsupported(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    request = JobRequest.from_dict(
        {"input_path": str(tmp_path / "nope.mov"), "output_dir": str(tmp_path / "out")}
    )
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.run_job(request)
    assert exc.value.kind == "unsupported"


def test_cancel_before_start_raises(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(JobCancelled):
        pipeline.run_job(_request(synth_video, tmp_path / "out"), cancel_event=cancel)


def test_progress_is_reported(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    seen: list[str] = []
    pipeline.run_job(
        _request(synth_video, tmp_path / "out"),
        on_progress=lambda stage, cur, total, msg: seen.append(stage),
    )
    assert "load" in seen
    assert "extract" in seen
    assert "propagate" in seen
    assert "done" in seen


def test_stages_are_reported_in_the_panel_order(monkeypatch, synth_video, tmp_path):
    """The engine must report its stages in the panel's checklist order.

    The panel only ever moves its checklist forwards, so a stage that arrives
    late is marked "done" the moment it appears. "Loading the model" used to be
    emitted *after* "reading the source", so the cold model load showed up as
    "Reading the source" — the exact silence it exists to remove.
    """
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    order = ["load", "extract", "analyze", "propagate", "export", "done"]
    seen: list[str] = []
    pipeline.run_job(
        _request(synth_video, tmp_path / "out"),
        on_progress=lambda stage, cur, total, msg: seen.append(stage),
    )

    positions = [order.index(stage) for stage in seen if stage in order]
    assert positions == sorted(positions), seen
    assert seen[0] == "load"
    assert "done" in seen


def test_the_loaded_model_is_reused_between_jobs(monkeypatch):
    """Reloading a checkpoint costs ~15s a run; the engine serves one job at a
    time, so the second run must not pay for it again."""
    built: list[object] = []

    class Counting(FakeManager):
        is_available = True

        def __init__(self, rec):
            super().__init__(rec)
            built.append(rec)

    monkeypatch.setattr("src.models.sam2_manager.SAM2Manager", Counting)
    first_rec = SimpleNamespace(model_size=SimpleNamespace(value="fake_model"))
    other_rec = SimpleNamespace(model_size=SimpleNamespace(value="other_model"))
    pipeline._MANAGER_CACHE.clear()
    try:
        first = pipeline._manager_for(first_rec)
        assert pipeline._manager_for(first_rec) is first
        assert len(built) == 1

        other = pipeline._manager_for(other_rec)
        assert other is not first
        assert len(built) == 2
        # One model at a time: two checkpoints would not both fit the card.
        assert list(pipeline._MANAGER_CACHE) == [pipeline._manager_key(other_rec)]
    finally:
        pipeline._MANAGER_CACHE.clear()


# --------------------------------------------------------------------- #
# Streaming export: the failure modes that used to be invisible
# --------------------------------------------------------------------- #
def _write_frames(tmp_path: Path, count: int, size: tuple[int, int]) -> Path:
    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), image)
    return frames


def _require_ffmpeg(monkeypatch) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not on PATH")
    monkeypatch.setattr("src.export.exporter._resolve_ffmpeg", lambda: ffmpeg)


def test_ffmpeg_failure_names_the_cause_and_the_exit_code():
    """A broken pipe alone ("[Errno 22] Invalid argument") explains nothing.

    The write error is the symptom; ffmpeg's own stderr is the cause, so it has
    to survive into the message the panel shows.
    """
    proc = SimpleNamespace(
        stderr=io.BytesIO(b"Picture size 0x0 is invalid\n"),
        poll=lambda: 1,
    )
    message = pipeline._ffmpeg_failure(proc, OSError(22, "Invalid argument"), 1)

    assert "Invalid argument" in message
    assert "exit code 1" in message
    assert "Picture size 0x0 is invalid" in message


def test_png_frames_keep_colour_and_alpha(tmp_path):
    """Regression: a whole-array reverse puts a colour channel into alpha.

    The preview and the PNG-sequence fallback were written with
    ``rgba[:, :, ::-1]``, which also moved alpha to the front — so the matte was
    replaced by the red channel (opaque, rotated colours) in the panel preview
    and in what After Effects imported.
    """
    path = tmp_path / "preview.png"
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 200  # red
    rgba[..., 1] = 40  # green
    rgba[..., 2] = 10  # blue
    rgba[..., 3] = 0  # fully transparent

    pipeline._write_rgba_png(path, rgba)

    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert back.shape == (4, 4, 4)
    assert back[:, :, 3].max() == 0, "alpha must survive the write"
    assert back[2, 2].tolist() == [10, 40, 200, 0], "colour (BGR on read) preserved"


def test_movie_export_size_change_is_explained(monkeypatch, tmp_path):
    _require_ffmpeg(monkeypatch)
    frames = _write_frames(tmp_path, 2, (32, 32))

    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline._streaming_mov(
            frames, {}, 2, 24.0, 64, 64, tmp_path / "alpha.mov", None, None
        )

    assert "32x32" in str(exc.value)
    assert "64x64" in str(exc.value)


def test_failed_export_reports_ffmpeg_stderr(monkeypatch, tmp_path):
    _require_ffmpeg(monkeypatch)
    frames = _write_frames(tmp_path, 1, (64, 64))
    blocked = tmp_path / "alpha.mov"
    blocked.mkdir()  # ffmpeg cannot write a movie over a directory

    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline._streaming_mov(
            frames, {}, 1, 24.0, 64, 64, blocked, None, None
        )

    assert "ffmpeg could not write the alpha movie" in str(exc.value)


# --------------------------------------------------------------------- #
# Matting + brush corrections
# --------------------------------------------------------------------- #
def _mask_png_b64(
    shape: tuple[int, int], rect: tuple[int, int, int, int] | None = None
) -> str:
    import base64

    image = np.zeros(shape, dtype=np.uint8)
    if rect is None:
        image[:] = 255
    else:
        x, y, w, h = rect
        image[y : y + h, x : x + w] = 255
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class FakeMattingRunner:
    """A matting model that returns a flat 0.5 alpha for every frame."""

    def __init__(self, model, device="cuda", cancel_event=None) -> None:
        self.model = model
        self.device = device

    def matte(
        self, indices, frame_getter, prompt_masks, settings, on_alpha=None, mask_getter=None
    ):
        out = {}
        for i in indices:
            alpha = np.full(frame_getter(i).shape[:2], 0.5, dtype=np.float32)
            out[i] = alpha
            if on_alpha is not None:
                on_alpha(i, alpha)
        return out

    def unload(self) -> None:
        ...


class BrokenMattingRunner(FakeMattingRunner):
    def matte(self, *args, **kwargs):
        raise RuntimeError("the matting weights are missing")


def _matting_request(video: Path, out_dir: Path) -> JobRequest:
    return JobRequest.from_dict(
        {
            "input_path": str(video),
            "output_dir": str(out_dir),
            "fps": 10.0,
            "quality": "fast",
            "matting": True,
        }
    )


def test_matting_refines_the_alpha_and_reports_its_stage(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    monkeypatch.setattr("src.processing.matting.MattingRunner", FakeMattingRunner)

    seen: list[str] = []
    result = pipeline.run_job(
        _matting_request(synth_video, tmp_path / "out"),
        on_progress=lambda stage, cur, total, msg: seen.append(stage),
    )

    assert result.matting is True
    assert "matte" in seen
    order = ["load", "extract", "analyze", "propagate", "matte", "export", "done"]
    positions = [order.index(s) for s in seen if s in order]
    assert positions == sorted(positions), seen

    # The 0.5 matte — not the outline — produced the alpha: an outline would be
    # 0 in the background and 255 inside, never a flat mid value.
    frame = cv2.imread(
        str(Path(result.output_path) / "00000.png"), cv2.IMREAD_UNCHANGED
    )
    assert frame[:, :, 3].min() == 127
    assert frame[:, :, 3].max() == 127


def test_matting_failure_falls_back_to_the_outline(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    monkeypatch.setattr("src.processing.matting.MattingRunner", BrokenMattingRunner)

    result = pipeline.run_job(_matting_request(synth_video, tmp_path / "out"))

    # The removal still ships; only the soft refinement is dropped.
    assert result.matting is False
    assert result.frame_count == 10
    frame = cv2.imread(
        str(Path(result.output_path) / "00000.png"), cv2.IMREAD_UNCHANGED
    )
    assert frame[:, :, 3].min() == 0
    assert frame[:, :, 3].max() == 255


def test_a_correction_reuses_the_stored_seed(monkeypatch, synth_video, tmp_path):
    """A brush correction must re-track, not search for the subject again."""
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    real = auto_subject.find_subject
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(auto_subject, "find_subject", counting)

    first = pipeline.run_job(_request(synth_video, tmp_path / "out"))
    assert (Path(first.work_dir) / "seed.png").is_file()
    assert (Path(first.work_dir) / "seed.json").is_file()
    assert calls["n"] > 0

    before = calls["n"]
    correction = JobRequest.from_dict(
        {
            "input_path": str(synth_video),
            "output_dir": str(tmp_path / "out2"),
            "fps": 10.0,
            "quality": "fast",
            "resume_dir": first.work_dir,
            # Trim a corner, not the subject: erasing everything is refused.
            "correction_remove": _mask_png_b64((256, 256), rect=(0, 0, 40, 40)),
        }
    )
    second = pipeline.run_job(correction)

    assert calls["n"] == before, "a correction must not run the subject search"
    assert second.frame_count == first.frame_count


def test_a_correction_refuses_to_erase_the_whole_subject(
    monkeypatch, synth_video, tmp_path
):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    first = pipeline.run_job(_request(synth_video, tmp_path / "out"))

    erased = JobRequest.from_dict(
        {
            "input_path": str(synth_video),
            "output_dir": str(tmp_path / "out2"),
            "fps": 10.0,
            "quality": "fast",
            "resume_dir": first.work_dir,
            "correction_remove": _mask_png_b64((256, 256)),
        }
    )
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_job(erased)


def test_a_correction_paints_the_frame_it_targets(monkeypatch, synth_video, tmp_path):
    """The painted frame becomes the seed, so a wrong frame is fixed where seen."""
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    first = pipeline.run_job(_request(synth_video, tmp_path / "a"))

    correction = JobRequest.from_dict(
        {
            "input_path": str(synth_video),
            "output_dir": str(tmp_path / "b"),
            "fps": 10.0,
            "quality": "fast",
            "resume_dir": first.work_dir,
            "correction_frame": 4,
            # A block in the corner: certainly not part of the auto mask.
            "correction_add": _mask_png_b64((256, 256), rect=(5, 5, 30, 30)),
        }
    )
    second = pipeline.run_job(correction)

    assert second.key_idx == 4
    mask = pipeline._mask_at(Path(second.work_dir), 4)
    assert mask is not None
    assert mask[10, 10], "the painted block must survive on the frame it was painted"


def test_fill_holes_closes_an_enclosed_gap():
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:180, 20:180] = True
    mask[80:100, 80:100] = False  # an enclosed hole, like a dark patch of hair

    filled = pipeline._fill_holes(mask)

    assert filled[90, 90], "the enclosed gap must be closed"
    assert not filled[5, 5], "the outside must stay outside"


def test_render_frame_returns_a_png_of_that_frame(monkeypatch, synth_video, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    result = pipeline.run_job(_request(synth_video, tmp_path / "out"))

    png = pipeline.render_frame(result.work_dir, 1)

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert frame.shape == (result.height, result.width, 4)

    with pytest.raises(pipeline.PipelineError):
        pipeline.render_frame(result.work_dir, result.frame_count + 5)


def test_render_frame_is_cached_until_the_masks_change(
    monkeypatch, synth_video, tmp_path
):
    """The brush scrubs back and forth, so a repeat must not re-encode."""
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    result = pipeline.run_job(_request(synth_video, tmp_path / "out"))
    work = Path(result.work_dir)

    first = pipeline.render_frame(result.work_dir, 1)
    reads = []
    real_frame_at = pipeline._frame_at

    def counting_frame_at(work_dir, index):
        reads.append(index)
        return real_frame_at(work_dir, index)

    monkeypatch.setattr(pipeline, "_frame_at", counting_frame_at)
    assert pipeline.render_frame(result.work_dir, 1) == first
    assert reads == [], "a cached frame must not touch the source again"

    masks = work / pipeline._MASKS_FILE
    os.utime(masks, (masks.stat().st_atime, masks.stat().st_mtime + 5))
    pipeline.render_frame(result.work_dir, 1)
    assert reads == [1], "a rewritten mask set invalidates the cache"


def test_render_source_frame_returns_the_untouched_frame(
    monkeypatch, synth_video, tmp_path
):
    """The compare view needs the original, not the cut-out."""
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "src.export.exporter._resolve_ffmpeg", lambda: "not-a-real-ffmpeg"
    )
    result = pipeline.run_job(_request(synth_video, tmp_path / "out"))

    body, media_type = pipeline.render_source_frame(result.work_dir, 1)

    assert media_type == "image/jpeg"
    assert body[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape[:2] == (result.height, result.width)


def test_decoded_frames_stream_in_order_without_keeping_them(
    tmp_path, monkeypatch
):
    """The export prefetches, but it must still hand frames over in order."""
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(6):
        pipeline._write_jpg(
            frames / f"{i:05d}.jpg", np.full((8, 8, 3), i * 10, np.uint8)
        )

    streamed = list(pipeline._decoded_frames(frames, 6, None))

    assert [i for i, _ in streamed] == [0, 1, 2, 3, 4, 5]
    for i, frame in streamed:
        assert frame.shape == (8, 8, 3)
        assert abs(int(frame.mean()) - i * 10) <= 2


def test_apply_correction_adds_then_removes():
    seed = np.zeros((4, 4), dtype=bool)
    seed[1, 1] = True
    add = np.zeros((4, 4), dtype=bool)
    add[2, 2] = True
    remove = np.zeros((4, 4), dtype=bool)
    remove[1, 1] = True

    out = pipeline._apply_correction(seed, add, remove)

    assert out[2, 2]
    assert not out[1, 1]
