"""Focused tests for the persistent render queue and cooperative runner."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.export.render_queue import (
    CallableRenderRunner,
    InvalidJobStateError,
    QueueState,
    RenderCancelled,
    RenderControl,
    RenderJob,
    RenderPaused,
    RenderPreset,
    RenderProgress,
    RenderQueue,
    RenderResult,
    execute_render_job,
)


class FailOnceRunner:
    """Fail the first attempt and succeed on the second."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, job, control, on_progress=None):
        """Record a two-frame attempt."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first attempt failed")
        if on_progress is not None:
            on_progress(RenderProgress("Rendering", 2, 2))
        control.checkpoint()
        return RenderResult(job.output_destination / "result.mov", 2)


class PauseFirstAttemptRunner:
    """Pause after the first progress event of the first attempt."""

    def __init__(self, queue: RenderQueue, job_id: str) -> None:
        self.queue = queue
        self.job_id = job_id

    def run(self, job, control, on_progress=None):
        """Pause once, then complete a two-frame retry."""
        for current in range(1, 3):
            control.checkpoint()
            if on_progress is not None:
                on_progress(RenderProgress("Rendering", current, 2))
            if job.attempts == 1:
                self.queue.pause(self.job_id)
        return RenderResult(job.output_destination / "resumed.mov", 2)


class BlockingRunner:
    """Wait cooperatively until the queue cancels the job."""

    def __init__(self) -> None:
        self.started = threading.Event()

    def run(self, job, control, on_progress=None):
        """Block at repeated cancellation checkpoints."""
        self.started.set()
        while True:
            control.checkpoint()


def test_named_presets_jobs_and_progress_persist(tmp_path):
    """Presets, per-job settings, outputs, and completed jobs survive reload."""

    def successful_runner(job, control, on_progress):
        """Render two deterministic frames."""
        for current in range(1, 3):
            control.checkpoint()
            if on_progress is not None:
                on_progress(RenderProgress("Rendering", current, 2))
        return RenderResult(job.output_destination / "cut.mov", 2)

    storage = tmp_path / "queue.json"
    queue = RenderQueue(storage, runner=CallableRenderRunner(successful_runner))
    preset = RenderPreset(
        name="Client Alpha",
        format="h264",
        base_name="client-alpha",
        resolution=(1280, 720),
        include_audio=False,
    )
    queue.save_preset(preset)
    job = queue.enqueue(
        tmp_path / "clip.bbproj",
        tmp_path / "renders" / "clip",
        preset,
        name="Opening shot",
        overrides={"base_name": "opening"},
    )

    assert queue.run_next() is job
    assert job.state is QueueState.COMPLETED
    assert job.attempts == 1
    assert job.progress == 1.0
    assert job.output_path == tmp_path / "renders" / "clip" / "cut.mov"

    reloaded = RenderQueue.load(storage, runner=CallableRenderRunner(successful_runner))
    restored = reloaded.get_job(job.id)
    assert restored.name == "Opening shot"
    assert restored.output_destination == job.output_destination
    assert restored.settings["base_name"] == "opening"
    assert restored.settings["resolution"] == [1280, 720]
    assert restored.state is QueueState.COMPLETED
    assert restored.output_path == job.output_path
    assert reloaded.get_preset("Client Alpha") == preset

    payload = json.loads(storage.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["jobs"][0]["attempts"] == 1
    assert not list(tmp_path.glob(".queue.json.*.tmp"))


def test_failure_retry_and_invalid_terminal_transition(tmp_path):
    """Failed jobs retain errors, increment attempts, and can be retried."""
    runner = FailOnceRunner()
    queue = RenderQueue(tmp_path / "queue.json", runner=runner)
    job = queue.enqueue(tmp_path / "one.bbproj", tmp_path / "out")

    assert queue.run_next() is job
    assert job.state is QueueState.FAILED
    assert job.attempts == 1
    assert job.last_error == "RuntimeError: first attempt failed"

    queue.retry(job.id, autostart=False)
    assert job.state is QueueState.QUEUED
    assert job.last_error is None
    assert job.completed == 0

    assert queue.run_next() is job
    assert job.state is QueueState.COMPLETED
    assert job.attempts == 2
    assert job.last_error is None
    with pytest.raises(InvalidJobStateError):
        queue.retry(job.id, autostart=False)


def test_pause_and_resume_transitions(tmp_path):
    """A cooperative pause persists and a resumed attempt starts fresh."""
    queue = RenderQueue(tmp_path / "queue.json")
    job = queue.enqueue(tmp_path / "clip.bbproj", tmp_path / "out")
    queue._runner = PauseFirstAttemptRunner(queue, job.id)

    assert queue.run_next() is job
    assert job.state is QueueState.PAUSED
    assert job.progress == 0.5
    assert job.attempts == 1

    queue.resume(job.id, autostart=False)
    assert job.state is QueueState.QUEUED
    assert queue.run_next() is job
    assert job.state is QueueState.COMPLETED
    assert job.progress == 1.0
    assert job.attempts == 2


def test_running_job_is_cancelled_and_worker_stops(tmp_path):
    """Cancellation wakes a blocked cooperative runner and persists its state."""
    runner = BlockingRunner()
    queue = RenderQueue(tmp_path / "queue.json", runner=runner)
    job = queue.enqueue(tmp_path / "clip.bbproj", tmp_path / "out")

    assert queue.start() is True
    assert runner.started.wait(2)
    queue.cancel(job.id)
    assert queue.wait(2) is True
    assert queue.wait_for_idle(2) is True
    assert job.state is QueueState.CANCELLED

    reloaded = RenderQueue(tmp_path / "queue.json", runner=runner)
    assert reloaded.get_job(job.id).state is QueueState.CANCELLED


def test_remove_clear_and_queued_pause(tmp_path):
    """Queue management transitions cover paused, queued, and terminal jobs."""
    queue = RenderQueue(tmp_path / "queue.json")
    paused = queue.enqueue(tmp_path / "paused.bbproj", tmp_path / "a")
    cancelled = queue.enqueue(tmp_path / "cancelled.bbproj", tmp_path / "b")
    queued = queue.enqueue(tmp_path / "queued.bbproj", tmp_path / "c")

    queue.pause(paused.id)
    assert queue.remove(paused.id) is paused
    queue.cancel(cancelled.id)
    assert queue.clear(QueueState.CANCELLED) == (cancelled.id,)
    assert queue.jobs == (queued,)

    queue.pause(queued.id)
    assert queue.clear() == (queued.id,)
    assert queue.jobs == ()


def test_interrupted_running_job_is_recovered_as_queued(tmp_path):
    """A process restart does not strand a job in the running state."""
    storage = tmp_path / "queue.json"
    runner = BlockingRunner()
    queue = RenderQueue(storage, runner=runner)
    job = queue.enqueue(tmp_path / "clip.bbproj", tmp_path / "out")
    assert queue.start() is True
    assert runner.started.wait(2)

    recovered = RenderQueue.load(storage, runner=runner)
    restored = recovered.get_job(job.id)
    assert restored.state is QueueState.QUEUED
    assert restored.stage == "Recovered after interruption"

    queue.cancel(job.id)
    assert queue.wait(2) is True


def test_control_cancellation_takes_priority_over_pause():
    """A cancel request wins when pause and cancel are both present."""
    control = RenderControl()
    control.request_pause()
    control.request_cancel()
    control.resume()
    assert control.is_cancelled
    with pytest.raises(RenderCancelled):
        control.checkpoint()


class FakeVideoReader:
    """Video reader double that returns two inexpensive frame sentinels."""

    total_frames = 2
    fps = 24.0

    def __init__(self, path: Path) -> None:
        self.path = path
        self.reads: list[int] = []
        self.closed = False

    def read_frame(self, index: int) -> object:
        """Return a stable per-frame sentinel."""
        self.reads.append(index)
        return f"frame-{index}"

    def close(self) -> None:
        """Mark the reader closed."""
        self.closed = True


def _install_render_module_doubles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[type[FakeVideoReader], type[Any]]:
    """Install optional-dependency doubles for the standalone render helper."""
    (tmp_path / "clip.bbproj").write_text("{}", encoding="utf-8")
    (tmp_path / "source.mp4").write_bytes(b"video")

    class FakeProjectManager:
        """Project loader double with relative video and two masks."""

        @staticmethod
        def load(path: Path) -> SimpleNamespace:
            """Return the minimum project data needed by the helper."""
            return SimpleNamespace(
                video_path="source.mp4",
                video_meta={"fps": 24.0},
                masks={0: object(), 1: object()},
            )

    class ExporterDouble:
        """Exporter double that iterates frames and records all options."""

        latest: ExporterDouble | None = None

        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir
            self.accessed_frames: list[int] = []
            self.options: dict[str, Any] = {}
            ExporterDouble.latest = self

        def export_video(self, frames, masks, fps, **options):
            """Exercise the same per-frame mapping access as Exporter."""
            self.options = {"fps": fps, **options}
            total = len(frames)
            for current, index in enumerate(sorted(frames), start=1):
                frames[index]
                self.accessed_frames.append(index)
                callback = options.get("on_progress")
                if callback is not None:
                    callback(current, total)
            return self.output_dir / "render.mov"

    project_module = ModuleType("src.core.project")
    project_module.ProjectManager = FakeProjectManager
    reader_module = ModuleType("src.processing.video_reader")
    reader_module.VideoReader = FakeVideoReader
    exporter_module = ModuleType("src.export.exporter")
    exporter_module.Exporter = ExporterDouble
    monkeypatch.setitem(sys.modules, "src.core.project", project_module)
    monkeypatch.setitem(sys.modules, "src.processing.video_reader", reader_module)
    monkeypatch.setitem(sys.modules, "src.export.exporter", exporter_module)
    return FakeVideoReader, ExporterDouble


def test_execute_render_job_loads_inputs_and_calls_exporter(
    tmp_path,
    monkeypatch,
):
    """The helper resolves project inputs and reports loading and render stages."""
    _, exporter_type = _install_render_module_doubles(monkeypatch, tmp_path)
    preset = RenderPreset(
        name="Batch h264",
        format="h264",
        base_name="batch",
        include_audio=False,
        background=(0, 0, 0),
    )
    job = RenderJob(
        project_path=tmp_path / "clip.bbproj",
        output_destination=tmp_path / "output",
        preset_name=preset.name,
        settings=preset.to_dict(),
    )
    progress: list[RenderProgress] = []

    result = execute_render_job(
        job,
        on_progress=lambda update: progress.append(update),
    )

    assert result.output_path == tmp_path / "output" / "render.mov"
    assert result.frames_rendered == 2
    assert result.details["video"] == str(tmp_path / "source.mp4")
    assert exporter_type.latest is not None
    assert exporter_type.latest.accessed_frames == [0, 1]
    assert exporter_type.latest.options["fmt"] == "h264"
    assert exporter_type.latest.options["audio_source"] is None
    assert [update.stage for update in progress] == [
        "Loading masks",
        "Loading video",
        "Loading video",
        "Loading video",
        "Preparing export",
        "Rendering video",
        "Rendering video",
        "Completed",
    ]


def test_execute_render_job_pauses_between_exporter_frames(
    tmp_path,
    monkeypatch,
):
    """A pause requested after frame one is observed before frame two."""
    control = RenderControl()

    def pause_after_first(update: RenderProgress) -> None:
        """Request a pause after the first exported frame."""
        if update.stage == "Rendering video" and update.current == 1:
            control.request_pause()

    _, exporter_type = _install_render_module_doubles(monkeypatch, tmp_path)
    job = RenderJob(
        project_path=tmp_path / "clip.bbproj",
        output_destination=tmp_path / "output",
        settings=RenderPreset(base_name="pausable").to_dict(),
    )

    with pytest.raises(RenderPaused):
        execute_render_job(job, control, on_progress=pause_after_first)

    assert exporter_type.latest is not None
    assert exporter_type.latest.accessed_frames == [0]


def test_retry_moves_job_to_the_front_of_the_queue(tmp_path):
    """A retried job is re-queued *ahead* of the jobs already waiting.

    ``dict`` preserves insertion order, so the old pop-and-re-add actually
    moved the job to the end — the opposite of the documented behaviour.
    """
    queue = RenderQueue(tmp_path / "queue.json")
    first = queue.enqueue(tmp_path / "one.bbproj", tmp_path / "out")
    second = queue.enqueue(tmp_path / "two.bbproj", tmp_path / "out")
    third = queue.enqueue(tmp_path / "three.bbproj", tmp_path / "out")

    queue.cancel(third.id)
    queue.retry(third.id, autostart=False)

    assert [job.id for job in queue.jobs] == [third.id, first.id, second.id]


def test_execute_render_job_separate_mode_progress_is_valid(
    tmp_path,
    monkeypatch,
):
    """Separate object mode must report valid progress, not crash the job.

    Each object contributes its own mask set, so the loaded-mask count exceeds
    the number of distinct frames.  Reporting that count against the frame
    total raised ``ValueError("current cannot exceed total")`` and failed the
    whole render.
    """
    (tmp_path / "clip.bbproj").write_text("{}", encoding="utf-8")
    (tmp_path / "source.mp4").write_bytes(b"video")

    class FakeProjectManager:
        """Project double with two objects each masked on both frames."""

        @staticmethod
        def load(path: Path) -> SimpleNamespace:
            obj_a = SimpleNamespace(
                obj_id=1, masks={0: "a0", 1: "a1"}, mattes={}, profile=None
            )
            obj_b = SimpleNamespace(
                obj_id=2, masks={0: "b0", 1: "b1"}, mattes={}, profile=None
            )
            return SimpleNamespace(
                video_path="source.mp4",
                video_meta={"fps": 24.0},
                masks={},
                objects=[obj_a, obj_b],
            )

    class ExporterDouble:
        """Exporter double that iterates the frame mapping."""

        latest: ExporterDouble | None = None

        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir
            self.accessed_frames: list[int] = []
            ExporterDouble.latest = self

        def export_video(self, frames, masks, fps, **options):
            callback = options.get("on_progress")
            total = len(frames)
            for current, index in enumerate(sorted(frames), start=1):
                frames[index]
                self.accessed_frames.append(index)
                if callback is not None:
                    callback(current, total)
            return self.output_dir / "render.mov"

    project_module = ModuleType("src.core.project")
    project_module.ProjectManager = FakeProjectManager
    reader_module = ModuleType("src.processing.video_reader")
    reader_module.VideoReader = FakeVideoReader
    exporter_module = ModuleType("src.export.exporter")
    exporter_module.Exporter = ExporterDouble
    monkeypatch.setitem(sys.modules, "src.core.project", project_module)
    monkeypatch.setitem(sys.modules, "src.processing.video_reader", reader_module)
    monkeypatch.setitem(sys.modules, "src.export.exporter", exporter_module)

    preset = RenderPreset(object_mode="separate", base_name="batch", include_audio=False)
    job = RenderJob(
        project_path=tmp_path / "clip.bbproj",
        output_destination=tmp_path / "output",
        preset_name=preset.name,
        settings=preset.to_dict(),
    )
    progress: list[RenderProgress] = []

    result = execute_render_job(job, on_progress=lambda update: progress.append(update))

    # Two objects x two frames were rendered without a progress OverflowError.
    assert result.frames_rendered == 4
    assert any(update.stage == "Loading masks" for update in progress)
