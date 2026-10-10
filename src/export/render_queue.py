"""Persistent, framework-independent batch rendering for Better Brush projects.

The queue and its runner protocol use only the Python standard library at import
time. Project, video, NumPy, and exporter dependencies are imported lazily by
:func:`execute_render_job` when a real render is requested.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeAlias
from uuid import uuid4

import numpy as np

__all__ = [
    "BatchRenderQueue",
    "CallableRenderRunner",
    "InvalidJobStateError",
    "JobNotFoundError",
    "ProjectRenderRunner",
    "QueueJob",
    "QueueState",
    "RenderCancelled",
    "RenderControl",
    "RenderJob",
    "RenderPaused",
    "RenderPreset",
    "RenderProgress",
    "RenderQueue",
    "RenderQueueError",
    "RenderQueueManager",
    "RenderResult",
    "RenderRunner",
    "execute_render_job",
]

PathLike: TypeAlias = str | os.PathLike[str]
ProgressCallback: TypeAlias = Callable[["RenderProgress"], None]
_QUEUE_VERSION = 1


class RenderQueueError(RuntimeError):
    """Base error for queue configuration and persistence failures."""


class JobNotFoundError(LookupError):
    """Raised when a queue job identifier does not exist."""


class InvalidJobStateError(RenderQueueError):
    """Raised when a requested job transition is not allowed."""


class RenderPaused(Exception):
    """Raised at a render checkpoint when a job has been paused."""


class RenderCancelled(Exception):
    """Raised at a render checkpoint when a job has been cancelled."""


class QueueState(str, Enum):
    """Persistent lifecycle states for a render job."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RenderProgress:
    """Immutable progress update emitted by a runner."""

    stage: str
    current: int
    total: int

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str):
            raise TypeError("stage must be a string")
        if isinstance(self.current, bool) or not isinstance(self.current, int):
            raise TypeError("current must be an integer")
        if isinstance(self.total, bool) or not isinstance(self.total, int):
            raise TypeError("total must be an integer")
        if self.current < 0 or self.total < 0:
            raise ValueError("progress values cannot be negative")
        if self.total and self.current > self.total:
            raise ValueError("current cannot exceed total")

    @property
    def fraction(self) -> float:
        """Return progress as a value from zero to one."""
        return self.current / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Output reported by a successful runner invocation."""

    output_path: Path | None = None
    frames_rendered: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))
        if isinstance(self.frames_rendered, bool) or not isinstance(
            self.frames_rendered, int
        ):
            raise TypeError("frames_rendered must be an integer")
        if self.frames_rendered < 0:
            raise ValueError("frames_rendered cannot be negative")
        object.__setattr__(self, "details", dict(self.details))


@dataclass(frozen=True, slots=True)
class RenderPreset:
    """Named, serializable options understood by :class:`ProjectRenderRunner`."""

    name: str = "Default"
    output_type: str = "video"
    format: str = "prores"
    base_name: str = "render"
    fps: float | None = None
    resolution: tuple[int, int] | None = None
    include_audio: bool = True
    background: tuple[int, int, int] | None = None
    watermark: bool | None = None
    object_mode: str = "combined"
    object_id: int | None = None
    range_mode: str = "all"
    start_frame: int = 0
    end_frame: int = 0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("preset name cannot be empty")
        output_type = str(self.output_type).strip().lower().replace("-", "_")
        if output_type in {"png", "png_sequence", "image_sequence"}:
            output_type = "png"
        if output_type != "video":
            raise ValueError("output_type must be 'video' or 'png'")
        output_format = str(self.format).strip().lower()
        if output_type == "video" and output_format not in {
            "prores",
            "qtrle",
            "h264",
        }:
            raise ValueError("format must be 'prores', 'qtrle', or 'h264'")
        base_name = str(self.base_name).strip()
        if not base_name:
            raise ValueError("base_name cannot be empty")
        if self.fps is not None:
            fps = float(self.fps)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("fps must be a positive finite number")
            object.__setattr__(self, "fps", fps)
        if self.resolution is not None:
            resolution = tuple(int(value) for value in self.resolution)
            if len(resolution) != 2 or any(value <= 0 for value in resolution):
                raise ValueError("resolution must contain two positive integers")
            object.__setattr__(self, "resolution", resolution)
        if not isinstance(self.include_audio, bool):
            raise TypeError("include_audio must be a boolean")
        if self.background is not None:
            background = tuple(int(value) for value in self.background)
            if len(background) != 3 or any(
                value < 0 or value > 255 for value in background
            ):
                raise ValueError("background must contain three values from 0 to 255")
            object.__setattr__(self, "background", background)
        if self.watermark is not None and not isinstance(self.watermark, bool):
            raise TypeError("watermark must be a boolean or None")
        object_mode = str(self.object_mode).strip().lower()
        if object_mode not in {"combined", "single", "separate"}:
            raise ValueError("object_mode must be combined, single, or separate")
        if self.object_id is not None:
            if isinstance(self.object_id, bool) or int(self.object_id) < 1:
                raise ValueError("object_id must be a positive integer")
            object_id = int(self.object_id)
        else:
            object_id = None
        range_mode = str(self.range_mode).strip().lower()
        if range_mode not in {"all", "range"}:
            raise ValueError("range_mode must be all or range")
        start_frame = int(self.start_frame)
        end_frame = int(self.end_frame)
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("frame range is invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "output_type", output_type)
        object.__setattr__(self, "format", output_format)
        object.__setattr__(self, "base_name", base_name)
        object.__setattr__(self, "object_mode", object_mode)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "range_mode", range_mode)
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "end_frame", end_frame)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the preset."""
        return {
            "name": self.name,
            "output_type": self.output_type,
            "format": self.format,
            "base_name": self.base_name,
            "fps": self.fps,
            "resolution": list(self.resolution) if self.resolution else None,
            "include_audio": self.include_audio,
            "background": list(self.background) if self.background else None,
            "watermark": self.watermark,
            "object_mode": self.object_mode,
            "object_id": self.object_id,
            "range_mode": self.range_mode,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenderPreset:
        """Build a preset from persisted or user-provided values."""
        if not isinstance(data, Mapping):
            raise TypeError("preset data must be a mapping")
        values = dict(data)
        if "kind" in values:
            values["output_type"] = values.pop("kind")
        known = {
            "name",
            "output_type",
            "format",
            "base_name",
            "fps",
            "resolution",
            "include_audio",
            "background",
            "watermark",
            "object_mode",
            "object_id",
            "range_mode",
            "start_frame",
            "end_frame",
        }
        values = {key: value for key, value in values.items() if key in known}
        return cls(**values)


@dataclass(slots=True)
class RenderJob:
    """A persistent queue item with independent output and preset settings."""

    project_path: Path
    output_destination: Path
    name: str = ""
    video_path: Path | None = None
    preset_name: str = "Default"
    settings: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    state: QueueState = QueueState.QUEUED
    stage: str = "Waiting"
    completed: int = 0
    total: int = 0
    attempts: int = 0
    last_error: str | None = None
    output_path: Path | None = None
    frames_rendered: int = 0
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        self.project_path = _normalized_path(self.project_path)
        self.output_destination = _normalized_path(self.output_destination)
        if self.video_path is not None:
            self.video_path = _normalized_path(self.video_path)
        if self.output_path is not None:
            self.output_path = _normalized_path(self.output_path)
        self.name = str(self.name).strip() or self.project_path.stem
        self.preset_name = str(self.preset_name).strip()
        if not self.preset_name:
            raise ValueError("preset_name cannot be empty")
        self.settings = _json_mapping(self.settings, "job settings")
        self.state = QueueState(self.state)
        self.completed = _non_negative_int(self.completed, "completed")
        self.total = _non_negative_int(self.total, "total")
        if self.total and self.completed > self.total:
            self.completed = self.total
        self.attempts = _non_negative_int(self.attempts, "attempts")
        self.frames_rendered = _non_negative_int(
            self.frames_rendered, "frames_rendered"
        )

    @property
    def progress(self) -> float:
        """Return completion as a value from zero to one."""
        return self.completed / self.total if self.total else 0.0

    @property
    def is_terminal(self) -> bool:
        """Return whether the job has reached a terminal state."""
        return self.state in {
            QueueState.COMPLETED,
            QueueState.FAILED,
            QueueState.CANCELLED,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the job."""
        return {
            "id": self.id,
            "name": self.name,
            "project_path": str(self.project_path),
            "video_path": str(self.video_path) if self.video_path else None,
            "output_destination": str(self.output_destination),
            "preset_name": self.preset_name,
            "settings": self.settings,
            "state": self.state.value,
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "output_path": str(self.output_path) if self.output_path else None,
            "frames_rendered": self.frames_rendered,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenderJob:
        """Reconstruct a job from its persistent representation."""
        if not isinstance(data, Mapping):
            raise TypeError("job data must be a mapping")
        try:
            project_path = Path(str(data["project_path"]))
            output_destination = Path(str(data["output_destination"]))
        except KeyError as exc:
            raise RenderQueueError(f"Job is missing {exc.args[0]}") from exc
        return cls(
            project_path=project_path,
            output_destination=output_destination,
            name=str(data.get("name", "")),
            video_path=(
                Path(str(data["video_path"])) if data.get("video_path") else None
            ),
            preset_name=str(data.get("preset_name", "Default")),
            settings=dict(data.get("settings") or {}),
            id=str(data.get("id") or uuid4().hex),
            state=QueueState(str(data.get("state", QueueState.QUEUED.value))),
            stage=str(data.get("stage", "Waiting")),
            completed=int(data.get("completed", 0)),
            total=int(data.get("total", 0)),
            attempts=int(data.get("attempts", 0)),
            last_error=(
                str(data["last_error"]) if data.get("last_error") is not None else None
            ),
            output_path=(
                Path(str(data["output_path"])) if data.get("output_path") else None
            ),
            frames_rendered=int(data.get("frames_rendered", 0)),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            started_at=(
                str(data["started_at"]) if data.get("started_at") is not None else None
            ),
            finished_at=(
                str(data["finished_at"])
                if data.get("finished_at") is not None
                else None
            ),
        )


class RenderControl:
    """Thread-safe cooperative pause and cancellation signal for one attempt."""

    def __init__(self) -> None:
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._cancel_event = threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        """Return the event expected by existing exporter cancellation APIs."""
        return self._cancel_event

    @property
    def is_paused(self) -> bool:
        """Return whether a pause has been requested."""
        return not self._resume_event.is_set()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancel_event.is_set()

    def request_pause(self) -> None:
        """Request a pause at the next cooperative checkpoint."""
        self._resume_event.clear()

    def request_cancel(self) -> None:
        """Request cancellation and wake any paused execution immediately."""
        self._cancel_event.set()
        self._resume_event.set()

    def resume(self) -> None:
        """Clear a pause request without clearing cancellation."""
        if not self._cancel_event.is_set():
            self._resume_event.set()

    def checkpoint(self) -> None:
        """Raise :class:`RenderCancelled` or :class:`RenderPaused` if requested."""
        if self._cancel_event.is_set():
            raise RenderCancelled("Render cancelled")
        if not self._resume_event.is_set():
            raise RenderPaused("Render paused")


class RenderRunner(Protocol):
    """Injectable runner interface used by :class:`RenderQueue`."""

    def run(
        self,
        job: RenderJob,
        control: RenderControl,
        on_progress: ProgressCallback | None = None,
    ) -> RenderResult:
        """Execute one attempt for *job* and return its successful output."""
        ...


class CallableRenderRunner:
    """Adapt a three-argument callable to the :class:`RenderRunner` protocol."""

    def __init__(
        self,
        handler: Callable[
            [RenderJob, RenderControl, ProgressCallback | None],
            RenderResult | PathLike | None,
        ],
    ) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handler = handler

    def run(
        self,
        job: RenderJob,
        control: RenderControl,
        on_progress: ProgressCallback | None = None,
    ) -> RenderResult:
        """Invoke the configured callable and normalize its return value."""
        value = self._handler(job, control, on_progress)
        if value is None:
            return RenderResult(frames_rendered=job.completed)
        if isinstance(value, RenderResult):
            return value
        if isinstance(value, (str, os.PathLike)):
            return RenderResult(output_path=Path(value), frames_rendered=job.completed)
        raise TypeError("runner must return RenderResult, a path, or None")


class _ControlledFrameMap(dict[int, Any]):
    """Dictionary that checks cooperative control before frame access."""

    def __init__(self, values: Mapping[int, Any], control: RenderControl) -> None:
        super().__init__(values)
        self._control = control

    def __getitem__(self, key: int) -> Any:
        """Return a frame after checking pause and cancellation state."""
        self._control.checkpoint()
        return super().__getitem__(key)


class ProjectRenderRunner:
    """Load a Better Brush project and delegate output to the existing Exporter."""

    def run(
        self,
        job: RenderJob,
        control: RenderControl,
        on_progress: ProgressCallback | None = None,
    ) -> RenderResult:
        """Run a real project render with lazy optional dependencies."""
        return execute_render_job(job, control, on_progress)


def execute_render_job(
    job: RenderJob,
    control: RenderControl | None = None,
    on_progress: ProgressCallback | None = None,
) -> RenderResult:
    """Load, render, and export one job without requiring Qt.

    Video and mask arrays are loaded first because the current Exporter API
    accepts frame mappings. During Exporter's per-frame loop, a controlled
    mapping checks pause and cancellation before each frame is accessed.
    Cancellation is also supplied through its native event interface.
    """
    from src.core.profiles import ShotProfileMap
    from src.core.project import ProjectManager
    from src.export.exporter import Exporter
    from src.processing.despill import apply_shot_transform
    from src.processing.video_reader import VideoReader

    active_control = control or RenderControl()

    def report(stage: str, current: int, total: int) -> None:
        active_control.checkpoint()
        if on_progress is not None:
            on_progress(RenderProgress(stage, current, total))

    active_control.checkpoint()
    project_path = _normalized_path(job.project_path)
    if project_path.suffix.lower() != ".bbproj":
        raise ValueError(f"Not a .bbproj file: {project_path}")
    if not project_path.is_file():
        raise FileNotFoundError(f"Project not found: {project_path}")
    project = ProjectManager.load(project_path)
    if job.video_path is not None:
        video_path = _resolve_project_relative(job.video_path, project_path.parent)
    elif project.video_path:
        video_path = _resolve_project_relative(project.video_path, project_path.parent)
    else:
        raise ValueError("The project does not contain a video path")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    preset = RenderPreset.from_dict(job.settings)
    mask_sets: list[tuple[dict[int, Any], dict[int, Any], str, int | None]] = []
    records = list(getattr(project, "objects", []) or [])
    if preset.object_mode == "single" and preset.object_id is not None:
        selected = next(
            (record for record in records if record.obj_id == preset.object_id), None
        )
        if selected is None:
            raise ValueError(f"Object {preset.object_id} is not present in the project")
        mask_sets.append(
            (
                dict(selected.masks),
                dict(getattr(selected, "mattes", {}) or {}),
                preset.base_name,
                selected.obj_id,
            )
        )
    elif preset.object_mode == "separate" and records:
        for record in records:
            if record.masks:
                mask_sets.append(
                    (
                        dict(record.masks),
                        dict(getattr(record, "mattes", {}) or {}),
                        f"{preset.base_name}_{record.obj_id}",
                        record.obj_id,
                    )
                )
    elif records:
        combined: dict[int, Any] = {}
        combined_mattes: dict[int, Any] = {}
        for record in records:
            for frame_index, mask in record.masks.items():
                if frame_index in combined:
                    combined[frame_index] = combined[frame_index] | mask
                else:
                    combined[frame_index] = mask
            for frame_index, alpha in (getattr(record, "mattes", {}) or {}).items():
                if frame_index in combined_mattes:
                    combined_mattes[frame_index] = np.maximum(
                        combined_mattes[frame_index], alpha
                    )
                else:
                    combined_mattes[frame_index] = np.asarray(alpha, dtype=np.float32)
        if combined:
            mask_sets.append((combined, combined_mattes, preset.base_name, None))
    else:
        mask_sets.append((dict(project.masks), {}, preset.base_name, None))
    if not mask_sets:
        raise ValueError("The project does not contain any masks")
    if preset.range_mode == "range":
        mask_sets = [
            (
                {
                    index: mask
                    for index, mask in masks.items()
                    if preset.start_frame <= index <= preset.end_frame
                },
                {
                    index: alpha
                    for index, alpha in alphas.items()
                    if preset.start_frame <= index <= preset.end_frame
                },
                name,
                object_id,
            )
            for masks, alphas, name, object_id in mask_sets
        ]
        mask_sets = [entry for entry in mask_sets if entry[0]]
    if not mask_sets:
        raise ValueError("The selected frame range has no masks")

    # Masks are stored un-transformed so the values stay adjustable in the GUI,
    # so the per-shot transforms are applied at render time - once the frames
    # are decoded, because the spill cleaner has to measure the actual pixels.
    profiles = ShotProfileMap(
        default=getattr(project, "default_profile", None),
        records=list(getattr(project, "shot_profiles", []) or []),
    )
    object_profiles = {
        record.obj_id: record.profile
        for record in records
        if record.profile is not None
    }
    highest = max((index for masks, _a, _n, _o in mask_sets for index in masks), default=0)
    total_frames = int(
        (getattr(project, "video_meta", None) or {}).get("total_frames", 0) or 0
    )
    total_frames = total_frames or highest + 1
    scene_cuts = list(getattr(project, "scene_cuts", []) or [])
    output_destination = _normalized_path(job.output_destination)
    output_destination.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(video_path)
    try:
        source_total = int(reader.total_frames)
        fps = float(reader.fps)
        if source_total <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        frame_indices = sorted({index for masks, _a, _n, _o in mask_sets for index in masks})
        total = len(frame_indices)
        if total <= 0:
            raise ValueError("The project has no masked frames")
        report("Loading masks", sum(len(masks) for masks, _a, _n, _o in mask_sets), total)
        report("Loading video", 0, total)
        frames: dict[int, Any] = {}
        for position, index in enumerate(frame_indices, start=1):
            active_control.checkpoint()
            frame = reader.read_frame(index)
            if frame is None:
                raise RuntimeError(f"Failed to read video frame {index}")
            frames[index] = frame
            report("Loading video", position, total)
            active_control.checkpoint()
    finally:
        reader.close()
    # The spill cleaner measures real pixels, so the transforms run here - after
    # the frames are decoded - through the same helper the main export uses, so
    # a queued render always matches the canvas preview.
    transformed: list[tuple[dict[int, Any], dict[int, Any], str, Any]] = []
    edge_softness = 0
    for masks, alphas, name, object_id in mask_sets:
        rendered: dict[int, Any] = {}
        rendered_alphas: dict[int, Any] = {}
        for index, mask in masks.items():
            soft = alphas.get(index)
            if soft is not None:
                # A learned soft matte keeps its own edge falloff; the binary
                # inset/spill transforms do not apply to it.
                alpha = np.clip(np.asarray(soft, dtype=np.float32), 0.0, 1.0)
                rendered_alphas[index] = alpha
                rendered[index] = alpha > 0.5
                continue
            profile = profiles.get_for_frame(
                index,
                total_frames,
                scene_cuts,
                object_id,
                object_profiles.get(object_id),
            )
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
            edge_softness = max(edge_softness, profile.edge_softness)
            if frame is not None:
                frames[index] = frame
            rendered[index] = result
        transformed.append((rendered, rendered_alphas, name, object_id))
    mask_sets = transformed
    exporter = Exporter(output_dir=output_destination)
    report("Preparing export", 0, total)
    output_paths: list[Path] = []
    frames_rendered = 0
    if preset.output_type == "png":
        for masks, alphas, name, _ in mask_sets:
            controlled_masks = _ControlledFrameMap(masks, active_control)
            result_path = exporter.export_png_sequence(
                controlled_masks,
                base_name=name,
                on_progress=lambda current, count: _report_export_frame(
                    active_control,
                    on_progress,
                    "Rendering PNG sequence",
                    current,
                    count,
                ),
                cancel_event=active_control.cancel_event,
                alphas=alphas or None,
            )
            output_paths.append(Path(result_path))
            frames_rendered += len(masks)
    else:
        export_fps = preset.fps
        if export_fps is None:
            metadata = project.video_meta or {}
            export_fps = float(metadata.get("fps", metadata.get("frame_rate", fps)))
        controlled_frames = _ControlledFrameMap(frames, active_control)
        for masks, alphas, name, _ in mask_sets:
            result_path = exporter.export_video(
                controlled_frames,
                masks,
                export_fps,
                base_name=name,
                fmt=preset.format,
                resolution=preset.resolution,
                on_progress=lambda current, count: _report_export_frame(
                    active_control, on_progress, "Rendering video", current, count
                ),
                cancel_event=active_control.cancel_event,
                audio_source=video_path if preset.include_audio else None,
                background=preset.background,
                edge_softness=edge_softness,
                watermark=preset.watermark,
                alphas=alphas or None,
            )
            output_paths.append(Path(result_path))
            frames_rendered += len(masks)
    active_control.checkpoint()
    report("Completed", frames_rendered, frames_rendered)
    return RenderResult(
        output_path=output_paths[0],
        frames_rendered=frames_rendered,
        details={
            "project": str(project_path),
            "video": str(video_path),
            "preset": preset.name,
            "outputs": [str(path) for path in output_paths],
        },
    )


def _report_export_frame(
    control: RenderControl,
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
) -> None:
    control.checkpoint()
    if callback is not None:
        callback(RenderProgress(stage, int(current), int(total)))


class RenderQueue:
    """Thread-safe persistent FIFO queue for sequential render jobs."""

    def __init__(
        self,
        storage_path: PathLike | None = None,
        runner: RenderRunner | None = None,
    ) -> None:
        if storage_path is None:
            from src.core.project import ProjectManager

            storage_path = ProjectManager.settings_dir() / "render_queue.json"
        self._storage_path = _normalized_path(storage_path)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._runner: RenderRunner = runner or ProjectRenderRunner()
        self._presets: dict[str, RenderPreset] = {}
        self._jobs: dict[str, RenderJob] = {}
        self._current_id: str | None = None
        self._current_control: RenderControl | None = None
        self._thread: threading.Thread | None = None
        self._load()

    @property
    def storage_path(self) -> Path:
        """Return the JSON file used for durable queue state."""
        return self._storage_path

    @property
    def jobs(self) -> tuple[RenderJob, ...]:
        """Return jobs in FIFO insertion order."""
        with self._lock:
            return tuple(self._jobs.values())

    @property
    def presets(self) -> dict[str, RenderPreset]:
        """Return a copy of all named presets."""
        with self._lock:
            return dict(self._presets)

    @classmethod
    def load(
        cls,
        storage_path: PathLike,
        runner: RenderRunner | None = None,
    ) -> RenderQueue:
        """Construct a queue from an existing atomic JSON file."""
        return cls(storage_path, runner=runner)

    def save_preset(
        self,
        preset: RenderPreset,
        *,
        overwrite: bool = False,
    ) -> RenderPreset:
        """Persist a named preset, optionally replacing an existing name."""
        if not isinstance(preset, RenderPreset):
            raise TypeError("preset must be a RenderPreset")
        with self._lock:
            if preset.name in self._presets and not overwrite:
                raise RenderQueueError(f"Preset already exists: {preset.name}")
            self._presets[preset.name] = preset
            self._persist_locked()
        return preset

    def delete_preset(self, name: str) -> RenderPreset:
        """Delete and return a named preset without changing existing jobs."""
        with self._lock:
            try:
                preset = self._presets.pop(str(name))
            except KeyError as exc:
                raise JobNotFoundError(f"Unknown preset: {name}") from exc
            self._persist_locked()
        return preset

    def get_preset(self, name: str) -> RenderPreset:
        """Return a named preset."""
        with self._lock:
            try:
                return self._presets[str(name)]
            except KeyError as exc:
                raise JobNotFoundError(f"Unknown preset: {name}") from exc

    def enqueue(
        self,
        project_path: PathLike,
        output_destination: PathLike,
        preset: str | RenderPreset | Mapping[str, Any] = "Default",
        *,
        name: str | None = None,
        video_path: PathLike | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> RenderJob:
        """Add a job with a snapshot of a named preset and independent output."""
        with self._lock:
            if isinstance(preset, str):
                selected = self._presets.get(preset)
                if selected is None:
                    raise JobNotFoundError(f"Unknown preset: {preset}")
            elif isinstance(preset, RenderPreset):
                selected = preset
                if selected.name not in self._presets:
                    self._presets[selected.name] = selected
            elif isinstance(preset, Mapping):
                selected = RenderPreset.from_dict(preset)
                if selected.name not in self._presets:
                    self._presets[selected.name] = selected
            else:
                raise TypeError("preset must be a name, RenderPreset, or mapping")
            values = selected.to_dict()
            values.update(_json_mapping(overrides or {}, "preset overrides"))
            values["name"] = selected.name
            settings = RenderPreset.from_dict(values).to_dict()
            job = RenderJob(
                project_path=_normalized_path(project_path),
                output_destination=_normalized_path(output_destination),
                name=name or _normalized_path(project_path).stem,
                video_path=(
                    _normalized_path(video_path) if video_path is not None else None
                ),
                preset_name=selected.name,
                settings=settings,
            )
            self._jobs[job.id] = job
            self._persist_locked()
            self._condition.notify_all()
        return job

    def get_job(self, job_id: str) -> RenderJob:
        """Return a job by identifier."""
        with self._lock:
            return self._require_job_locked(job_id)

    def jobs_in_state(self, state: QueueState | str) -> tuple[RenderJob, ...]:
        """Return all jobs currently in a requested state."""
        selected = QueueState(state)
        with self._lock:
            return tuple(job for job in self._jobs.values() if job.state == selected)

    def update_progress(
        self,
        job_id: str,
        current: int,
        total: int,
        stage: str | None = None,
    ) -> RenderJob:
        """Record durable progress for a running job."""
        progress = RenderProgress(stage or "Rendering", current, total)
        with self._lock:
            job = self._require_job_locked(job_id)
            if self._current_id != job.id or job.state != QueueState.RUNNING:
                raise InvalidJobStateError(
                    "Only the active running job can report progress"
                )
            job.completed = progress.current
            job.total = progress.total
            job.stage = progress.stage
            job.updated_at = _utc_now()
            self._persist_locked()
        return job

    def start(self) -> bool:
        """Start a background worker and return whether one was launched."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._next_queued_locked() is None:
                return False
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="BetterBrushRenderQueue",
                daemon=True,
            )
            self._thread.start()
            return True

    def run_next(self) -> RenderJob | None:
        """Run the oldest queued job synchronously for deterministic testing."""
        with self._lock:
            if self._current_id is not None:
                raise RenderQueueError("A render job is already active")
            job = self._next_queued_locked()
            if job is None:
                return None
            control = self._claim_locked(job)
        self._execute_claimed(job, control)
        return job

    def run_pending(self) -> tuple[RenderJob, ...]:
        """Synchronously run queued jobs in order until none remain."""
        completed_attempts: list[RenderJob] = []
        while True:
            job = self.run_next()
            if job is None:
                return tuple(completed_attempts)
            completed_attempts.append(job)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the background worker and return whether it stopped."""
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait until no queue worker or render attempt is active."""
        with self._condition:
            return self._condition.wait_for(
                lambda: (
                    self._current_id is None
                    and (self._thread is None or not self._thread.is_alive())
                ),
                timeout=timeout,
            )

    def pause(self, job_id: str) -> RenderJob:
        """Pause a queued or running job at its next cooperative checkpoint."""
        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state == QueueState.PAUSED:
                return job
            if job.state not in {QueueState.QUEUED, QueueState.RUNNING}:
                raise InvalidJobStateError(f"Cannot pause a {job.state.value} job")
            job.state = QueueState.PAUSED
            job.stage = "Paused"
            job.updated_at = _utc_now()
            if self._current_id == job.id and self._current_control is not None:
                self._current_control.request_pause()
            self._persist_locked()
        return job

    def resume(
        self,
        job_id: str,
        *,
        autostart: bool = True,
    ) -> RenderJob:
        """Resume a paused job, optionally starting the background worker."""
        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state != QueueState.PAUSED:
                raise InvalidJobStateError(f"Cannot resume a {job.state.value} job")
            job.state = (
                QueueState.RUNNING if self._current_id == job.id else QueueState.QUEUED
            )
            job.stage = "Resuming" if job.state == QueueState.RUNNING else "Waiting"
            job.last_error = None
            job.updated_at = _utc_now()
            if self._current_id == job.id and self._current_control is not None:
                self._current_control.resume()
            self._persist_locked()
        if autostart and job.state == QueueState.QUEUED:
            self.start()
        return job

    def cancel(self, job_id: str) -> RenderJob:
        """Cancel a queued, running, or paused job."""
        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state == QueueState.CANCELLED:
                return job
            if job.state not in {
                QueueState.QUEUED,
                QueueState.RUNNING,
                QueueState.PAUSED,
            }:
                raise InvalidJobStateError(f"Cannot cancel a {job.state.value} job")
            job.state = QueueState.CANCELLED
            job.stage = "Cancelled"
            job.finished_at = _utc_now()
            job.updated_at = _utc_now()
            if self._current_id == job.id and self._current_control is not None:
                self._current_control.request_cancel()
            self._persist_locked()
            self._condition.notify_all()
        return job

    def retry(self, job_id: str, *, autostart: bool = True) -> RenderJob:
        """Move a failed or cancelled job back to the front of the queue."""
        with self._lock:
            job = self._require_job_locked(job_id)
            if self._current_id == job.id:
                raise InvalidJobStateError("Cannot retry the active job")
            if job.state not in {QueueState.FAILED, QueueState.CANCELLED}:
                raise InvalidJobStateError(f"Cannot retry a {job.state.value} job")
            job.state = QueueState.QUEUED
            job.stage = "Waiting for retry"
            job.completed = 0
            job.total = 0
            job.last_error = None
            job.output_path = None
            job.frames_rendered = 0
            job.started_at = None
            job.finished_at = None
            job.updated_at = _utc_now()
            self._jobs.pop(job.id)
            self._jobs[job.id] = job
            self._persist_locked()
            self._condition.notify_all()
        if autostart:
            self.start()
        return job

    def remove(self, job_id: str) -> RenderJob:
        """Remove a non-running job and return it."""
        with self._lock:
            job = self._require_job_locked(job_id)
            if self._current_id == job.id or job.state == QueueState.RUNNING:
                raise InvalidJobStateError("Cancel the active job before removing it")
            del self._jobs[job.id]
            self._persist_locked()
            self._condition.notify_all()
        return job

    def clear(
        self,
        state: QueueState | str | Iterable[QueueState | str] | None = None,
    ) -> tuple[str, ...]:
        """Remove matching non-running jobs and return their identifiers."""
        with self._lock:
            if state is None:
                selected = set(QueueState) - {QueueState.RUNNING}
            elif isinstance(state, (QueueState, str)):
                selected = {QueueState(state)}
            else:
                selected = {QueueState(value) for value in state}
            removed = tuple(
                job.id
                for job in self._jobs.values()
                if job.state in selected and self._current_id != job.id
            )
            for job_id in removed:
                del self._jobs[job_id]
            if removed:
                self._persist_locked()
                self._condition.notify_all()
        return removed

    def _load(self) -> None:
        if not self._storage_path.is_file():
            self._presets["Default"] = RenderPreset()
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RenderQueueError(
                f"Cannot load render queue: {self._storage_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RenderQueueError("Render queue JSON must contain an object")
        version = int(payload.get("version", _QUEUE_VERSION))
        if version != _QUEUE_VERSION:
            raise RenderQueueError(f"Unsupported render queue version: {version}")
        raw_presets = payload.get("presets", {})
        if not isinstance(raw_presets, Mapping):
            raise RenderQueueError("Render queue presets must be an object")
        try:
            for name, raw_preset in raw_presets.items():
                preset = RenderPreset.from_dict(raw_preset)
                self._presets[str(name)] = replace(preset, name=str(name))
        except (TypeError, ValueError) as exc:
            raise RenderQueueError("Render queue contains an invalid preset") from exc
        self._presets.setdefault("Default", RenderPreset())
        raw_jobs = payload.get("jobs", [])
        if not isinstance(raw_jobs, list):
            raise RenderQueueError("Render queue jobs must be an array")
        recovered = False
        try:
            for raw_job in raw_jobs:
                job = RenderJob.from_dict(raw_job)
                if job.id in self._jobs:
                    raise RenderQueueError(f"Duplicate render job id: {job.id}")
                if job.state == QueueState.RUNNING:
                    job.state = QueueState.QUEUED
                    job.stage = "Recovered after interruption"
                    job.updated_at = _utc_now()
                    recovered = True
                self._jobs[job.id] = job
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, RenderQueueError):
                raise
            raise RenderQueueError("Render queue contains an invalid job") from exc
        if recovered:
            self._persist_locked()

    def _persist_locked(self) -> None:
        payload = {
            "version": _QUEUE_VERSION,
            "presets": {
                name: preset.to_dict() for name, preset in self._presets.items()
            },
            "jobs": [job.to_dict() for job in self._jobs.values()],
        }
        parent = self._storage_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._storage_path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._storage_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _require_job_locked(self, job_id: str) -> RenderJob:
        try:
            return self._jobs[str(job_id)]
        except KeyError as exc:
            raise JobNotFoundError(f"Unknown render job: {job_id}") from exc

    def _next_queued_locked(self) -> RenderJob | None:
        return next(
            (job for job in self._jobs.values() if job.state == QueueState.QUEUED),
            None,
        )

    def _claim_locked(self, job: RenderJob) -> RenderControl:
        control = RenderControl()
        job.state = QueueState.RUNNING
        job.stage = "Starting"
        job.completed = 0
        job.total = 0
        job.attempts += 1
        job.last_error = None
        job.output_path = None
        job.frames_rendered = 0
        job.started_at = _utc_now()
        job.finished_at = None
        job.updated_at = job.started_at
        self._current_id = job.id
        self._current_control = control
        self._persist_locked()
        return control

    def _execute_claimed(self, job: RenderJob, control: RenderControl) -> None:
        def report(progress: RenderProgress) -> None:
            self.update_progress(
                job.id, progress.current, progress.total, progress.stage
            )

        try:
            result = self._runner.run(job, control, report)
            if not isinstance(result, RenderResult):
                raise TypeError("Render runner must return RenderResult")
            with self._lock:
                if job.state == QueueState.RUNNING:
                    job.state = QueueState.COMPLETED
                    job.stage = "Completed"
                    job.output_path = result.output_path
                    job.frames_rendered = result.frames_rendered
                    job.completed = result.frames_rendered
                    job.total = max(result.frames_rendered, job.total)
                    job.finished_at = _utc_now()
                    job.updated_at = job.finished_at
                    self._persist_locked()
        except RenderCancelled:
            with self._lock:
                if job.state != QueueState.CANCELLED:
                    job.state = QueueState.CANCELLED
                    job.stage = "Cancelled"
                    job.finished_at = _utc_now()
                    job.updated_at = job.finished_at
                self._persist_locked()
        except RenderPaused:
            with self._lock:
                if job.state not in {QueueState.PAUSED, QueueState.CANCELLED}:
                    job.state = QueueState.QUEUED
                    job.stage = "Waiting to resume"
                    job.updated_at = _utc_now()
                self._persist_locked()
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            with self._lock:
                if control.is_cancelled or job.state == QueueState.CANCELLED:
                    job.state = QueueState.CANCELLED
                    job.stage = "Cancelled"
                elif job.state == QueueState.PAUSED:
                    job.stage = "Paused after runner exit"
                else:
                    job.state = QueueState.FAILED
                    job.stage = "Failed"
                    job.last_error = f"{type(exc).__name__}: {exc}"
                job.finished_at = _utc_now()
                job.updated_at = job.finished_at
                self._persist_locked()
        finally:
            with self._lock:
                if self._current_id == job.id:
                    self._current_id = None
                    self._current_control = None
                self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                job = self._next_queued_locked()
                if job is None:
                    self._condition.notify_all()
                    return
                control = self._claim_locked(job)
            self._execute_claimed(job, control)


RenderQueueManager = RenderQueue
BatchRenderQueue = RenderQueue
QueueJob = RenderJob


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _normalized_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_project_relative(path: PathLike, project_directory: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_directory / candidate
    return candidate.resolve()


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return result


def _json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        cloned = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain JSON-compatible values") from exc
    if not isinstance(cloned, dict):
        raise TypeError(f"{field_name} must contain a JSON object")
    return cloned
