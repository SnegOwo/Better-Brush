"""JSON-facing contracts shared by the headless engine and the AE panel.

Standard library only, on purpose: the server, the job registry and these
dataclasses must import on a machine without torch so the HTTP surface stays
testable. Anything that needs a GPU lives behind a lazy import elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------- #
# Job lifecycle
# --------------------------------------------------------------------- #
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

TERMINAL_STATES = frozenset({DONE, ERROR, CANCELLED})

# --------------------------------------------------------------------- #
# Error kinds — the panel keys its UX off these strings, so they are part
# of the contract. "no_character" / "multiple_subjects" both mean "hand
# this off to the full app", with different copy.
# --------------------------------------------------------------------- #
ERR_NO_CHARACTER = "no_character"
ERR_MULTIPLE_SUBJECTS = "multiple_subjects"
ERR_UNSUPPORTED = "unsupported"
ERR_BUSY = "busy"
ERR_BAD_REQUEST = "bad_request"
ERR_INTERNAL = "internal"

VALID_PRESETS = ("anime", "human", "object")
#: Cheapest first. ``tiny`` is SAM 2.1 Tiny — the fastest option, for iterating.
VALID_QUALITIES = ("tiny", "fast", "balanced", "best")
VALID_MODELS = ("tiny", "small", "base_plus", "large", "custom")


class RequestError(ValueError):
    """A malformed job request. Carries the error kind for the HTTP layer."""

    def __init__(self, message: str, kind: str = ERR_BAD_REQUEST) -> None:
        super().__init__(message)
        self.kind = kind


class JobCancelled(Exception):
    """Raised inside the pipeline when the cancel flag is observed."""


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"'{name}' must be a number")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"'{name}' must be a number")
    return float(value)


def _as_optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RequestError(f"'{name}' must be a string")
    return value


def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"'{name}' must be an integer")
    return int(value)


# --------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class JobRequest:
    """One background-removal job.

    ``frame_in`` / ``frame_out`` are inclusive indices into the *source* file
    of the selected layer (v1 always processes the layer's whole source, so a
    caller normally sends ``0`` and ``total_frames - 1``). ``frame_out=-1``
    means "to the end".
    """

    input_path: str
    frame_in: int = 0
    frame_out: int = -1
    fps: float = 0.0
    width: int = 0
    height: int = 0
    preset: str = "anime"
    quality: str = "balanced"
    model: str | None = None
    output_dir: str | None = None
    #: Field-of-view hint from the host (0..1 fractions of the frame). The AE
    #: panel sends the selected layer's bounds so an off-centre layer still
    #: lands with its subject near the centre of the analysis. ``None`` skips
    #: the hint. Reserved for hosts that ever pass a sub-region.
    roi: tuple[float, float, float, float] | None = None
    #: Refine the tracked outline into a soft alpha matte (hair, motion blur,
    #: partial coverage) with the MatAnyone 2 network. Off by default so a bare
    #: engine request stays fast; the panel turns it on from its dock.
    matting: bool = False
    #: Mux the source file's own audio into the alpha movie. Off by default —
    #: the bare API returns the picture — but the host panels ask for it, since
    #: the cut-out stands in for the clip it came from: Premiere's
    #: ``changeMediaPath`` refuses a replacement that has no audio where the
    #: media it replaces had some, and silently losing the sound is the wrong
    #: answer in either host.
    keep_audio: bool = False
    #: Output directory of an earlier job. When set, this job re-uses the seed
    #: mask that job produced instead of searching for a subject again — the
    #: panel's brush correction path.
    resume_dir: str | None = None
    #: Brush strokes from the panel, as base64 PNG masks at source resolution.
    #: ``add`` forces those pixels to foreground, ``remove`` to background,
    #: before the clip is re-tracked from the stored key frame.
    correction_add: str | None = None
    correction_remove: str | None = None
    #: Frame the correction was painted on (0-based within this job's range).
    #: The clip is re-tracked from *this* frame, so a wrong frame is fixed where
    #: the user saw it rather than only at the original key frame. ``None``
    #: falls back to the stored key frame.
    correction_frame: int | None = None

    @classmethod
    def from_dict(cls, data: Any) -> JobRequest:
        if not isinstance(data, dict):
            raise RequestError("request body must be a JSON object")
        input_path = data.get("input_path")
        if not isinstance(input_path, str) or not input_path.strip():
            raise RequestError("'input_path' is required")

        preset = str(data.get("preset", "anime")).lower()
        if preset not in VALID_PRESETS:
            raise RequestError(
                f"'preset' must be one of {', '.join(VALID_PRESETS)}"
            )
        quality = str(data.get("quality", "balanced")).lower()
        if quality not in VALID_QUALITIES:
            raise RequestError(
                f"'quality' must be one of {', '.join(VALID_QUALITIES)}"
            )
        model = data.get("model")
        if model is not None:
            model = str(model).lower()
            if model not in VALID_MODELS:
                raise RequestError(
                    f"'model' must be one of {', '.join(VALID_MODELS)}"
                )

        frame_in = _as_int(data.get("frame_in", 0), "frame_in")
        frame_out = _as_int(data.get("frame_out", -1), "frame_out")
        if frame_in < 0:
            raise RequestError("'frame_in' must be >= 0")
        if frame_out != -1 and frame_out < frame_in:
            raise RequestError("'frame_out' must be >= 'frame_in' (or -1)")

        fps = _as_float(data.get("fps", 0.0), "fps")
        if fps < 0:
            raise RequestError("'fps' must be >= 0")
        width = _as_int(data.get("width", 0), "width")
        height = _as_int(data.get("height", 0), "height")

        output_dir = data.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise RequestError("'output_dir' must be a string")

        resume_dir = _as_optional_str(data.get("resume_dir"), "resume_dir")
        correction_add = _as_optional_str(
            data.get("correction_add"), "correction_add"
        )
        correction_remove = _as_optional_str(
            data.get("correction_remove"), "correction_remove"
        )
        correction_frame = _as_optional_int(
            data.get("correction_frame"), "correction_frame"
        )
        matting = bool(data.get("matting", False))
        keep_audio = bool(data.get("keep_audio", False))

        roi_raw = data.get("roi")
        roi: tuple[float, float, float, float] | None = None
        if roi_raw is not None:
            if not isinstance(roi_raw, (list, tuple)) or len(roi_raw) != 4:
                raise RequestError("'roi' must be [x, y, w, h] fractions")
            roi = tuple(float(v) for v in roi_raw)  # type: ignore[assignment]

        return cls(
            input_path=input_path.strip(),
            frame_in=frame_in,
            frame_out=frame_out,
            fps=fps,
            width=width,
            height=height,
            preset=preset,
            quality=quality,
            model=model,
            output_dir=output_dir,
            roi=roi,
            matting=matting,
            keep_audio=keep_audio,
            resume_dir=resume_dir,
            correction_add=correction_add,
            correction_remove=correction_remove,
            correction_frame=correction_frame,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "frame_in": self.frame_in,
            "frame_out": self.frame_out,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "preset": self.preset,
            "quality": self.quality,
            "model": self.model,
            "output_dir": self.output_dir,
            "matting": self.matting,
            "keep_audio": self.keep_audio,
            "resume_dir": self.resume_dir,
            "correction_frame": self.correction_frame,
            # The correction strokes are one-shot payloads: echoing the base64
            # in every progress poll would dwarf the job state itself.
        }


# --------------------------------------------------------------------- #
# Progress / results
# --------------------------------------------------------------------- #
@dataclass
class JobProgress:
    stage: str = QUEUED
    current: int = 0
    total: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "message": self.message,
        }


@dataclass
class SubjectInfo:
    """What the auto-subject heuristic decided about the key frame."""

    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None
    face_seeded: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "box": list(self.box) if self.box else None,
            "face_seeded": self.face_seeded,
            "reasons": list(self.reasons),
        }


@dataclass
class JobResult:
    output_path: str
    output_kind: str  # "mov_alpha" | "png_sequence"
    frame_in: int
    frame_out: int
    frame_count: int
    fps: float
    width: int
    height: int
    preview_path: str | None = None
    subject: SubjectInfo | None = None
    qc: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    #: Where the job left its working files (seed mask + metadata). The panel
    #: hands this back as ``resume_dir`` so a brush correction re-tracks from
    #: the same key frame instead of searching for the subject again.
    work_dir: str = ""
    #: Whether the soft matte was applied (False = the segment outline only).
    matting: bool = False
    #: The frame the subject was seeded on (0-based within this job's range).
    #: The panel opens its correction scrubber here.
    key_idx: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "output_kind": self.output_kind,
            "frame_in": self.frame_in,
            "frame_out": self.frame_out,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "preview_path": self.preview_path,
            "subject": self.subject.to_dict() if self.subject else None,
            "qc": dict(self.qc),
            "model": self.model,
            "work_dir": self.work_dir,
            "matting": self.matting,
            "key_idx": self.key_idx,
        }


@dataclass
class JobError:
    kind: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": self.message}
