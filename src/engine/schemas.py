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
VALID_QUALITIES = ("fast", "balanced", "best")
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
        }


@dataclass
class JobError:
    kind: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": self.message}
