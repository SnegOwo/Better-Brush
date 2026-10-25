"""Wire protocol between the EffectCraft ScriptUI panel and the Python adapter.

Two flat JSON files live in one *bridge folder*:

``job-<id>.json``     written once by the panel (``BetterBrush.jsx``), never
                      rewritten by either side after that.
``status-<id>.json``  rewritten by the adapter, atomically, as the job moves.

Nothing here imports torch, numpy or Qt: the protocol must be testable on a
machine that only has the standard library.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1

#: Directory name both sides agree on. The panel derives its own location from
#: ``$.fileName`` (the only reliable way — EffectCraft has no ``Folder.temp`` and
#: its settings folder is ``%APPDATA%\EffectCraft`` normally but sits next to the
#: executable in portable mode); the adapter is told the folder with
#: ``--bridge-dir``, with ``bridge.path.json`` and a bounded scan as fallbacks.
BRIDGE_DIRNAME = "betterbrush-bridge"

JOB_PREFIX = "job-"
STATUS_PREFIX = "status-"
CANCEL_PREFIX = "cancel-"
#: The adapter's liveness stamp, rewritten every watch cycle and while a job
#: runs (``touch_alive``). The panel reads it to decide whether to start one.
ALIVE_NAME = "adapter.alive"
JOB_SUFFIX = ".json"
STATUS_SUFFIX = ".json"

# Job states. ``queued`` and ``running`` are the only non-terminal ones.
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"

TERMINAL_STATES = frozenset({STATE_DONE, STATE_ERROR, STATE_CANCELLED})

#: The engine's stage ids, in pipeline order. ``load`` has no frame count and
#: reports 0/0, so the progress bar goes indeterminate for it. ``post`` is the
#: adapter's own pass (see ``postprocess.py``), not an engine stage.
STAGES = ("load", "extract", "analyze", "propagate", "matte", "post", "export")

MODES = ("simulate", "engine")


class ProtocolError(ValueError):
    """A job or status document that does not match the protocol."""


# --------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------- #
def candidate_bridge_dirs(explicit: str | os.PathLike[str] | None = None) -> list[Path]:
    """Every folder the adapter should watch, most likely first.

    ``explicit`` (or ``$BETTERBRUSH_BRIDGE_DIR``) always wins. The rest are
    derived: a recorded path, the per-user EffectCraft settings folder, the
    portable ``EffectCraftData`` folder next to an EffectCraft install we can
    see, and finally the OS temp directory — the panel falls back to those too
    when ``$.fileName`` resolution surprises us.
    """
    out: list[Path] = []

    def add(path: Any) -> None:
        if not path:
            return
        p = Path(str(path)).expanduser()
        if p not in out:
            out.append(p)

    add(explicit)
    add(os.environ.get("BETTERBRUSH_BRIDGE_DIR"))
    recorded = recorded_bridge_dir()
    add(recorded)

    appdata = os.environ.get("APPDATA")
    if appdata:
        add(Path(appdata) / "EffectCraft" / BRIDGE_DIRNAME)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        add(Path(local) / "EffectCraft" / BRIDGE_DIRNAME)
        # Portable installs keep settings in <install>\EffectCraftData.
        add(Path(local) / "EffectCraftData" / BRIDGE_DIRNAME)
    add(Path.home() / "AppData" / "Roaming" / "EffectCraft" / BRIDGE_DIRNAME)
    temp = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp:
        add(Path(temp) / BRIDGE_DIRNAME)
    else:
        import tempfile

        add(Path(tempfile.gettempdir()) / BRIDGE_DIRNAME)
    return out


def recorded_bridge_dir(cfg: os.PathLike[str] | str | None = None) -> Path | None:
    """The folder ``scripts/install.ps1`` resolved, if it ran here.

    ``cfg`` exists for tests; normally the ``bridge.path.json`` next to the
    MVP folder.
    """
    if cfg is None:
        here = Path(__file__).resolve().parent.parent
        cfg = here / "bridge.path.json"
    path = Path(cfg)
    try:
        # utf-8-sig: install.ps1 may write this file with a BOM, and a BOM
        # makes json.loads raise, which would silently drop the recorded path
        # and send the adapter to a fallback folder the panel never writes to.
        raw_text = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw_text)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("bridge_dir")
    return Path(raw) if isinstance(raw, str) and raw else None


def bridge_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """First candidate that exists, else the most likely one (created on use)."""
    for cand in candidate_bridge_dirs(explicit):
        if cand.is_dir():
            return cand
    cands = candidate_bridge_dirs(explicit)
    return cands[0] if cands else Path.cwd() / BRIDGE_DIRNAME


def job_path(root: Path, job_id: str) -> Path:
    return root / f"{JOB_PREFIX}{job_id}{JOB_SUFFIX}"


def status_path(root: Path, job_id: str) -> Path:
    return root / f"{STATUS_PREFIX}{job_id}{STATUS_SUFFIX}"


def touch_alive(root: Path) -> None:
    """Stamp the adapter's liveness file, ``adapter.alive``.

    The content is the epoch in milliseconds — a number the panel can compare
    with its own clock without parsing a date, in whatever timezone it has.
    The watcher stamps it every cycle *and* the pipeline stamps it while a job
    runs, so a busy adapter is never mistaken for a dead one (a render can
    block the watch loop for minutes).
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / ALIVE_NAME).write_text(str(int(time.time() * 1000)),
                                       encoding="ascii")
    except OSError:
        pass  # a read-only bridge must not stop the adapter working


def cancel_path(root: Path, job_id: str) -> Path:
    return root / f"{CANCEL_PREFIX}{job_id}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write through a temp file + ``os.replace``.

    The panel polls this file every few hundred milliseconds; a half-written
    document would show up as a parse error in the UI on exactly the frame the
    job finished.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:  # pragma: no cover - race with a writer
        raise ProtocolError(f"cannot read {path.name}: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ProtocolError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError(f"{path.name} must hold a JSON object")
    return data


def new_job_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------- #
# Correction (panel -> adapter, inside a job)
# --------------------------------------------------------------------- #
@dataclass
class Correction:
    """Brush strokes that re-track an earlier run instead of starting over.

    The panel paints them on the frame it is looking at and sends them as
    polygons in media pixels — the same space the engine's correction mask is
    rasterised in. The adapter turns them into the ``correction_add`` /
    ``correction_remove`` PNG masks ``src.engine.schemas.JobRequest`` expects.

    ``resume_dir`` is the *work* directory of the run being corrected: the
    engine re-uses its stored seed and masks instead of searching for a subject
    again, so the clip re-tracks from ``frame`` rather than from scratch.
    """

    #: Work folder of the earlier engine run (``<bridge>/out/<job-id>``).
    resume_dir: str
    #: Frame the strokes were painted on, 0-based *within the job's range*.
    frame: int = 0
    #: Polygons to force into the foreground / background, layer pixel space.
    add: list[list[tuple[float, float]]] = field(default_factory=list)
    remove: list[list[tuple[float, float]]] = field(default_factory=list)
    #: Footage item (id, else name) of the matte layer already in the comp, so
    #: the finished frames can *replace* it instead of stacking a second matte.
    #: Kept as the panel sent it (a JSON number for an id, a string for a
    #: name): EffectCraft's ``file.replaceFootage`` takes either, and coercing
    #: an id to a string is exactly the sort of guess that picks the wrong item.
    replace_item: str | int = ""
    #: Masks that were used as strokes; the commands park them in mask mode
    #: ``None`` afterwards so they never clip the footage they were drawn on.
    mask_names: list[str] = field(default_factory=list)
    #: How hard a ``remove`` stroke bites, 1-5. The adapter grows the painted
    #: region by a margin and stamps it out of the finished frames, because
    #: the engine's own remove only trims the *seed* and re-propagation can
    #: creep back over a large object.
    remove_force: int = 3

    @classmethod
    def from_dict(cls, data: Any) -> Correction:
        if not isinstance(data, dict):
            raise ProtocolError("'correction' must be an object")
        resume = data.get("resume_dir")
        if not isinstance(resume, str) or not resume.strip():
            raise ProtocolError("'correction.resume_dir' is required")
        add = _polygons(data.get("add"), "correction.add")
        remove = _polygons(data.get("remove"), "correction.remove")
        if not add and not remove:
            raise ProtocolError(
                "'correction' carries no strokes (both 'add' and 'remove' are empty)"
            )
        replace_item = data.get("replace_item", "")
        if isinstance(replace_item, float):
            if not replace_item.is_integer():
                raise ProtocolError("'correction.replace_item' must be an item id or name")
            replace_item = int(replace_item)
        if isinstance(replace_item, bool) or not isinstance(replace_item, (str, int)):
            raise ProtocolError("'correction.replace_item' must be an item id or name")
        force = _as_int(data.get("remove_force", 3), "correction.remove_force")
        return cls(
            resume_dir=resume.strip(),
            frame=max(0, _as_int(data.get("frame", 0), "correction.frame")),
            add=add,
            remove=remove,
            replace_item=replace_item,
            mask_names=_strings(data.get("mask_names"), "correction.mask_names"),
            remove_force=min(5, max(1, force)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resume_dir": self.resume_dir,
            "frame": self.frame,
            "add": [[[x, y] for x, y in poly] for poly in self.add],
            "remove": [[[x, y] for x, y in poly] for poly in self.remove],
            "replace_item": self.replace_item,
            "mask_names": list(self.mask_names),
            "remove_force": self.remove_force,
        }


# --------------------------------------------------------------------- #
# Job (panel -> adapter)
# --------------------------------------------------------------------- #
@dataclass
class Job:
    """One "remove the background of this layer" request."""

    id: str
    source_path: str
    frame_in: int = 0
    frame_out: int = -1
    fps: float = 0.0
    width: int = 0
    height: int = 0
    mode: str = "simulate"
    preset: str = "anime"
    quality: str = "balanced"
    matting: bool = False
    #: Run the adapter's post-processing pass over the finished alpha: temporal
    #: smoothing (less flicker between frames) plus an edge cleanup. The panel
    #: exposes it as a checkbox, on by default like the desktop app's
    #: "Post-process masks".
    post_process: bool = True
    #: Brush strokes for a re-track; ``None`` for a plain run.
    correction: Correction | None = None

    #: Composition the layer lives in (comp time, seconds).
    comp_name: str = ""
    comp_width: int = 0
    comp_height: int = 0
    comp_frame_rate: float = 0.0
    comp_duration: float = 0.0
    comp_time: float = 0.0

    #: The layer being rotoscoped.
    layer_id: int = 0
    layer_name: str = ""
    layer_index: int = 1
    layer_in: float = 0.0
    layer_out: float = 0.0
    layer_start: float = 0.0
    layer_stretch: float = 100.0

    #: 1-based stack position the matte sequence is inserted at. It must be the
    #: source layer's own index: the import pushes the source down by one, so
    #: afterwards the matte is at ``matte_index`` and the source at
    #: ``matte_index + 1``.
    matte_index: int = 1

    #: The source layer's transform, in comp pixels / percent, copied onto the
    #: matte layer so both occupy the same layer pixel space.
    transform: dict[str, Any] = field(default_factory=dict)

    created: str = ""
    protocol: int = PROTOCOL_VERSION

    # -------------------------------------------------------------- #
    @classmethod
    def from_dict(cls, data: Any) -> Job:
        if not isinstance(data, dict):
            raise ProtocolError("job document must be a JSON object")
        if data.get("protocol") != PROTOCOL_VERSION:
            raise ProtocolError(
                f"unsupported protocol {data.get('protocol')!r} "
                f"(adapter speaks {PROTOCOL_VERSION})"
            )
        job_id = data.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ProtocolError("'id' is required")
        source = _require_str(data.get("source"), "source")
        mode = str(data.get("mode", "simulate")).lower()
        if mode not in MODES:
            raise ProtocolError(f"'mode' must be one of {', '.join(MODES)}")

        rng = data.get("range") or {}
        if not isinstance(rng, dict):
            raise ProtocolError("'range' must be an object")
        frame_in = _as_int(rng.get("frame_in", 0), "range.frame_in")
        frame_out = _as_int(rng.get("frame_out", -1), "range.frame_out")
        if frame_out != -1 and frame_out < frame_in:
            raise ProtocolError("'range.frame_out' must be >= 'range.frame_in'")

        comp = data.get("comp") or {}
        layer = data.get("source_layer") or {}
        target = data.get("target") or {}
        for name, blob in (("comp", comp), ("source_layer", layer), ("target", target)):
            if not isinstance(blob, dict):
                raise ProtocolError(f"'{name}' must be an object")

        fps = _as_float(comp.get("frame_rate", 0.0), "comp.frame_rate")
        media = data.get("media") or {}
        if not isinstance(media, dict):
            raise ProtocolError("'media' must be an object")
        width = _as_int(media.get("width", 0), "media.width")
        height = _as_int(media.get("height", 0), "media.height")
        if width <= 0 or height <= 0:
            # Fall back to the comp's size: a solid or a precomp has no media
            # dimensions of its own and the matte still has to line up.
            width = _as_int(comp.get("width", 0), "comp.width")
            height = _as_int(comp.get("height", 0), "comp.height")
        if width <= 0 or height <= 0:
            raise ProtocolError("cannot determine the matte's pixel size")

        transform = data.get("transform") or {}
        if not isinstance(transform, dict):
            raise ProtocolError("'transform' must be an object")

        options = data.get("options") or {}
        if not isinstance(options, dict):
            raise ProtocolError("'options' must be an object")

        raw_correction = data.get("correction")
        correction = None
        if raw_correction is not None:
            correction = Correction.from_dict(raw_correction)

        return cls(
            id=job_id.strip(),
            source_path=source,
            frame_in=frame_in,
            frame_out=frame_out,
            fps=_as_float(media.get("frame_rate", fps), "media.frame_rate") or fps,
            width=width,
            height=height,
            mode=mode,
            preset=str(options.get("preset", "anime")).lower(),
            quality=str(options.get("quality", "balanced")).lower(),
            matting=bool(options.get("matting", False)),
            post_process=bool(options.get("post_process", True)),
            correction=correction,
            comp_name=str(comp.get("name", "")),
            comp_width=_as_int(comp.get("width", 0), "comp.width"),
            comp_height=_as_int(comp.get("height", 0), "comp.height"),
            comp_frame_rate=fps,
            comp_duration=_as_float(comp.get("duration", 0.0), "comp.duration"),
            comp_time=_as_float(comp.get("time", 0.0), "comp.time"),
            layer_id=_as_int(layer.get("id", 0), "source_layer.id"),
            layer_name=str(layer.get("name", "")),
            layer_index=max(1, _as_int(layer.get("index", 1), "source_layer.index")),
            layer_in=_as_float(layer.get("in", 0.0), "source_layer.in"),
            layer_out=_as_float(layer.get("out", 0.0), "source_layer.out"),
            layer_start=_as_float(layer.get("start", 0.0), "source_layer.start"),
            layer_stretch=_as_float(layer.get("stretch", 100.0), "source_layer.stretch"),
            matte_index=max(1, _as_int(target.get("matte_index", 1), "target.matte_index")),
            transform=dict(transform),
            created=str(data.get("created", "")),
            protocol=PROTOCOL_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "id": self.id,
            "created": self.created or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": "effectcraft",
            "mode": self.mode,
            "source": self.source_path,
            "range": {"frame_in": self.frame_in, "frame_out": self.frame_out},
            "media": {
                "width": self.width,
                "height": self.height,
                "frame_rate": self.fps,
            },
            "comp": {
                "name": self.comp_name,
                "width": self.comp_width,
                "height": self.comp_height,
                "frame_rate": self.comp_frame_rate,
                "duration": self.comp_duration,
                "time": self.comp_time,
            },
            "source_layer": {
                "id": self.layer_id,
                "name": self.layer_name,
                "index": self.layer_index,
                "in": self.layer_in,
                "out": self.layer_out,
                "start": self.layer_start,
                "stretch": self.layer_stretch,
            },
            "target": {"matte_index": self.matte_index},
            "transform": self.transform,
            "options": {
                "preset": self.preset,
                "quality": self.quality,
                "matting": self.matting,
                "post_process": self.post_process,
            },
            "correction": self.correction.to_dict() if self.correction else None,
        }

    # -------------------------------------------------------------- #
    @property
    def frame_count(self) -> int:
        """How many frames the job covers (``frame_out == -1`` means unknown)."""
        if self.frame_out == -1:
            return 0
        return self.frame_out - self.frame_in + 1

    def png_dir(self, root: Path) -> Path:
        """Where the adapter writes the alpha PNG sequence for this job.

        ``root`` is the bridge folder the job was actually found in, so the
        panel is told a path it can reach even when the adapter is watching
        several candidates.
        """
        return root / "out" / self.id / "png"


# --------------------------------------------------------------------- #
# Status (adapter -> panel)
# --------------------------------------------------------------------- #
@dataclass
class Status:
    id: str
    state: str = STATE_QUEUED
    stage: str = ""
    current: int = 0
    total: int = 0
    message: str = ""
    started: str = ""
    updated: str = ""
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_VERSION,
            "id": self.id,
            "state": self.state,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "started": self.started,
            "updated": self.updated or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": self.error,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Status:
        if not isinstance(data, dict):
            raise ProtocolError("status document must be a JSON object")
        state = str(data.get("state", STATE_QUEUED)).lower()
        if state not in (STATE_QUEUED, STATE_RUNNING, *TERMINAL_STATES):
            raise ProtocolError(f"unknown state {state!r}")
        return cls(
            id=str(data.get("id", "")),
            state=state,
            stage=str(data.get("stage", "")),
            current=_as_int(data.get("current", 0), "current"),
            total=_as_int(data.get("total", 0), "total"),
            message=str(data.get("message", "")),
            started=str(data.get("started", "")),
            updated=str(data.get("updated", "")),
            error=data.get("error") if isinstance(data.get("error"), dict) else None,
            result=data.get("result") if isinstance(data.get("result"), dict) else None,
        )

    @property
    def fraction(self) -> float:
        """0..1 for the progress bar; ``-1`` means "indeterminate"."""
        if self.state in TERMINAL_STATES:
            return 1.0 if self.state == STATE_DONE else 0.0
        if self.total <= 0:
            return -1.0
        return max(0.0, min(1.0, self.current / float(self.total)))


def write_status(root: Path, status: Status) -> Path:
    status.updated = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = status_path(root, status.id)
    write_json_atomic(path, status.to_dict())
    return path


def read_status(root: Path, job_id: str) -> Status | None:
    path = status_path(root, job_id)
    if not path.is_file():
        return None
    return Status.from_dict(read_json(path))


def cancel_requested(root: Path, job_id: str) -> bool:
    return cancel_path(root, job_id).exists()


# --------------------------------------------------------------------- #
# Small coercions — every value comes out of a JavaScript engine, so be
# strict and say which field was wrong instead of failing later.
# --------------------------------------------------------------------- #
def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"'{name}' must be a number")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"'{name}' must be a number")
    return float(value)


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"'{name}' is required")
    return value.strip()


def _polygons(value: Any, name: str) -> list[list[tuple[float, float]]]:
    """A list of strokes: each a list of ``[x, y]`` points in layer pixels."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProtocolError(f"'{name}' must be a list of polygons")
    out: list[list[tuple[float, float]]] = []
    for i, poly in enumerate(value):
        if not isinstance(poly, list):
            raise ProtocolError(f"'{name}[{i}]' must be a list of points")
        if len(poly) < 3:
            raise ProtocolError(
                f"'{name}[{i}]' needs at least 3 points to enclose an area"
            )
        points: list[tuple[float, float]] = []
        for j, pt in enumerate(poly):
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                x, y = pt
            else:
                raise ProtocolError(f"'{name}[{i}][{j}]' must be [x, y]")
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(
                x, (int, float)
            ) or not isinstance(y, (int, float)):
                raise ProtocolError(f"'{name}[{i}][{j}]' must be [x, y] numbers")
            points.append((float(x), float(y)))
        out.append(points)
    return out


def _strings(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProtocolError(f"'{name}' must be a list of strings")
    return [str(v) for v in value]
