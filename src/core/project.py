"""Project files: save/load rotoscoping sessions (.bbproj).

A project stores the video reference, prompt points, keyframes, masks and
view state as one JSON file so work can be resumed later without losing
annotations.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.profiles import ShotProfile, ShotProfileRecord

logger = logging.getLogger(__name__)

PROJECT_EXT = ".bbproj"
RECENTS_FILE = "recent_projects.json"
USER_SETTINGS_FILE = "user_settings.json"
AUTOSAVE_DIR_NAME = "autosaves"
_MAX_RECENT = 8
_PROJECT_VERSION = 3


# ── Serialization helpers ────────────────────────────────────────────────
def mask_to_b64(mask: np.ndarray) -> str:
    """Encode a bool mask as base64 PNG bytes."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def mask_from_b64(data: str) -> np.ndarray:
    """Decode a base64 PNG back into a bool mask."""
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("L")
    return np.asarray(img) > 127


def matte_to_b64(alpha: np.ndarray) -> str:
    """Encode a float32 soft matte as a base64 grayscale PNG."""
    values = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    img = Image.fromarray((values * 255.0).round().astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def matte_from_b64(data: str) -> np.ndarray:
    """Decode a base64 PNG back into a float32 soft matte in [0, 1]."""
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _profile_payload(value: ShotProfile | dict | None) -> dict | None:
    if value is None:
        return None
    if isinstance(value, ShotProfile):
        return value.to_dict()
    if isinstance(value, dict):
        return ShotProfile.from_dict(value).to_dict()
    return ShotProfile().to_dict()


def _shot_profile_payload(value: ShotProfileRecord | dict) -> dict:
    if isinstance(value, ShotProfileRecord):
        return value.to_dict()
    if isinstance(value, dict):
        return ShotProfileRecord.from_dict(value).to_dict()
    return ShotProfileRecord(1, 0, 0).to_dict()


@dataclass
class ProjectData:
    """Serializable snapshot of a rotoscoping session."""

    version: int = 1
    video_path: str | None = None
    video_meta: dict = field(
        default_factory=dict
    )  # fps / width / height / total_frames
    points: list[dict] = field(default_factory=list)  # {x, y, label}
    keyframes: list[int] = field(default_factory=list)
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    current_frame: int = 0
    edit_mode: str = "points"
    brush_size: int = 20
    timeline_selection: list[int] | None = None
    objects: list[ObjectRecord] = field(default_factory=list)
    active_object: int | None = None
    scene_cuts: list[int] = field(default_factory=list)
    default_profile: ShotProfile = field(default_factory=ShotProfile)
    shot_profiles: list[ShotProfileRecord] = field(default_factory=list)

    @property
    def video_name(self) -> str:
        if self.video_path:
            return Path(self.video_path).name
        return ""


@dataclass
class ObjectRecord:
    """Serialized per-object data for multi-object scenes."""

    obj_id: int
    name: str
    color: str
    points: list[dict] = field(default_factory=list)
    keyframes: list[int] = field(default_factory=list)
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    #: Learned soft mattes per frame (float32 alpha in [0, 1]).
    mattes: dict[int, np.ndarray] = field(default_factory=dict)
    profile: ShotProfile | None = None


class ProjectManager:
    """Handle .bbproj persistence and the recent-projects registry."""

    # ── Settings / registry ───────────────────────────────────────────────
    @staticmethod
    def settings_dir() -> Path:
        base = os.getenv("APPDATA")
        d = (Path(base) / "BetterBrush") if base else (Path.home() / ".betterbrush")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def _registry_path(cls) -> Path:
        return cls.settings_dir() / RECENTS_FILE

    @classmethod
    def _read_registry(cls) -> dict:
        path = cls._registry_path()
        if not path.is_file():
            return {"recent": [], "first_run_done": False}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"recent": [], "first_run_done": False}

    @classmethod
    def _write_registry(cls, registry: dict) -> None:
        cls._registry_path().write_text(
            json.dumps(registry, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_recent(cls) -> list[dict]:
        """Return recent projects (most recent first)."""
        return cls._read_registry().get("recent", [])

    @classmethod
    def add_recent(cls, project_path: Path, video_path: Path | None = None) -> None:
        """Record a saved project at the top of the recent list."""
        registry = cls._read_registry()
        recent = registry.get("recent", [])
        key = str(project_path)
        recent = [r for r in recent if r.get("project") != key]
        entry = {
            "project": key,
            "video": str(video_path) if video_path else "",
            "video_name": video_path.name if video_path else project_path.name,
            "saved_at": datetime.now(UTC).isoformat(timespec="minutes"),
        }
        recent.insert(0, entry)
        registry["recent"] = recent[:_MAX_RECENT]
        cls._write_registry(registry)

    @classmethod
    def remove_recent(cls, project_path: Path) -> None:
        registry = cls._read_registry()
        key = str(project_path)
        registry["recent"] = [
            r for r in registry.get("recent", []) if r.get("project") != key
        ]
        cls._write_registry(registry)

    @classmethod
    def is_first_run(cls) -> bool:
        return not bool(cls._read_registry().get("first_run_done"))

    @classmethod
    def set_first_run_done(cls) -> None:
        registry = cls._read_registry()
        registry["first_run_done"] = True
        cls._write_registry(registry)

    # ── User settings (auto-save etc.) ───────────────────────────────────
    @classmethod
    def settings_file(cls) -> Path:
        """Path to the persistent user-preferences JSON file."""
        return cls.settings_dir() / USER_SETTINGS_FILE

    @classmethod
    def load_user_settings(cls) -> dict:
        """Load persisted user preferences, falling back to defaults."""
        defaults = {
            "auto_save_enabled": True,
            "auto_save_interval_s": 60,
            "post_process_enabled": True,
            "smooth_strength": 3,
            "console_shortcut": "F7",
        }
        path = cls.settings_file()
        if not path.is_file():
            return defaults
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return defaults
        except (OSError, json.JSONDecodeError):
            return defaults
        merged = dict(defaults)
        if "post_process_enabled" not in data and "smooth_strength" in data:
            try:
                merged["post_process_enabled"] = int(data["smooth_strength"]) > 0
            except (TypeError, ValueError):
                pass
        merged.update(data)
        return merged

    @classmethod
    def save_user_settings(cls, data: dict) -> None:
        """Persist user preferences (merged with current defaults)."""
        current = cls.load_user_settings()
        current.update(data)
        try:
            cls.settings_file().write_text(
                json.dumps(current, indent=2), encoding="utf-8"
            )
        except OSError:
            return

    @classmethod
    def autosave_dir(cls) -> Path:
        """Directory that holds automatic-save project files."""
        d = cls.settings_dir() / AUTOSAVE_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Project save / load ───────────────────────────────────────────────
    @staticmethod
    def _ensure_ext(path: Path) -> Path:
        return path if path.suffix == PROJECT_EXT else path.with_suffix(PROJECT_EXT)

    @classmethod
    def save(cls, path: Path, data: ProjectData) -> Path:
        """Serialize a project to disk. Returns the canonical path."""
        path = cls._ensure_ext(path)
        has_new_state = bool(data.objects or data.shot_profiles)
        payload = {
            "app": "Better Brush",
            "version": _PROJECT_VERSION if has_new_state else int(data.version or 1),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "video_path": data.video_path,
            "video_meta": data.video_meta,
            "current_frame": int(data.current_frame or 0),
            "edit_mode": data.edit_mode or "points",
            "brush_size": int(data.brush_size or 20),
            "timeline_selection": list(data.timeline_selection)
            if data.timeline_selection
            else None,
            "scene_cuts": sorted(int(c) for c in data.scene_cuts) if data.scene_cuts else [],
            "default_profile": _profile_payload(data.default_profile) or ShotProfile().to_dict(),
            "shot_profiles": [_shot_profile_payload(record) for record in (data.shot_profiles or [])],
            "points": [
                {
                    "x": int(p.get("x", 0)),
                    "y": int(p.get("y", 0)),
                    "label": int(p.get("label", 1)),
                }
                for p in data.points
            ],
            "keyframes": sorted(int(k) for k in data.keyframes),
            "masks": {str(i): mask_to_b64(m) for i, m in sorted(data.masks.items())},
            "active_object": data.active_object,
            "objects": [
                {
                    "id": int(rec.obj_id),
                    "name": rec.name,
                    "color": rec.color,
                    "points": [
                        {
                            "x": int(p.get("x", 0)),
                            "y": int(p.get("y", 0)),
                            "label": int(p.get("label", 1)),
                        }
                        for p in rec.points
                    ],
                    "keyframes": sorted(int(k) for k in rec.keyframes),
                    "masks": {
                        str(i): mask_to_b64(m) for i, m in sorted(rec.masks.items())
                    },
                    "mattes": {
                        str(i): matte_to_b64(a)
                        for i, a in sorted(rec.mattes.items())
                    },
                    "profile": _profile_payload(rec.profile),
                }
                for rec in data.objects
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: Path) -> ProjectData:
        """Deserialize a project. Raises if malformed or missing."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Not a valid Better Brush project: {path.name}")
        masks: dict[int, np.ndarray] = {}
        masks_raw = payload.get("masks", {})
        if isinstance(masks_raw, dict):
            for k, b64 in masks_raw.items():
                try:
                    masks[int(k)] = mask_from_b64(b64)
                except Exception:  # noqa: BLE001 - keep other frames on bad frame
                    logger.debug("Skipping unreadable mask for frame %s", k)
                    continue

        selection = payload.get("timeline_selection")
        if isinstance(selection, (list, tuple)) and len(selection) == 2:
            selection = [int(selection[0]), int(selection[1])]
        else:
            selection = None

        points = payload.get("points", [])
        if not isinstance(points, list):
            points = []
        points = [
            {
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "label": int(p.get("label", 1)),
            }
            for p in points
            if isinstance(p, dict)
        ]

        objects: list[ObjectRecord] = []
        active_object: int | None = None
        keyframes = sorted(
            int(k) for k in payload.get("keyframes", []) if isinstance(k, int)
        )
        raw_objects = payload.get("objects")
        if isinstance(raw_objects, list) and raw_objects:
            for raw in raw_objects:
                if not isinstance(raw, dict):
                    continue
                obj_masks: dict[int, np.ndarray] = {}
                obj_masks_raw = raw.get("masks", {})
                if isinstance(obj_masks_raw, dict):
                    for k, b64 in obj_masks_raw.items():
                        try:
                            obj_masks[int(k)] = mask_from_b64(b64)
                        except Exception:  # noqa: BLE001
                            logger.debug("Skipping unreadable mask frame %s", k)
                            continue
                obj_mattes: dict[int, np.ndarray] = {}
                obj_mattes_raw = raw.get("mattes", {})
                if isinstance(obj_mattes_raw, dict):
                    for k, b64 in obj_mattes_raw.items():
                        try:
                            obj_mattes[int(k)] = matte_from_b64(b64)
                        except Exception:  # noqa: BLE001
                            logger.debug("Skipping unreadable matte frame %s", k)
                            continue
                obj_points = raw.get("points", [])
                if not isinstance(obj_points, list):
                    obj_points = []
                objects.append(
                    ObjectRecord(
                        obj_id=int(raw.get("id", 0) or 0),
                        name=str(raw.get("name", "Object 1")),
                        color=str(raw.get("color", "#00b4f0")),
                        keyframes=sorted(
                            int(k)
                            for k in raw.get("keyframes", [])
                            if isinstance(k, int)
                        ),
                        points=[
                            {
                                "x": int(p.get("x", 0)),
                                "y": int(p.get("y", 0)),
                                "label": int(p.get("label", 1)),
                            }
                            for p in obj_points
                            if isinstance(p, dict)
                        ],
                        masks=obj_masks,
                        mattes=obj_mattes,
                        profile=(
                            ShotProfile.from_dict(raw.get("profile"))
                            if isinstance(raw.get("profile"), dict)
                            else None
                        ),
                    )
                )
            active_object = payload.get("active_object")
            active_object = (
                int(active_object)
                if isinstance(active_object, int)
                and not isinstance(active_object, bool)
                else None
            )
            # Top-level fields fall back to the first object for tools that
            # do not understand multi-object projects yet.
            if objects:
                masks = objects[0].masks
                points = objects[0].points
                keyframes = set(objects[0].keyframes)

        default_profile = ShotProfile.from_dict(
            payload.get("default_profile")
            if isinstance(payload.get("default_profile"), dict)
            else {}
        )
        shot_profiles: list[ShotProfileRecord] = []
        raw_profiles = payload.get("shot_profiles", [])
        if isinstance(raw_profiles, list):
            for raw_profile in raw_profiles:
                if not isinstance(raw_profile, dict):
                    continue
                try:
                    shot_profiles.append(ShotProfileRecord.from_dict(raw_profile))
                except (TypeError, ValueError):
                    logger.debug("Skipping invalid shot profile")

        return ProjectData(
            version=int(payload.get("version", 1) or 1),
            video_path=payload.get("video_path"),
            video_meta=dict(payload.get("video_meta") or {}),
            points=points,
            keyframes=list(keyframes),
            masks=masks,
            current_frame=int(payload.get("current_frame", 0) or 0),
            edit_mode=str(payload.get("edit_mode", "points")),
            brush_size=int(payload.get("brush_size", 20) or 20),
            timeline_selection=selection,
            objects=objects,
            active_object=active_object,
            scene_cuts=sorted(
                int(c)
                for c in payload.get("scene_cuts", [])
                if isinstance(c, (int, float))
            ),
            default_profile=default_profile,
            shot_profiles=shot_profiles,
        )

    # ── Misc ──────────────────────────────────────────────────────────────
    @staticmethod
    def touch_mtime(path: Path) -> float:
        return path.stat().st_mtime if path.is_file() else 0.0


def format_relative_time(timestamp: str) -> str:
    """Human-read "just now / Nm ago / date" from an ISO string."""
    try:
        dt = datetime.fromisoformat(timestamp)
    except (ValueError, TypeError):
        return ""
    delta = time.time() - dt.timestamp()
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)}d ago"
    return dt.strftime("%b %d, %Y")
