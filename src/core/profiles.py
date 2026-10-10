"""Per-shot and per-object processing profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.hardware.profiler import ModelSize

#: Spill-cleaner modes. ``off`` disables the cleaner entirely.
SPILL_MODES = ("off", "despill", "alpha_key")

#: What a spill region box means. ``clean`` restricts the cleaner to the box;
#: ``protect`` does the opposite and shields the box from the cleaner.
SPILL_REGION_MODES = ("clean", "protect")


def _coerce_spill_color(value: Any) -> tuple[int, int, int] | None:
    """Return *value* as an RGB triple, or ``None`` when it is unusable.

    Accepts an ``(r, g, b)`` sequence (JSON gives back a list) or a
    ``"#rrggbb"`` / ``"rrggbb"`` string, so hand-edited project files and
    eyedropper output both work.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) != 6:
            return None
        try:
            value = (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
        except ValueError:
            return None
    try:
        red, green, blue = (int(channel) for channel in value)
    except (TypeError, ValueError):
        return None
    if not all(0 <= channel <= 255 for channel in (red, green, blue)):
        return None
    return red, green, blue


def _coerce_spill_region(value: Any) -> tuple[int, int, int, int] | None:
    """Return *value* as an ``(x, y, w, h)`` box, or ``None`` when unusable.

    Boxes are stored in image pixels to match the matte they constrain. A box
    needs a positive width and height to mean anything, so a degenerate one is
    dropped rather than clamped - an empty region would silently do nothing and
    look like a broken feature.
    """
    if value is None:
        return None
    try:
        x, y, width, height = (int(part) for part in value)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


@dataclass
class ShotProfile:
    """Settings used to process one shot or object."""

    name: str = "Default"
    model_size: ModelSize | str | None = None
    edge_smoothness: int = 3
    temporal_smoothing: bool = True
    detail_preservation: float = 0.75
    mask_inset: int = 0
    spill_color: tuple[int, int, int] | None = None
    spill_mode: str = "off"
    spill_tolerance: int = 30
    spill_radius: int = 2
    spill_region: tuple[int, int, int, int] | None = None
    spill_region_mode: str = "clean"
    cast_neutralize: bool = False
    cast_strength: int = 100
    edge_softness: int = 2
    # Learned matting (soft alpha) and AI object removal.  ``matting_model``
    # of "off" keeps the binary-mask pipeline; the other values select a
    # matting network.  The removal fields configure the diffusion remover.
    matting_model: str = "off"
    matting_resolution: int = 480
    matting_gamma: float = 1.0
    matting_grow: int = 0
    matting_overlap: int = 2
    matting_batch: int = 16
    matting_combine_objects: bool = False
    removal_resolution: int = 480
    removal_steps: int = 6
    removal_grow: int = 21
    removal_feather: int = 21
    removal_vae_tiling: bool = True

    def __post_init__(self) -> None:
        self.name = str(self.name).strip() or "Default"
        if self.model_size is not None and not isinstance(self.model_size, ModelSize):
            try:
                raw_model = getattr(self.model_size, "value", self.model_size)
                self.model_size = ModelSize(str(raw_model))
            except ValueError:
                self.model_size = None
        try:
            self.edge_smoothness = max(0, min(10, int(self.edge_smoothness)))
        except (TypeError, ValueError):
            self.edge_smoothness = 3
        self.temporal_smoothing = bool(self.temporal_smoothing)
        try:
            self.detail_preservation = float(self.detail_preservation)
        except (TypeError, ValueError):
            self.detail_preservation = 0.75
        self.detail_preservation = max(0.0, min(1.0, self.detail_preservation))
        try:
            self.mask_inset = max(0, min(10, int(self.mask_inset)))
        except (TypeError, ValueError):
            self.mask_inset = 0
        self.spill_color = _coerce_spill_color(self.spill_color)
        mode = str(self.spill_mode or "off").strip().lower()
        self.spill_mode = mode if mode in SPILL_MODES else "off"
        try:
            self.spill_tolerance = max(0, min(100, int(self.spill_tolerance)))
        except (TypeError, ValueError):
            self.spill_tolerance = 30
        try:
            self.spill_radius = max(1, min(10, int(self.spill_radius)))
        except (TypeError, ValueError):
            self.spill_radius = 2
        self.spill_region = _coerce_spill_region(self.spill_region)
        region_mode = str(self.spill_region_mode or "clean").strip().lower()
        self.spill_region_mode = (
            region_mode if region_mode in SPILL_REGION_MODES else "clean"
        )
        self.cast_neutralize = bool(self.cast_neutralize)
        try:
            self.cast_strength = max(0, min(100, int(self.cast_strength)))
        except (TypeError, ValueError):
            self.cast_strength = 100
        try:
            self.edge_softness = max(0, min(4, int(self.edge_softness)))
        except (TypeError, ValueError):
            self.edge_softness = 2
        self.matting_model = str(self.matting_model or "off").strip().lower()
        try:
            self.matting_resolution = max(352, min(2160, int(self.matting_resolution)))
        except (TypeError, ValueError):
            self.matting_resolution = 480
        try:
            self.matting_gamma = max(0.1, min(10.0, float(self.matting_gamma)))
        except (TypeError, ValueError):
            self.matting_gamma = 1.0
        try:
            self.matting_grow = max(-20, min(20, int(self.matting_grow)))
        except (TypeError, ValueError):
            self.matting_grow = 0
        try:
            self.matting_overlap = max(0, min(8, int(self.matting_overlap)))
        except (TypeError, ValueError):
            self.matting_overlap = 2
        try:
            self.matting_batch = max(16, min(512, int(self.matting_batch)))
        except (TypeError, ValueError):
            self.matting_batch = 16
        self.matting_combine_objects = bool(self.matting_combine_objects)
        try:
            self.removal_resolution = max(352, min(1080, int(self.removal_resolution)))
        except (TypeError, ValueError):
            self.removal_resolution = 480
        try:
            self.removal_steps = max(4, min(12, int(self.removal_steps)))
        except (TypeError, ValueError):
            self.removal_steps = 6
        try:
            self.removal_grow = max(0, min(64, int(self.removal_grow)))
        except (TypeError, ValueError):
            self.removal_grow = 21
        try:
            self.removal_feather = max(0, min(64, int(self.removal_feather)))
        except (TypeError, ValueError):
            self.removal_feather = 21
        self.removal_vae_tiling = bool(self.removal_vae_tiling)

    @property
    def model(self) -> ModelSize | None:
        """Return the selected model, or ``None`` for automatic selection."""
        return self.model_size if isinstance(self.model_size, ModelSize) else None

    @property
    def temporal(self) -> bool:
        """Return whether temporal smoothing is enabled."""
        return self.temporal_smoothing

    @property
    def spill_active(self) -> bool:
        """Return whether the spill cleaner should run for this profile."""
        return self.spill_mode != "off" and self.spill_color is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "model_size": self.model.value
            if isinstance(self.model_size, ModelSize)
            else self.model_size,
            "edge_smoothness": self.edge_smoothness,
            "temporal_smoothing": self.temporal_smoothing,
            "detail_preservation": self.detail_preservation,
            "mask_inset": self.mask_inset,
            "spill_color": list(self.spill_color) if self.spill_color else None,
            "spill_mode": self.spill_mode,
            "spill_tolerance": self.spill_tolerance,
            "spill_radius": self.spill_radius,
            "spill_region": list(self.spill_region) if self.spill_region else None,
            "spill_region_mode": self.spill_region_mode,
            "cast_neutralize": self.cast_neutralize,
            "cast_strength": self.cast_strength,
            "edge_softness": self.edge_softness,
            "matting_model": self.matting_model,
            "matting_resolution": self.matting_resolution,
            "matting_gamma": self.matting_gamma,
            "matting_grow": self.matting_grow,
            "matting_overlap": self.matting_overlap,
            "matting_batch": self.matting_batch,
            "matting_combine_objects": self.matting_combine_objects,
            "removal_resolution": self.removal_resolution,
            "removal_steps": self.removal_steps,
            "removal_grow": self.removal_grow,
            "removal_feather": self.removal_feather,
            "removal_vae_tiling": self.removal_vae_tiling,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ShotProfile:
        """Build a profile from persisted or user-facing values."""
        values = dict(data or {})
        if "model" in values and "model_size" not in values:
            values["model_size"] = values["model"]
        if "temporal_smooth" in values and "temporal_smoothing" not in values:
            values["temporal_smoothing"] = values["temporal_smooth"]
        if "detail" in values and "detail_preservation" not in values:
            values["detail_preservation"] = values["detail"]
        if "inset" in values and "mask_inset" not in values:
            values["mask_inset"] = values["inset"]
        if "spill" in values and "spill_color" not in values:
            values["spill_color"] = values["spill"]
        if "spill_tol" in values and "spill_tolerance" not in values:
            values["spill_tolerance"] = values["spill_tol"]
        return cls(
            name=values.get("name", "Default"),
            model_size=values.get("model_size"),
            edge_smoothness=values.get("edge_smoothness", values.get("smoothness", 3)),
            temporal_smoothing=values.get("temporal_smoothing", True),
            detail_preservation=values.get("detail_preservation", 0.75),
            mask_inset=values.get("mask_inset", 0),
            spill_color=values.get("spill_color"),
            spill_mode=values.get("spill_mode", "off"),
            spill_tolerance=values.get("spill_tolerance", 30),
            spill_radius=values.get("spill_radius", 2),
            spill_region=values.get("spill_region"),
            spill_region_mode=values.get("spill_region_mode", "clean"),
            cast_neutralize=values.get("cast_neutralize", False),
            cast_strength=values.get("cast_strength", 100),
            edge_softness=values.get("edge_softness", 2),
            matting_model=values.get("matting_model", "off"),
            matting_resolution=values.get("matting_resolution", 480),
            matting_gamma=values.get("matting_gamma", 1.0),
            matting_grow=values.get("matting_grow", 0),
            matting_overlap=values.get("matting_overlap", 2),
            matting_batch=values.get("matting_batch", 16),
            matting_combine_objects=values.get("matting_combine_objects", False),
            removal_resolution=values.get("removal_resolution", 480),
            removal_steps=values.get("removal_steps", 6),
            removal_grow=values.get("removal_grow", 21),
            removal_feather=values.get("removal_feather", 21),
            removal_vae_tiling=values.get("removal_vae_tiling", True),
        )

    def copy(self) -> ShotProfile:
        """Return an independent profile copy."""
        return ShotProfile.from_dict(self.to_dict())


@dataclass(frozen=True)
class ShotRange:
    """A one-based shot number with an inclusive frame range."""

    shot_id: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        """Return the number of frames in the shot."""
        return self.end_frame - self.start_frame + 1

    @property
    def label(self) -> str:
        """Return a human-readable shot label."""
        return f"Shot {self.shot_id} ({self.start_frame}–{self.end_frame})"


def shot_ranges(
    total_frames: int, scene_cuts: list[int] | set[int] | tuple[int, ...]
) -> list[ShotRange]:
    """Return normalized shot ranges using scene cuts as shot starts."""
    total = max(1, int(total_frames))
    cuts = sorted({int(cut) for cut in scene_cuts if 0 < int(cut) < total})
    ranges: list[ShotRange] = []
    start = 0
    shot_id = 1
    for cut in cuts:
        ranges.append(ShotRange(shot_id, start, cut - 1))
        start = cut
        shot_id += 1
    ranges.append(ShotRange(shot_id, start, total - 1))
    return ranges


def shot_id_for_frame(
    frame: int, total_frames: int, scene_cuts: list[int] | set[int] | tuple[int, ...]
) -> int:
    """Return the one-based shot ID containing ``frame``."""
    ranges = shot_ranges(total_frames, scene_cuts)
    for shot in ranges:
        if shot.start_frame <= int(frame) <= shot.end_frame:
            return shot.shot_id
    return ranges[-1].shot_id if ranges else 1


@dataclass
class ShotProfileRecord:
    """A serialized profile bound to an object and shot range."""

    shot_id: int
    start_frame: int
    end_frame: int
    profile: ShotProfile = field(default_factory=ShotProfile)
    object_id: int | None = None

    def __post_init__(self) -> None:
        self.shot_id = max(1, int(self.shot_id))
        self.start_frame = max(0, int(self.start_frame))
        self.end_frame = max(self.start_frame, int(self.end_frame))
        self.profile = ShotProfile.from_dict(
            self.profile.to_dict()
            if isinstance(self.profile, ShotProfile)
            else self.profile
        )
        if self.object_id is not None:
            self.object_id = max(1, int(self.object_id))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "object_id": self.object_id,
            "profile": self.profile.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShotProfileRecord:
        """Build a record from persisted data."""
        values = dict(data or {})
        profile = values.get("profile", values)
        return cls(
            shot_id=values.get("shot_id", 1),
            start_frame=values.get("start_frame", 0),
            end_frame=values.get("end_frame", 0),
            profile=ShotProfile.from_dict(profile if isinstance(profile, dict) else {}),
            object_id=values.get("object_id"),
        )


class ShotProfileMap:
    """Resolve the effective profile for a frame and tracked object."""

    def __init__(
        self,
        default: ShotProfile | None = None,
        records: list[ShotProfileRecord] | None = None,
    ) -> None:
        self.default = (default or ShotProfile()).copy()
        self._profiles: dict[tuple[int | None, int], ShotProfile] = {}
        for record in records or []:
            self._profiles[(record.object_id, record.shot_id)] = record.profile.copy()

    @property
    def records(self) -> list[ShotProfileRecord]:
        """Return records in stable object/shot order."""
        return [
            ShotProfileRecord(
                shot_id=shot_id,
                start_frame=0,
                end_frame=0,
                profile=profile,
                object_id=object_id,
            )
            for (object_id, shot_id), profile in sorted(
                self._profiles.items(), key=lambda item: (item[0][0] or 0, item[0][1])
            )
        ]

    def get_for_shot(self, object_id: int | None, shot_id: int) -> ShotProfile:
        """Return a copy of the effective profile for an object and shot."""
        return self._profiles.get((object_id, int(shot_id)), self.default).copy()

    def get_for_frame(
        self,
        frame: int,
        total_frames: int,
        scene_cuts: list[int] | set[int] | tuple[int, ...],
        object_id: int | None = None,
        object_default: ShotProfile | None = None,
    ) -> ShotProfile:
        """Return the effective profile at a frame."""
        shot_id = shot_id_for_frame(frame, total_frames, scene_cuts)
        if (object_id, shot_id) in self._profiles:
            return self.get_for_shot(object_id, shot_id)
        if object_default is not None:
            return object_default.copy()
        return self.default.copy()

    def set_for_shot(
        self, object_id: int | None, shot_id: int, profile: ShotProfile
    ) -> ShotProfile:
        """Bind a profile to an object and shot and return its copy."""
        selected = profile.copy()
        self._profiles[(object_id, max(1, int(shot_id)))] = selected
        return selected.copy()

    def clear_shot(self, object_id: int | None, shot_id: int) -> bool:
        """Remove an object/shot override."""
        return self._profiles.pop((object_id, int(shot_id)), None) is not None

    def clear(self) -> None:
        """Remove every object/shot override."""
        self._profiles.clear()

    def to_records(
        self,
        total_frames: int,
        scene_cuts: list[int] | set[int] | tuple[int, ...],
    ) -> list[ShotProfileRecord]:
        """Return overrides with current shot ranges attached."""
        ranges = {shot.shot_id: shot for shot in shot_ranges(total_frames, scene_cuts)}
        records: list[ShotProfileRecord] = []
        for (object_id, shot_id), profile in sorted(
            self._profiles.items(), key=lambda item: (item[0][0] or 0, item[0][1])
        ):
            shot = ranges.get(shot_id, ShotRange(shot_id, 0, max(0, total_frames - 1)))
            records.append(
                ShotProfileRecord(
                    shot_id=shot_id,
                    start_frame=shot.start_frame,
                    end_frame=shot.end_frame,
                    profile=profile.copy(),
                    object_id=object_id,
                )
            )
        return records

    @classmethod
    def from_records(
        cls,
        records: list[ShotProfileRecord] | None,
        default: ShotProfile | None = None,
    ) -> ShotProfileMap:
        """Build a map from serialized records."""
        return cls(default=default, records=records or [])


__all__ = [
    "ShotProfile",
    "ShotProfileMap",
    "ShotProfileRecord",
    "ShotRange",
    "shot_id_for_frame",
    "shot_ranges",
]
