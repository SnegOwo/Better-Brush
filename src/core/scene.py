"""Scene model: a video can hold multiple tracked objects (characters).

Each object keeps its own prompt points, masks and keyframes. The active
object's data is aliased into the main window's ``_points`` / ``_masks`` /
``_keyframes`` fields so all existing editing code keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.sam2_manager import PointPrompt

# Muted palette, cycling when a new object is added.
OBJECT_PALETTE = [
    "#00b4f0",  # cyan
    "#f4b400",  # amber
    "#00e08a",  # green
    "#e05b5b",  # red
    "#c77dff",  # violet
    "#ff9e00",  # orange
    "#00c8c8",  # teal
    "#ff6ad5",  # pink
]


@dataclass
class SceneObject:
    """One tracked character / object within the current video."""

    obj_id: int
    name: str
    color: str
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> bool mask
    points: list[PointPrompt] = field(default_factory=list)
    keyframes: set[int] = field(default_factory=set)


def next_object_name(objects: list[SceneObject]) -> str:
    """Return the next free ``Object N`` name for the given scene."""
    taken = {obj.name for obj in objects}
    n = len(objects) + 1
    while f"Object {n}" in taken:
        n += 1
    return f"Object {n}"


def next_object_color(objects: list[SceneObject]) -> str:
    """Pick the next palette color (skips colors already in use)."""
    used = {obj.color for obj in objects}
    for color in OBJECT_PALETTE:
        if color not in used:
            return color
    return OBJECT_PALETTE[len(objects) % len(OBJECT_PALETTE)]