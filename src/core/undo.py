"""Snapshot-based undo/redo manager for rotoscoping state.

State model: the stack stores snapshots of the editing state. A baseline
snapshot is recorded when a session starts (video/project load); after every
mutation the resulting state is pushed, so undo walks backwards and redo
forwards.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.models.sam2_manager import PointPrompt


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of the editing state."""

    points: tuple[PointPrompt, ...]
    masks: dict[int, np.ndarray]
    keyframes: set[int] = field(default_factory=set)


class UndoManager:
    """Maintains an undo/redo stack of editing state snapshots."""

    def __init__(self, max_steps: int = 30) -> None:
        self._max_steps = max_steps
        self._stack: list[StateSnapshot] = []
        self._index: int = -1  # points at the current state

    def reset(self, points: list[PointPrompt], masks: dict[int, np.ndarray], keyframes: set[int]) -> None:
        """Start a fresh session with the given state as the baseline."""
        self._stack.clear()
        self._push_snapshot(points, masks, keyframes)
        self._index = 0

    def push(self, points: list[PointPrompt], masks: dict[int, np.ndarray], keyframes: set[int]) -> None:
        """Save the current state after a mutation. Discards any redo history."""
        # Discard redo history first so the freshly pushed state is not removed
        del self._stack[self._index + 1:]
        self._push_snapshot(points, masks, keyframes)
        self._index = len(self._stack) - 1
        # Enforce max depth (keep the baseline at index 0 if possible)
        if len(self._stack) > self._max_steps:
            overflow = len(self._stack) - self._max_steps
            self._stack = self._stack[overflow:]
            self._index -= overflow

    def _push_snapshot(self, points: list[PointPrompt], masks: dict[int, np.ndarray], keyframes: set[int]) -> None:
        """Copy the live state into a snapshot and append it."""
        snap = StateSnapshot(
            points=tuple(copy.deepcopy(points)),
            masks={k: v.copy() for k, v in masks.items()},
            keyframes=set(keyframes),
        )
        self._stack.append(snap)

    def undo(self, points: list[PointPrompt], masks: dict[int, np.ndarray], keyframes: set[int]) -> Optional[StateSnapshot]:
        """Revert to the previous state. Returns the restored snapshot or None."""
        if self._index <= 0:
            return None
        self._index -= 1
        return self._restore(points, masks, keyframes)

    def redo(self, points: list[PointPrompt], masks: dict[int, np.ndarray], keyframes: set[int]) -> Optional[StateSnapshot]:
        """Advance to the next state. Returns the restored snapshot or None."""
        if self._index >= len(self._stack) - 1:
            return None
        self._index += 1
        return self._restore(points, masks, keyframes)

    def _restore(self, points: list[PointPrompt], masks: dict[int, np.ndarray], keyframes: set[int]) -> StateSnapshot:
        """Copy the current snapshot into the live state containers."""
        snap = self._stack[self._index]
        points.clear()
        points.extend(copy.deepcopy(snap.points))
        masks.clear()
        for k, v in snap.masks.items():
            masks[k] = v.copy()
        keyframes.clear()
        keyframes.update(snap.keyframes)
        return snap

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._stack) - 1

    def clear(self) -> None:
        """Reset the undo history (e.g. on project load)."""
        self._stack.clear()
        self._index = -1