"""Focused tests for the versioned project history service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.core.history import ProjectHistory, default_history_dir
from src.core.project import ProjectData, ProjectManager


def _data(frame: int = 0) -> ProjectData:
    """Return a small ProjectData snapshot for history tests."""
    return ProjectData(
        video_path="clip.mp4",
        points=[{"x": frame + 1, "y": 2, "label": 1}],
        keyframes=[frame],
        masks={frame: np.array([[True, False]], dtype=bool)},
        current_frame=frame,
    )


def test_checkpoint_round_trip_uses_project_loader(tmp_path: Path) -> None:
    """Checkpoints remain ordinary project files and expose history metadata."""
    history = ProjectHistory(path=tmp_path / "history")
    timestamp = "2026-01-02T03:04:05+00:00"
    data = _data(4)

    entry = history.create_checkpoint(data, "before experiment", timestamp=timestamp)
    loaded = history.load(entry.version)

    assert entry.version == 1
    assert entry.label == "before experiment"
    assert entry.timestamp == timestamp
    assert entry.path.is_file()
    assert ProjectManager.load(entry.path).current_frame == 4
    assert loaded is not None
    assert loaded.current_frame == data.current_frame
    assert hashlib.sha256(entry.path.read_bytes()).hexdigest() == entry.sha256
    assert entry["version"] == 1


def test_versions_are_monotonic_after_delete_and_new_service(tmp_path: Path) -> None:
    """Deleting the newest entry does not allow version reuse."""
    history = ProjectHistory(path=tmp_path / "history")
    first = history.create_checkpoint(_data(1))
    second = history.create_checkpoint(_data(2))
    assert second.version == first.version + 1

    assert history.delete(second.version) is True
    third = ProjectHistory(path=tmp_path / "history").create_checkpoint(_data(3))
    assert third.version == 3
    assert [item.version for item in history.list()] == [3, 1]


def test_retention_prunes_oldest_versions(tmp_path: Path) -> None:
    """The configured retention limit removes only the oldest checkpoints."""
    history = ProjectHistory(path=tmp_path / "history", max_versions=2)
    first = history.create_checkpoint(_data(1), "one")
    second = history.create_checkpoint(_data(2), "two")
    third = history.create_checkpoint(_data(3), "three")

    entries = history.list()
    assert [item.version for item in entries] == [third.version, second.version]
    assert first.path.exists() is False
    assert first.metadata_path is not None
    assert first.metadata_path.exists() is False
    assert second.path.exists() is True


def test_corrupt_and_missing_entries_are_safe(tmp_path: Path) -> None:
    """Invalid sidecars, payloads, and versions are ignored or return None."""
    history = ProjectHistory(path=tmp_path / "history")
    entry = history.create_checkpoint(_data(1), "valid")

    assert history.load(999) is None
    assert history.delete(999) is False
    entry.path.write_text("{broken", encoding="utf-8")
    assert history.list() == []
    assert history.load(entry.version) is None
    assert history.delete(entry.version) is True
    assert history.list() == []
    history.clear()
    assert history.list() == []


def test_metadata_tampering_invalidates_checkpoint(tmp_path: Path) -> None:
    """Changing payload bytes without updating metadata fails verification."""
    history = ProjectHistory(path=tmp_path / "history")
    entry = history.create_checkpoint(_data(1))
    entry.path.write_bytes(entry.path.read_bytes() + b" ")

    assert history.verify(entry.version) is False
    assert history.list() == []


def test_default_directory_is_project_local_and_root_is_injectable(
    tmp_path: Path,
) -> None:
    """History follows the project path and supports a test root."""
    project_path = tmp_path / "project.bbproj"
    assert default_history_dir(project_path).parent == tmp_path

    history = ProjectHistory("project.bbproj", root=tmp_path)
    assert history.history_dir.parent == tmp_path
    assert history.history_dir == default_history_dir(project_path)


def test_restore_returns_data_and_can_write_destination(tmp_path: Path) -> None:
    """Restore is non-mutating by default and can serialize a target file."""
    history = ProjectHistory(path=tmp_path / "history")
    entry = history.create_checkpoint(_data(7), "restore me")

    restored = history.restore(entry.version)
    assert isinstance(restored, ProjectData)
    assert restored.current_frame == 7

    destination = tmp_path / "restored.bbproj"
    result = history.restore_to(entry.version, destination)
    assert result == destination
    assert ProjectManager.load(destination).current_frame == 7


def test_metadata_sidecar_contains_version_and_timestamp(tmp_path: Path) -> None:
    """The sidecar is useful to integrations that need raw metadata."""
    history = ProjectHistory(path=tmp_path / "history")
    entry = history.create_checkpoint(_data(), label="named")
    assert entry.metadata_path is not None
    metadata = json.loads(entry.metadata_path.read_text(encoding="utf-8"))
    assert metadata["version"] == entry.version
    assert metadata["label"] == "named"
    assert metadata["created_at"] == entry.created_at
    assert metadata["sha256"] == entry.sha256
