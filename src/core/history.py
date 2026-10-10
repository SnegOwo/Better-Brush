"""Versioned, project-local history for :class:`ProjectData` snapshots.

The checkpoint payloads in this module are ordinary Better Brush project files.
They are serialized through ``ProjectManager.save`` and read back through
``ProjectManager.load`` so history does not create a second snapshot format.
Small JSON sidecars contain labels, timestamps, versions, sizes, and SHA-256
checksums.  Every durable file is written with a same-directory temporary file
and ``os.replace`` so readers never observe a partially written checkpoint.

The public service is :class:`ProjectHistory`.  It can keep history beside a
saved project, below an explicitly supplied directory, or below an injected
root for tests and headless callers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from src.core.project import ProjectData, ProjectManager

logger = logging.getLogger(__name__)

HISTORY_FORMAT = "betterbrush-project-history"
HISTORY_FORMAT_VERSION = 1
DEFAULT_MAX_VERSIONS = 20
DEFAULT_HISTORY_DIR_NAME = ".betterbrush-history"
HISTORY_DIR_SUFFIX = ".history"
CHECKPOINT_SUFFIX = ".bbproj"
METADATA_SUFFIX = ".meta.json"
STATE_FILENAME = ".history-state.json"

_VERSION_NAME = re.compile(r"^(?:v|checkpoint[-_]?)?(?P<version>\d+)$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_PROJECT_LOAD_ERRORS = (
    OSError,
    UnicodeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    OverflowError,
    RuntimeError,
)
_LOCKS_GUARD = Lock()
_HISTORY_LOCKS: dict[str, RLock] = {}


def _directory_lock(path: Path) -> RLock:
    """Return a process-local lock for one history directory."""
    key = os.path.normcase(str(path.absolute()))
    with _LOCKS_GUARD:
        lock = _HISTORY_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _HISTORY_LOCKS[key] = lock
        return lock


def _as_path(value: str | Path) -> Path:
    """Convert a path-like value to a path without resolving missing paths."""
    return Path(value).expanduser()


def _apply_root(value: str | Path, root: str | Path | None) -> Path:
    """Apply an optional root to a relative path."""
    path = _as_path(value)
    if root is not None and not path.is_absolute():
        return _as_path(root) / path
    return path


def default_history_dir(
    project_path: str | Path | None = None,
    root: str | Path | None = None,
) -> Path:
    """Return the default project-local history directory.

    A saved project uses a sibling directory named ``.<project-file>.history``.
    An unsaved project, or a path that currently names a directory, uses a
    ``.betterbrush-history`` directory below its parent.  ``root`` selects the
    parent for injected storage, including when the project path is absolute.
    """
    if project_path is None:
        base = _as_path(root) if root is not None else Path.cwd()
        return base / DEFAULT_HISTORY_DIR_NAME

    candidate = _as_path(project_path)
    if root is not None:
        base = _as_path(root)
        candidate = base / (candidate.name if candidate.is_absolute() else candidate)
    if candidate.is_dir():
        return candidate / DEFAULT_HISTORY_DIR_NAME
    return candidate.with_name(f".{candidate.name}{HISTORY_DIR_SUFFIX}")


def get_history_dir(
    project_path: str | Path | None = None,
    root: str | Path | None = None,
    *,
    path: str | Path | None = None,
) -> Path:
    """Resolve a project-local or explicitly injected history directory."""
    if path is None:
        return default_history_dir(project_path, root)
    candidate = _apply_root(path, root)
    return candidate


def history_dir_for(
    project_path: str | Path | None = None,
    root: str | Path | None = None,
    *,
    path: str | Path | None = None,
) -> Path:
    """Alias for :func:`get_history_dir`."""
    return get_history_dir(project_path, root, path=path)


def _timestamp(value: str | datetime | None) -> str:
    """Normalize an optional checkpoint timestamp to an ISO string."""
    if value is None:
        return datetime.now(UTC).isoformat(timespec="microseconds")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("timestamp must be a non-empty string or datetime")


def _valid_timestamp(value: str) -> bool:
    """Return whether a metadata timestamp is parseable as ISO-8601."""
    try:
        datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize metadata as deterministic UTF-8 JSON."""
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    """Best-effort durability sync for a directory after replacement."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes) -> None:
    """Atomically replace a file with bytes from the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _positive_integer(value: Any) -> int | None:
    """Parse a positive integer without accepting booleans or floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _nonnegative_integer(value: Any) -> int | None:
    """Parse a nonnegative integer without accepting booleans or floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _version_from_name(path: Path) -> int | None:
    """Extract a version from a checkpoint filename when present."""
    if path.suffix.lower() != CHECKPOINT_SUFFIX:
        return None
    match = _VERSION_NAME.fullmatch(path.stem)
    if match is None:
        return None
    return _positive_integer(match.group("version"))


def _metadata_path(checkpoint_path: Path) -> Path:
    """Return the sidecar path associated with a checkpoint."""
    return checkpoint_path.with_name(f"{checkpoint_path.stem}{METADATA_SUFFIX}")


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` for missing or corrupt data."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _remove_file(path: Path) -> bool:
    """Remove one file if it exists, returning whether it disappeared."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not remove history file %s", path, exc_info=True)
        return False
    return not path.exists()


@dataclass(frozen=True, eq=False)
class HistoryEntry(Mapping[str, object]):
    """Metadata for one verified project-history checkpoint."""

    version: int
    path: Path
    label: str = "Checkpoint"
    timestamp: str = ""
    sha256: str = ""
    size: int = 0
    metadata_path: Path | None = None
    valid: bool = True

    @property
    def history_version(self) -> int:
        """Return the version using the descriptive history spelling."""
        return self.version

    @property
    def created_at(self) -> str:
        """Return the checkpoint creation timestamp."""
        return self.timestamp

    @property
    def checksum(self) -> str:
        """Return the SHA-256 checksum under its alternate name."""
        return self.sha256

    @property
    def checkpoint_path(self) -> Path:
        """Return the project payload path."""
        return self.path

    @property
    def name(self) -> str:
        """Return the checkpoint filename."""
        return self.path.name

    @property
    def suffix(self) -> str:
        """Return the checkpoint filename suffix."""
        return self.path.suffix

    def exists(self) -> bool:
        """Return whether the checkpoint file still exists."""
        return self.path.is_file()

    def is_file(self) -> bool:
        """Return whether the checkpoint path is a regular file."""
        return self.path.is_file()

    def read_bytes(self) -> bytes:
        """Read the underlying project payload bytes."""
        return self.path.read_bytes()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the entry."""
        return {
            "version": self.version,
            "history_version": self.version,
            "label": self.label,
            "timestamp": self.timestamp,
            "created_at": self.timestamp,
            "sha256": self.sha256,
            "checksum": self.sha256,
            "size": self.size,
            "path": str(self.path),
            "metadata_path": str(self.metadata_path)
            if self.metadata_path is not None
            else None,
            "valid": self.valid,
        }

    def __getitem__(self, key: str) -> object:
        """Provide mapping-style access for UI and test integrations."""
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over mapping keys."""
        return iter(self.as_dict())

    def __len__(self) -> int:
        """Return the number of public metadata fields."""
        return len(self.as_dict())

    def __int__(self) -> int:
        """Allow an entry to be used as its numeric history version."""
        return self.version

    def __index__(self) -> int:
        """Allow an entry to be used where an integer index is expected."""
        return self.version

    def __fspath__(self) -> str:
        """Expose the entry as a path-like checkpoint object."""
        return str(self.path)

    def __str__(self) -> str:
        """Return the underlying checkpoint path."""
        return str(self.path)

    def __eq__(self, other: object) -> bool:
        """Compare entries structurally or compare directly with a version."""
        if isinstance(other, HistoryEntry):
            return self.as_dict() == other.as_dict()
        if isinstance(other, int) and not isinstance(other, bool):
            return self.version == other
        return NotImplemented

    def __hash__(self) -> int:
        """Return a stable hash for immutable entry objects."""
        return hash(
            (
                self.version,
                self.path,
                self.label,
                self.timestamp,
                self.sha256,
                self.size,
                self.metadata_path,
                self.valid,
            )
        )


class ProjectHistory:
    """Store, inspect, prune, and restore versioned project snapshots.

    The checkpoint payload is always a regular ``.bbproj`` file produced by
    ``ProjectManager.save``.  A ``.meta.json`` sidecar supplies the history
    metadata and stores the checksum of the exact payload bytes.  A small
    state file records the next version so deleting the newest entry does not
    cause a later save to reuse its version.
    """

    def __init__(
        self,
        project_path: str | Path | None = None,
        history_dir: str | Path | None = None,
        max_versions: int = DEFAULT_MAX_VERSIONS,
        *,
        root: str | Path | None = None,
        root_path: str | Path | None = None,
        path: str | Path | None = None,
        history_path: str | Path | None = None,
        retention: int | None = None,
        max_history: int | None = None,
        max_checkpoints: int | None = None,
        history_root: str | Path | None = None,
    ) -> None:
        """Create a history service without requiring an existing history."""
        explicit_directories = [
            value for value in (history_dir, path, history_path) if value is not None
        ]
        if len(explicit_directories) > 1:
            raise TypeError("pass only one history directory argument")
        explicit_roots = [
            value for value in (root, root_path, history_root) if value is not None
        ]
        if len(explicit_roots) > 1 and any(
            _as_path(value) != _as_path(explicit_roots[0])
            for value in explicit_roots[1:]
        ):
            raise ValueError("root arguments must refer to the same path")

        effective_root = explicit_roots[0] if explicit_roots else None
        explicit_directory = explicit_directories[0] if explicit_directories else None
        self.root = _as_path(effective_root) if effective_root is not None else None
        self.project_path = (
            _apply_root(project_path, effective_root)
            if project_path is not None
            else None
        )
        if explicit_directory is not None:
            selected = _apply_root(explicit_directory, effective_root)
        else:
            selected = default_history_dir(project_path, effective_root)
        self.history_dir = selected
        self._lock = _directory_lock(self.history_dir)

        limits = [
            value
            for value in (retention, max_history, max_checkpoints)
            if value is not None
        ]
        if limits and any(value != limits[0] for value in limits[1:]):
            raise ValueError("retention limits must agree")
        configured_limit = limits[0] if limits else max_versions
        if isinstance(configured_limit, bool) or not isinstance(configured_limit, int):
            raise TypeError("max_versions must be an integer")
        if configured_limit < 1:
            raise ValueError("max_versions must be at least one")
        self.max_versions = configured_limit

    @property
    def directory(self) -> Path:
        """Return the directory containing checkpoints and sidecars."""
        return self.history_dir

    @property
    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory."""
        return self.history_dir

    @property
    def path(self) -> Path:
        """Return the history directory using a short path alias."""
        return self.history_dir

    @property
    def retention(self) -> int:
        """Return the maximum number of retained versions."""
        return self.max_versions

    @property
    def state_path(self) -> Path:
        """Return the persisted monotonic version-state path."""
        return self.history_dir / STATE_FILENAME

    @classmethod
    def for_project(
        cls,
        project_path: str | Path | None = None,
        **options: Any,
    ) -> ProjectHistory:
        """Return a service for another project using the same conventions."""
        return cls(project_path, **options)

    def _ensure_directory(self) -> None:
        """Create the history directory when a write is requested."""
        if self.history_dir.exists() and not self.history_dir.is_dir():
            raise NotADirectoryError(str(self.history_dir))
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _files(self) -> list[Path]:
        """Return regular files currently present in the history directory."""
        try:
            if not self.history_dir.is_dir():
                return []
            return sorted(path for path in self.history_dir.iterdir() if path.is_file())
        except OSError:
            return []

    def _checkpoint_path(self, version: int) -> Path:
        """Return the canonical filename for a version."""
        return self.history_dir / f"v{version:08d}{CHECKPOINT_SUFFIX}"

    def _metadata_path_for(self, checkpoint: Path) -> Path:
        """Return the sidecar path for a checkpoint filename."""
        return _metadata_path(checkpoint)

    def checkpoint_path(self, version: int) -> Path:
        """Return the canonical checkpoint path without creating it."""
        parsed = _positive_integer(version)
        if parsed is None:
            raise ValueError("version must be a positive integer")
        return self._checkpoint_path(parsed)

    def metadata_path(self, version: int) -> Path:
        """Return the canonical sidecar path without creating it."""
        return _metadata_path(self.checkpoint_path(version))

    def _read_state_next(self) -> int:
        """Read the next persisted version, ignoring corrupt state."""
        state = _read_json_object(self.state_path)
        if state is None:
            return 1
        value = _positive_integer(state.get("next_version"))
        return value if value is not None else 1

    def _write_state(self, next_version: int, timestamp: str) -> None:
        """Atomically persist the next monotonic version."""
        _atomic_write(
            self.state_path,
            _json_bytes(
                {
                    "format": HISTORY_FORMAT,
                    "format_version": HISTORY_FORMAT_VERSION,
                    "next_version": next_version,
                    "updated_at": timestamp,
                }
            ),
        )

    def _version_from_file(self, path: Path) -> int | None:
        """Return a version inferred from a filename or sidecar."""
        version = _version_from_name(path)
        if version is not None:
            return version
        if path.name.endswith(METADATA_SUFFIX):
            metadata = _read_json_object(path)
            if metadata is not None:
                return _positive_integer(
                    metadata.get("version", metadata.get("history_version"))
                )
        if path.suffix.lower() == CHECKPOINT_SUFFIX:
            metadata = _read_json_object(self._metadata_path_for(path))
            if metadata is not None:
                return _positive_integer(
                    metadata.get("version", metadata.get("history_version"))
                )
        return None

    def _version_groups(self) -> dict[int, set[Path]]:
        """Group checkpoint and sidecar files by discovered version."""
        groups: dict[int, set[Path]] = {}
        for path in self._files():
            version = self._version_from_file(path)
            if version is None:
                continue
            if path.suffix.lower() == CHECKPOINT_SUFFIX:
                groups.setdefault(version, set()).add(path)
                sidecar = self._metadata_path_for(path)
                if sidecar in self._files():
                    groups[version].add(sidecar)
            elif path.name.endswith(METADATA_SUFFIX):
                groups.setdefault(version, set()).add(path)
        return groups

    def _next_version(self) -> int:
        """Choose a version greater than every persisted or discovered one."""
        state_next = self._read_state_next()
        discovered = max(self._version_groups(), default=0) + 1
        version = max(1, state_next, discovered)
        while (
            self._checkpoint_path(version).exists()
            or self._metadata_path_for(self._checkpoint_path(version)).exists()
        ):
            version += 1
        return version

    def _serialize_snapshot(self, data: ProjectData) -> bytes:
        """Serialize a ProjectData snapshot through the existing project API."""
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".snapshot-",
            suffix=CHECKPOINT_SUFFIX,
            dir=str(self.history_dir),
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            ProjectManager.save(temporary_path, data)
            return temporary_path.read_bytes()
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _entry_from_checkpoint(
        self,
        checkpoint: Path,
        *,
        validate_project: bool = True,
    ) -> HistoryEntry | None:
        """Build a verified entry or return ``None`` for unsafe data."""
        try:
            if not checkpoint.is_file():
                return None
            sidecar = self._metadata_path_for(checkpoint)
            metadata = _read_json_object(sidecar)
            if metadata is None:
                return None

            format_name = metadata.get("format")
            if format_name is not None and format_name != HISTORY_FORMAT:
                return None
            format_version = metadata.get("format_version")
            if format_version is not None and format_version != HISTORY_FORMAT_VERSION:
                return None

            filename_version = _version_from_name(checkpoint)
            metadata_version = _positive_integer(
                metadata.get("version", metadata.get("history_version"))
            )
            if (
                filename_version is not None
                and metadata_version is not None
                and filename_version != metadata_version
            ):
                return None
            version = metadata_version or filename_version
            if version is None:
                return None

            checksum = metadata.get("sha256", metadata.get("checksum"))
            if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
                return None
            raw = checkpoint.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if not hmac.compare_digest(actual, checksum.lower()):
                return None

            size = metadata.get("size", metadata.get("file_size"))
            if size is not None:
                parsed_size = _nonnegative_integer(size)
                if parsed_size is None or parsed_size != len(raw):
                    return None

            label = metadata.get("label", "Checkpoint")
            if not isinstance(label, str):
                return None
            timestamp = metadata.get("created_at", metadata.get("timestamp"))
            if not isinstance(timestamp, str) or not _valid_timestamp(timestamp):
                return None

            if validate_project:
                try:
                    ProjectManager.load(checkpoint)
                except _PROJECT_LOAD_ERRORS:
                    logger.debug("Ignoring invalid project checkpoint %s", checkpoint)
                    return None

            return HistoryEntry(
                version=version,
                path=checkpoint,
                label=label,
                timestamp=timestamp,
                sha256=actual,
                size=len(raw),
                metadata_path=sidecar,
            )
        except _PROJECT_LOAD_ERRORS:
            logger.debug("Ignoring unreadable history checkpoint %s", checkpoint)
            return None

    def _entries(self, newest_first: bool = True) -> list[HistoryEntry]:
        """Return verified entries in version order."""
        entries: list[HistoryEntry] = []
        for path in self._files():
            if path.suffix.lower() != CHECKPOINT_SUFFIX:
                continue
            entry = self._entry_from_checkpoint(path)
            if entry is not None:
                entries.append(entry)
        entries.sort(key=lambda item: (item.version, item.timestamp, str(item.path)))
        if newest_first:
            entries.reverse()
        return entries

    def _requested_version(self, value: Any) -> int | None:
        """Normalize a version argument without trusting path input."""
        if isinstance(value, HistoryEntry):
            return value.version
        if isinstance(value, Path):
            version = self._version_from_file(value)
            if version is not None:
                return version
            return None
        if isinstance(value, str):
            return _positive_integer(value)
        return _positive_integer(value)

    def _paths_for_version(self, version: int) -> set[Path]:
        """Return all files associated with one history version."""
        paths: set[Path] = set()
        files = self._files()
        for path in files:
            if self._version_from_file(path) == version:
                paths.add(path)
        return paths

    def _entry_for_version(self, value: Any) -> HistoryEntry | None:
        """Find one verified entry by version or latest-entry sentinel."""
        if value is None:
            entries = self._entries(newest_first=True)
            return entries[0] if entries else None
        version = self._requested_version(value)
        if version is None:
            return None
        candidates = sorted(
            (
                path
                for path in self._files()
                if path.suffix.lower() == CHECKPOINT_SUFFIX
                and self._version_from_file(path) == version
            ),
            key=lambda path: str(path),
        )
        for path in candidates:
            entry = self._entry_from_checkpoint(path)
            if entry is not None:
                return entry
        return None

    def _remove_paths(self, paths: set[Path] | list[Path]) -> int:
        """Remove associated files and return the checkpoint-file count."""
        removed = 0
        for path in sorted(paths, key=lambda item: str(item), reverse=True):
            if not path.exists():
                continue
            is_checkpoint = path.suffix.lower() == CHECKPOINT_SUFFIX
            if _remove_file(path) and is_checkpoint:
                removed += 1
        return removed

    def create_checkpoint(
        self,
        data: ProjectData,
        label: str | None = None,
        *,
        timestamp: str | datetime | None = None,
        created_at: str | datetime | None = None,
    ) -> HistoryEntry:
        """Atomically save a new labeled ProjectData checkpoint.

        The returned entry contains the assigned version, path, timestamp, and
        checksum.  A failed metadata or serialization operation never leaves a
        new usable checkpoint behind.
        """
        if timestamp is not None and created_at is not None:
            raise TypeError("pass either timestamp or created_at, not both")
        selected_timestamp = _timestamp(
            timestamp if timestamp is not None else created_at
        )
        selected_label = "Checkpoint" if label is None else str(label)

        with self._lock:
            self._ensure_directory()
            version = self._next_version()
            checkpoint = self._checkpoint_path(version)
            sidecar = self._metadata_path_for(checkpoint)
            raw = self._serialize_snapshot(data)
            checksum = hashlib.sha256(raw).hexdigest()
            metadata = {
                "format": HISTORY_FORMAT,
                "format_version": HISTORY_FORMAT_VERSION,
                "version": version,
                "history_version": version,
                "label": selected_label,
                "timestamp": selected_timestamp,
                "created_at": selected_timestamp,
                "sha256": checksum,
                "size": len(raw),
            }
            committed = False
            try:
                _atomic_write(checkpoint, raw)
                try:
                    _atomic_write(sidecar, _json_bytes(metadata))
                except Exception:
                    _remove_file(checkpoint)
                    raise
                committed = True
                self._write_state(version + 1, selected_timestamp)
            except Exception:
                if not committed:
                    _remove_file(sidecar)
                    _remove_file(checkpoint)
                raise

            self._prune_locked(self.max_versions)
            return HistoryEntry(
                version=version,
                path=checkpoint,
                label=selected_label,
                timestamp=selected_timestamp,
                sha256=checksum,
                size=len(raw),
                metadata_path=sidecar,
            )

    def create(
        self,
        data: ProjectData,
        label: str | None = None,
        *,
        timestamp: str | datetime | None = None,
        created_at: str | datetime | None = None,
    ) -> HistoryEntry:
        """Create a labeled checkpoint."""
        return self.create_checkpoint(
            data,
            label,
            timestamp=timestamp,
            created_at=created_at,
        )

    def save_checkpoint(
        self,
        data: ProjectData,
        label: str | None = None,
        *,
        timestamp: str | datetime | None = None,
        created_at: str | datetime | None = None,
    ) -> HistoryEntry:
        """Alias for :meth:`create_checkpoint`."""
        return self.create_checkpoint(
            data,
            label,
            timestamp=timestamp,
            created_at=created_at,
        )

    def checkpoint(
        self,
        data: ProjectData,
        label: str | None = None,
        *,
        timestamp: str | datetime | None = None,
        created_at: str | datetime | None = None,
    ) -> HistoryEntry:
        """Alias for :meth:`create_checkpoint`."""
        return self.create_checkpoint(
            data,
            label,
            timestamp=timestamp,
            created_at=created_at,
        )

    def save(
        self,
        data: ProjectData,
        label: str | None = None,
        *,
        timestamp: str | datetime | None = None,
        created_at: str | datetime | None = None,
    ) -> HistoryEntry:
        """Save a checkpoint and return its history entry."""
        return self.create_checkpoint(
            data,
            label,
            timestamp=timestamp,
            created_at=created_at,
        )

    def save_snapshot(
        self,
        data: ProjectData,
        label: str | None = None,
        *,
        timestamp: str | datetime | None = None,
        created_at: str | datetime | None = None,
    ) -> HistoryEntry:
        """Alias for :meth:`create_checkpoint`."""
        return self.create_checkpoint(
            data,
            label,
            timestamp=timestamp,
            created_at=created_at,
        )

    def record(
        self,
        data: ProjectData,
        label: str | None = None,
        *,
        timestamp: str | datetime | None = None,
        created_at: str | datetime | None = None,
    ) -> HistoryEntry:
        """Record a checkpoint using the service's default label when omitted."""
        return self.create_checkpoint(
            data,
            label,
            timestamp=timestamp,
            created_at=created_at,
        )

    def list(
        self,
        newest_first: bool = True,
        *,
        ascending: bool | None = None,
    ) -> list[HistoryEntry]:
        """List verified checkpoints, newest version first by default."""
        if ascending is not None:
            newest_first = not ascending
        with self._lock:
            return self._entries(newest_first=newest_first)

    def list_checkpoints(
        self,
        newest_first: bool = True,
        *,
        ascending: bool | None = None,
    ) -> list[HistoryEntry]:
        """List verified checkpoints using the explicit service API name."""
        return self.list(newest_first=newest_first, ascending=ascending)

    def list_snapshots(
        self,
        newest_first: bool = True,
        *,
        ascending: bool | None = None,
    ) -> list[HistoryEntry]:
        """Alias for :meth:`list`."""
        return self.list(newest_first=newest_first, ascending=ascending)

    def entries(
        self,
        newest_first: bool = True,
        *,
        ascending: bool | None = None,
    ) -> list[HistoryEntry]:
        """Return verified history entries in version order."""
        return self.list(newest_first=newest_first, ascending=ascending)

    def list_oldest_first(self) -> list[HistoryEntry]:
        """List verified checkpoints from oldest to newest."""
        return self.list(newest_first=False)

    def latest(self) -> HistoryEntry | None:
        """Return the newest verified checkpoint, if one exists."""
        with self._lock:
            entries = self._entries(newest_first=True)
            return entries[0] if entries else None

    @property
    def latest_entry(self) -> HistoryEntry | None:
        """Return the newest verified checkpoint as a property."""
        return self.latest()

    def get_entry(self, version: Any) -> HistoryEntry | None:
        """Return metadata for one verified checkpoint without restoring it."""
        with self._lock:
            return self._entry_for_version(version)

    def metadata(self, version: Any) -> dict[str, object] | None:
        """Return mapping-style metadata for one verified checkpoint."""
        entry = self.get_entry(version)
        return entry.as_dict() if entry is not None else None

    def load(self, version: Any = None) -> ProjectData | None:
        """Load a verified snapshot through ``ProjectManager.load``.

        Missing, corrupt, checksum-mismatched, and otherwise invalid entries
        return ``None`` instead of escaping parser or filesystem exceptions.
        Passing no version loads the newest verified checkpoint.
        """
        with self._lock:
            entry = self._entry_for_version(version)
            if entry is None:
                return None
            try:
                return ProjectManager.load(entry.path)
            except _PROJECT_LOAD_ERRORS:
                logger.debug("Could not load history checkpoint %s", entry.path)
                return None

    def load_checkpoint(self, version: Any = None) -> ProjectData | None:
        """Alias for :meth:`load`."""
        return self.load(version)

    def load_snapshot(self, version: Any = None) -> ProjectData | None:
        """Alias for :meth:`load`."""
        return self.load(version)

    def restore(
        self,
        version: Any = None,
        *,
        destination: str | Path | None = None,
        target: str | Path | None = None,
    ) -> ProjectData | Path | None:
        """Return a snapshot, optionally writing it to a project path.

        With no destination, restore is deliberately non-mutating and returns
        the loaded ``ProjectData`` for the caller's existing project-state
        integration.  With a destination, the verified snapshot is serialized
        there through ``ProjectManager.save`` and the canonical path is
        returned.
        """
        if destination is not None and target is not None:
            raise TypeError("pass either destination or target, not both")
        data = self.load(version)
        if data is None:
            return None
        selected_destination = destination if destination is not None else target
        if selected_destination is None:
            return data
        destination_path = _apply_root(selected_destination, self.root)
        if destination_path.is_dir():
            filename = (
                self.project_path.name if self.project_path else "restored.bbproj"
            )
            destination_path = destination_path / filename
        return ProjectManager.save(destination_path, data)

    def restore_checkpoint(
        self,
        version: Any = None,
        *,
        destination: str | Path | None = None,
        target: str | Path | None = None,
    ) -> ProjectData | Path | None:
        """Alias for :meth:`restore`."""
        return self.restore(version, destination=destination, target=target)

    def restore_to(self, version: Any, destination: str | Path) -> Path | None:
        """Restore a verified checkpoint to a project file path."""
        restored = self.restore(version, destination=destination)
        return restored if isinstance(restored, Path) else None

    def delete(self, version: Any) -> bool:
        """Delete one checkpoint and its sidecar, returning whether work was done."""
        with self._lock:
            parsed = self._requested_version(version)
            if parsed is None:
                return False
            paths = self._paths_for_version(parsed)
            return self._remove_paths(paths) > 0

    def delete_checkpoint(self, version: Any) -> bool:
        """Alias for :meth:`delete`."""
        return self.delete(version)

    def remove(self, version: Any) -> bool:
        """Alias for :meth:`delete`."""
        return self.delete(version)

    def clear(self) -> int:
        """Remove all history-managed files and return the number removed."""
        with self._lock:
            removed = 0
            for path in self._files():
                name = path.name
                managed = (
                    path.suffix.lower() == CHECKPOINT_SUFFIX
                    or name.endswith(METADATA_SUFFIX)
                    or name == STATE_FILENAME
                    or name.startswith(
                        (f".{STATE_FILENAME}.", ".snapshot-", f".{CHECKPOINT_SUFFIX}")
                    )
                    or (name.startswith(".") and name.endswith(".tmp"))
                )
                if managed and _remove_file(path):
                    removed += 1
            return removed

    def clear_history(self) -> int:
        """Alias for :meth:`clear`."""
        return self.clear()

    def prune(self, max_versions: int | None = None) -> int:
        """Remove oldest versions and return the number of checkpoints removed."""
        limit = self.max_versions if max_versions is None else max_versions
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("max_versions must be an integer")
        if limit < 1:
            raise ValueError("max_versions must be at least one")
        with self._lock:
            return self._prune_locked(limit)

    def apply_retention(self, max_versions: int | None = None) -> int:
        """Apply the configured or supplied retention limit."""
        return self.prune(max_versions=max_versions)

    def enforce_retention(self, max_versions: int | None = None) -> int:
        """Alias for :meth:`prune`."""
        return self.prune(max_versions=max_versions)

    def _prune_locked(self, limit: int) -> int:
        """Prune while the service lock is already held."""
        groups = self._version_groups()
        ordered = sorted(groups)
        if len(ordered) <= limit:
            return 0
        removed = 0
        for version in ordered[: len(ordered) - limit]:
            removed += self._remove_paths(groups[version])
        return removed

    def verify(self, version: Any) -> bool:
        """Return whether one version has valid metadata and payload integrity."""
        return self.get_entry(version) is not None

    def check_integrity(self, version: Any) -> bool:
        """Alias for :meth:`verify`."""
        return self.verify(version)

    list_history = list
    load_version = load
    delete_version = delete
    restore_version = restore
    prune_history = prune
    add_checkpoint = create_checkpoint

    def __len__(self) -> int:
        """Return the number of verified checkpoints."""
        return len(self.list())


HistorySnapshot = HistoryEntry
ProjectHistoryManager = ProjectHistory
HistoryService = ProjectHistory
HistoryManager = ProjectHistory

__all__ = [
    "CHECKPOINT_SUFFIX",
    "DEFAULT_HISTORY_DIR_NAME",
    "DEFAULT_MAX_VERSIONS",
    "HISTORY_DIR_SUFFIX",
    "HISTORY_FORMAT",
    "HISTORY_FORMAT_VERSION",
    "METADATA_SUFFIX",
    "STATE_FILENAME",
    "HistoryEntry",
    "HistoryManager",
    "HistoryService",
    "HistorySnapshot",
    "ProjectHistory",
    "ProjectHistoryManager",
    "default_history_dir",
    "get_history_dir",
    "history_dir_for",
]
