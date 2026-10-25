"""Open a clip in the Better Brush desktop app.

A ScriptUI panel cannot start a process — this EffectCraft build's
``system.callSystem`` throws and ``File.execute`` only opens a file — so the
panel's "Open in Better Brush" button is a bridge message like everything
else:

    panel  ->  app-open-<id>.json       {id, source, created}
    adapter -> launches BetterBrush.exe <source>
    adapter ->  app-open-status-<id>.json  {state, exe, source, message}

The exe is remembered in ``app-path.json`` in the bridge. It gets there from
one of two places, and this module writes it back after a successful search —
which is what makes "point at it once" true:

* the user pastes a path into the panel's field and clicks Save;
* the adapter finds it in a well-known install folder or next to the repo's
  own builds.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

APP_PATH_FILE = "app-path.json"
APP_OPEN_PREFIX = "app-open-"
APP_STATUS_PREFIX = "app-open-status-"

APP_EXE_NAME = "BetterBrush.exe"

#: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``: the app gets its own
#: process group with no console, so closing the adapter (or the panel) can
#: never take it down. Ignored off Windows.
_DETACHED = 0x00000008 | 0x00000200

Log = Callable[[str], None]


# --------------------------------------------------------------------- #
# Where the app might be
# --------------------------------------------------------------------- #
def _repo_root() -> Path | None:
    """The checkout this adapter lives in, when it can find one.

    Frozen adapters (``sys.frozen``) have no ``__file__`` worth walking, so
    ``BETTERBRUSH_ROOT`` names the checkout for them.
    """
    override = os.environ.get("BETTERBRUSH_ROOT")
    if override:
        path = Path(override).expanduser()
        return path if path.is_dir() else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return None


def candidate_exes() -> list[Path]:
    """Places worth looking, most specific first. Existence is not checked."""
    out: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    for base in (local, program_files, program_files_x86):
        if base:
            out.append(Path(base) / "Programs" / "BetterBrush" / APP_EXE_NAME)
            out.append(Path(base) / "BetterBrush" / APP_EXE_NAME)
    out.append(Path.home() / "AppData" / "Local" / "Programs" / "BetterBrush" / APP_EXE_NAME)

    root = _repo_root()
    if root is not None:
        # The repo's own PyInstaller builds: dist-final/BetterBrush-1.0.0-
        # Windows/BetterBrush.exe, BetterbrushDistV3/BetterBrush.exe, dist/*.
        for folder in ("dist-final", "BetterbrushDistV3", "dist", "build"):
            base = root / folder
            if not base.is_dir():
                continue
            out.extend(sorted(base.glob(f"**/{APP_EXE_NAME}"), reverse=True))
    return out


def read_recorded_exe(root: Path) -> Path | None:
    """The exe the user (or an earlier search) remembered, if it is still there."""
    path = root / APP_PATH_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    exe = data.get("exe")
    if not isinstance(exe, str) or not exe.strip():
        return None
    found = Path(exe.strip()).expanduser()
    return found if found.is_file() else None


def remember_exe(root: Path, exe: Path) -> None:
    """Record the path so the next request does not have to search again."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / APP_PATH_FILE).write_text(
            json.dumps({"exe": str(exe), "found": time.strftime("%Y-%m-%dT%H:%M:%S")},
                       indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass  # a read-only bridge must not stop the launch from happening


def resolve_exe(root: Path) -> tuple[Path | None, str]:
    """The app to launch: the recorded one, else the first candidate on disk.

    Returns ``(exe, message)``. A found exe is written back to the bridge, so
    the search happens once per machine.
    """
    recorded = read_recorded_exe(root)
    if recorded is not None:
        return recorded, ""
    for candidate in candidate_exes():
        try:
            if candidate.is_file():
                remember_exe(root, candidate)
                return candidate, ""
        except OSError:
            continue
    return None, (
        "Better Brush was not found. Paste the path to "
        f"{APP_EXE_NAME} in the panel's field and click Save."
    )


# --------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------- #
def is_status_file(name: str) -> bool:
    """``app-open-status-x.json`` is an answer, never a request."""
    return name.startswith(APP_STATUS_PREFIX)


def pending_requests(root: Path) -> list[Path]:
    """Request files the panel is waiting on, oldest first."""
    out: list[Path] = []
    for path in sorted(root.glob(f"{APP_OPEN_PREFIX}*.json")):
        if is_status_file(path.name):
            continue
        out.append(path)
    return out


def answer(root: Path, request_id: str, payload: dict[str, Any]) -> None:
    """Write the panel's answer. Atomic, like every other status file here."""
    target = root / f"{APP_STATUS_PREFIX}{request_id}.json"
    tmp = target.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        try:
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass


def _request_id(path: Path) -> str:
    name = path.name[len(APP_OPEN_PREFIX):]
    return name[: -len(".json")] if name.endswith(".json") else name


def launch(exe: Path, clip: str) -> None:
    """Start the app on ``clip``, detached, with no console of its own."""
    kwargs: dict[str, Any] = {
        "cwd": str(exe.parent),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = _DETACHED
    else:  # pragma: no cover - the bridge runs on Windows in the field
        kwargs["start_new_session"] = True
    subprocess.Popen([str(exe), str(clip)], **kwargs)


def handle_requests(root: Path, *, log: Log | None = None) -> int:
    """Serve every pending request. Returns how many were handled.

    Nothing here may raise: a broken request must answer the panel, not kill
    the watcher that is also feeding it jobs.
    """
    note = log or (lambda _message: None)
    handled = 0
    for path in pending_requests(root):
        request_id = _request_id(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            answer(root, request_id, {
                "state": "error",
                "message": f"Unreadable request: {exc}",
            })
            _remove(path)
            continue
        source = raw.get("source") if isinstance(raw, dict) else None
        if not isinstance(source, str) or not source.strip():
            answer(root, request_id, {
                "state": "error",
                "message": "The request carries no clip path.",
            })
            _remove(path)
            continue
        if not Path(source).is_file():
            answer(root, request_id, {
                "state": "error",
                "message": f"The clip is no longer on disk: {source}",
            })
            _remove(path)
            continue

        exe, message = resolve_exe(root)
        if exe is None:
            answer(root, request_id, {"state": "error", "message": message})
            note(f"open in app: {message}")
            _remove(path)
            continue
        try:
            launch(exe, source)
        except OSError as exc:
            answer(root, request_id, {
                "state": "error",
                "message": f"Could not start {exe.name}: {exc}",
            })
            note(f"open in app failed: {exc}")
            _remove(path)
            continue

        answer(root, request_id, {"state": "ok", "exe": str(exe), "source": source})
        note(f"opened {source} in {exe}")
        _remove(path)
        handled += 1
    return handled


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass  # a leftover request is answered; the panel has moved on


def looks_like_exe(value: str) -> bool:
    """The same check the panel makes before it remembers a pasted path.

    ``BetterBrush.exe`` and ``Better Brush 2.exe`` both count; ``python.exe``
    does not.
    """
    return bool(re.search(r"better[\s_-]*brush[^\\/]*\.exe$", value, re.IGNORECASE))
