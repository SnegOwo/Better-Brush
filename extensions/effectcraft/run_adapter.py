#!/usr/bin/env python3
"""Entry point for the EffectCraft adapter.

Kept as a thin shim so the panel's install script and any shell can call

    python extensions/effectcraft/run_adapter.py watch

without knowing where the package lives. It puts this folder on ``sys.path``
first, then the repository root (the real engine's ``src.*`` imports need it).

The same file is the source of the packaged ``betterbrush-adapter.exe``, so
it also handles the case where there is no source tree at all: a frozen
adapter's ``__file__`` lives in PyInstaller's temporary extraction folder,
which has no ``bridge.path.json`` beside it. There the adapter's own folder
— ``sys.executable``'s — is where the installer put one, and
``protocol.candidate_bridge_dirs()`` only understands the environment, so
the recorded path is translated into ``$BETTERBRUSH_BRIDGE_DIR``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
HERE = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def give_the_process_streams() -> None:
    """Attach the null device when this process was started with no console.

    ``start-adapter.vbs`` launches the adapter through wscript — no console —
    and Python then leaves ``sys.stdout`` and ``sys.stderr`` as ``None``.
    ``print()`` silently does nothing in that state (which is why adapter.log
    looked perfectly healthy while jobs failed), but anything that writes to
    the stream itself blows up: SAM 2's tqdm progress bar in the engine's
    propagate stage raised ``'NoneType' object has no attribute 'write'`` and
    took the whole job down with it.

    Pointing both at the null device before anything is imported keeps a
    console-free start identical to a terminal one. The panel reads status
    files, never stdout.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8",
                                    errors="replace"))


give_the_process_streams()

for path in (str(HERE), str(REPO_ROOT)):
    if not FROZEN and path not in sys.path:
        sys.path.insert(0, path)


def _apply_recorded() -> None:
    """Act on the ``bridge.path.json`` the installer wrote beside this copy.

    ``bridge_dir`` (frozen builds, and only when the environment does not say
    otherwise) becomes ``$BETTERBRUSH_BRIDGE_DIR``: a packaged adapter's
    ``__file__`` lives in PyInstaller's temp folder, where no such file
    exists.

    ``repo_root`` goes on ``sys.path`` so ``src.engine`` imports — that is
    the difference between a real matte (engine mode) and a simulated one for
    an adapter installed outside the checkout. ``BETTERBRUSH_ROOT`` beats it,
    so either can be set by hand.
    """
    recorded = HERE / "bridge.path.json"
    try:
        data = json.loads(recorded.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        data = {}

    bridge = data.get("bridge_dir")
    if (FROZEN and not os.environ.get("BETTERBRUSH_BRIDGE_DIR")
            and isinstance(bridge, str) and bridge.strip()):
        os.environ["BETTERBRUSH_BRIDGE_DIR"] = bridge.strip()

    repo = os.environ.get("BETTERBRUSH_ROOT") or data.get("repo_root")
    if isinstance(repo, str) and repo.strip():
        root = Path(repo.strip())
        try:
            if root.is_dir() and str(root) not in sys.path:
                sys.path.insert(0, str(root))
        except OSError:
            pass  # an unreachable path simply does not get added


_apply_recorded()

from effectcraft_bridge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
