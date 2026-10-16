"""Launch child processes without ever flashing a console window.

Better Brush runs under ``pythonw.exe`` — the panel's engine and the desktop app
both do — so the child has no console of its own to hand down. On Windows a
console-subsystem child (ffmpeg above all) then allocates a fresh one, and a
black window pops up over After Effects in the middle of a job. That is ugly
enough on its own, but the window also belongs to a *new* console that can
outlive the job, so the child is started hidden instead.

``CREATE_NO_WINDOW`` is Windows-only; elsewhere these functions cost nothing.
"""

from __future__ import annotations

import subprocess
import sys

#: ``subprocess.CREATE_NO_WINDOW`` (0 elsewhere, which is also a valid value).
CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def no_window() -> dict[str, int]:
    """Keyword arguments that keep a ``subprocess`` child from popping up.

    Spread into any ``Popen`` / ``run`` call::

        subprocess.Popen(cmd, **no_window())

    Never returns ``creationflags`` on platforms (or interpreters) that have no
    ``CREATE_NO_WINDOW``, so callers stay portable.
    """
    if sys.platform == "win32" and CREATE_NO_WINDOW:
        return {"creationflags": CREATE_NO_WINDOW}
    return {}
