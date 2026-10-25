"""Command line for the EffectCraft adapter.

Run from the repository root::

    python extensions/effectcraft/run_adapter.py watch
    python extensions/effectcraft/run_adapter.py once --mode simulate
    python extensions/effectcraft/run_adapter.py doctor

``watch`` is the mode you leave running next to EffectCraft; ``once`` drains
whatever is pending and exits, which is what CI and the tests use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import appbridge
from .commands import build_commands
from .pipeline import execute
from .protocol import (
    BRIDGE_DIRNAME,
    JOB_PREFIX,
    JOB_SUFFIX,
    TERMINAL_STATES,
    Job,
    ProtocolError,
    Status,
    bridge_dir,
    candidate_bridge_dirs,
    new_job_id,
    read_json,
    read_status,
    touch_alive,
    write_json_atomic,
)

LOG_NAME = "adapter.log"


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def log(root: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / LOG_NAME).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # a full disk must not stop the adapter from working


def bridge_dir_or_dirs(explicit: Path | None) -> list[Path]:
    """Every bridge folder to serve, most important first.

    ``explicit`` (``--bridge-dir`` / ``$BETTERBRUSH_BRIDGE_DIR``) means
    exactly that one. Otherwise this is the recorded bridge plus every other
    bridge that exists on this machine.

    That matters because a machine can hold more than one EffectCraft install
    — a portable 0.11.1 next to the 0.17.0 it was upgraded to — and each one
    has its own panel stamped with its own bridge. An adapter that watches
    only the recorded folder leaves the other panel's jobs "queued — waiting
    for the adapter…" forever, with nothing in any log to say why.
    """
    primary = bridge_dir(str(explicit) if explicit else None)
    if explicit:
        return [primary]

    out: list[Path] = [primary]

    def add(path: Path) -> None:
        try:
            if path not in out and path.is_dir():
                out.append(path)
        except OSError:
            pass

    for candidate in candidate_bridge_dirs():
        add(candidate)
    for candidate in portable_bridges():
        add(candidate)
    return out


def portable_bridges(depth: int = 4, roots: list[Path] | None = None) -> list[Path]:
    """Bridges of portable installs near the usual places.

    A portable EffectCraft keeps its settings in ``<install>\\EffectCraftData``
    and people keep the unzipped folders in Downloads/Desktop/Documents. The
    search is bounded on purpose (a few directory listings per level): this
    runs once when the watcher starts, not on every tick.
    """
    if roots is None:
        roots = [
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            Path.home() / "Documents",
        ]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.append(Path(local))
    seen: list[Path] = []
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        for level in range(1, depth + 1):
            pattern = "/".join(["*"] * level) + f"/EffectCraftData/{BRIDGE_DIRNAME}"
            try:
                found = sorted(root.glob(pattern))
            except OSError:
                continue
            for path in found:
                if path not in seen:
                    seen.append(path)
    return seen


def pending_jobs(root: Path) -> list[Path]:
    """Job files with no terminal status yet, oldest first."""
    out: list[Path] = []
    for path in sorted(root.glob(f"{JOB_PREFIX}*{JOB_SUFFIX}")):
        job_id = path.name[len(JOB_PREFIX): -len(JOB_SUFFIX)]
        if not job_id:
            continue
        try:
            status = read_status(root, job_id)
        except ProtocolError:
            # A corrupt status would otherwise block the job forever; treat it
            # as "not finished" so the job is retried and rewritten.
            status = None
        if status is None or status.state not in TERMINAL_STATES:
            out.append(path)
    return out


def load_job(path: Path) -> Job:
    return Job.from_dict(read_json(path))


def _decorator(root: Path, job: Any):
    """Attach the EffectCraft commands to a finished result."""

    def _finish(result: dict[str, Any]) -> dict[str, Any]:
        out_dir = Path(result.get("png_dir", ""))
        try:
            commands = build_commands(job, out_dir)
        except (FileNotFoundError, OSError) as exc:
            # The matte exists but the import cannot be described — still a
            # success for the render, and the message says what is missing.
            result["command_error"] = str(exc)
            return result
        result["commands"] = commands
        if job.correction is not None and job.correction.replace_item:
            # Correction: the matte already sits above the source, so nothing
            # moves — the source keeps the index the panel recorded.
            result["matte_index"] = max(1, job.matte_index - 1)
            result["source_index"] = job.matte_index
            result["corrected"] = True
        else:
            result["matte_index"] = job.matte_index
            result["source_index"] = job.matte_index + 1
        result["matte_name"] = "BetterBrush Matte"
        params = commands[0]["params"]
        head = params.get("paths") or [params.get("path")]
        result["first_frame"] = str(head[0])
        return result

    return _finish


def run_one(root: Path, path: Path, mode: str | None) -> Status | None:
    try:
        job = load_job(path)
    except ProtocolError as exc:
        # An unreadable job must still get a terminal status, otherwise the
        # watcher retries it forever and the panel waits forever.
        job_id = path.name[len(JOB_PREFIX): -len(JOB_SUFFIX)]
        status = Status(id=job_id, state="error",
                        error={"kind": "bad_request", "message": str(exc)},
                        message=str(exc))
        write_json_atomic(root / f"status-{job_id}.json", status.to_dict())
        log(root, f"rejected {path.name}: {exc}")
        return status
    log(root, f"starting {job.id} ({job.mode}) {job.source_path}")
    status = execute(job, root, mode=mode, log=lambda m: log(root, m),
                     on_done=_decorator(root, job))
    log(root, f"finished {job.id}: {status.state}: {status.message}")
    return status


# --------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------- #
def cmd_once(args: argparse.Namespace) -> int:
    roots = bridge_dir_or_dirs(args.bridge_dir)
    roots[0].mkdir(parents=True, exist_ok=True)
    served = 0
    opened = 0
    for root in roots:
        touch_alive(root)
        try:
            opened += appbridge.handle_requests(root, log=lambda m, r=root: log(r, m))
        except Exception as exc:  # noqa: BLE001 - one bridge must not stop the rest
            log(root, f"app-open: {type(exc).__name__}: {exc}")
        for path in pending_jobs(root):
            try:
                run_one(root, path, args.mode)
                served += 1
            except Exception as exc:  # noqa: BLE001
                log(root, f"{path.name}: {type(exc).__name__}: {exc}")
    if not served and not opened:
        print(f"nothing pending in {roots[0]}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    roots = bridge_dir_or_dirs(args.bridge_dir)
    roots[0].mkdir(parents=True, exist_ok=True)
    log(roots[0], f"watching {len(roots)} bridge(s) every {args.interval:.1f}s "
                  f"(mode={args.mode or 'per-job'})")
    for extra in roots[1:]:
        log(roots[0], f"  also {extra}")
    try:
        while True:
            for root in roots:
                touch_alive(root)
                try:
                    for path in pending_jobs(root):
                        try:
                            run_one(root, path, args.mode)
                        except Exception as exc:  # noqa: BLE001 - keep the watcher alive
                            # A crash inside one job must never stop the loop: the
                            # panel is polling and would sit at 0% forever.
                            log(root, f"{path.name}: {type(exc).__name__}: {exc}")
                except Exception as exc:  # noqa: BLE001 - an unreadable folder
                    log(root, f"scan: {type(exc).__name__}: {exc}")
                # "Open in Better Brush" is served between jobs, never during
                # one: the app is a second, heavier process on the same GPU.
                try:
                    appbridge.handle_requests(root, log=lambda m, r=root: log(r, m))
                except Exception as exc:  # noqa: BLE001 - same reason
                    log(root, f"app-open: {type(exc).__name__}: {exc}")
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        log(roots[0], "stopped")
        return 0


def cmd_print_commands(args: argparse.Namespace) -> int:
    root = bridge_dir(args.bridge_dir)
    job_path = Path(args.job)
    job = load_job(job_path)
    out_dir = job.png_dir(root)
    first = None
    if not out_dir.is_dir():
        # Dry run: name the file the import *would* point at.
        first = out_dir / "00000.png"
    commands = build_commands(job, out_dir, first_frame=first)
    print(json.dumps({"job": job.id, "png_dir": str(out_dir), "commands": commands},
                     indent=2))
    return 0


def cmd_sample_job(args: argparse.Namespace) -> int:
    from .protocol import Job

    root = bridge_dir(args.bridge_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve()
    job = Job.from_dict(
        {
            "protocol": 1,
            "id": new_job_id(),
            "mode": args.mode,
            "source": str(source),
            "range": {"frame_in": 0, "frame_out": max(0, args.frames - 1)},
            "media": {"width": args.width, "height": args.height, "frame_rate": 24.0},
            "comp": {
                "name": "Sample",
                "width": args.width,
                "height": args.height,
                "frame_rate": 24.0,
                "duration": args.frames / 24.0,
                "time": 0.0,
            },
            "source_layer": {
                "id": 1,
                "name": "Sample layer",
                "index": 1,
                "in": 0.0,
                "out": args.frames / 24.0,
                "start": 0.0,
                "stretch": 100.0,
            },
            "target": {"matte_index": 1},
            "transform": {
                "anchor": [args.width / 2, args.height / 2, 0],
                "position": [args.width / 2, args.height / 2, 0],
                "scale": [100, 100, 100],
                "rotation": 0,
            },
            "options": {"preset": "anime", "quality": "balanced", "matting": False},
        }
    )
    path = root / f"job-{job.id}.json"
    write_json_atomic(path, job.to_dict())
    print(f"wrote {path}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print("bridge folder candidates (first existing one is used):")
    for i, cand in enumerate(candidate_bridge_dirs(args.bridge_dir)):
        mark = "*" if cand.is_dir() else " "
        print(f"  {mark} {cand}")
    root = bridge_dir(args.bridge_dir)
    print(f"\nactive: {root}")
    print(f"exists: {root.is_dir()}")

    from .pipeline import find_ffmpeg

    print(f"ffmpeg: {find_ffmpeg() or 'NOT FOUND (engine mode needs it)'}")

    try:
        import numpy  # noqa: F401

        print("numpy:  ok")
    except ImportError:
        print("numpy:  MISSING")
    try:
        import cv2  # noqa: F401

        print("cv2:    ok")
    except ImportError:
        print("cv2:    MISSING (both modes need it)")

    try:
        import src.engine.pipeline  # noqa: F401

        print("engine: importable (mode=engine available)")
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        print(f"engine: NOT importable ({exc}); mode=simulate only")

    if root.is_dir():
        jobs = list(root.glob("job-*.json"))
        print(f"jobs:   {len(jobs)} file(s), {len(pending_jobs(root))} pending")
    return 0


# --------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="effectcraft-bridge",
        description="Better Brush <-> EffectCraft job adapter",
    )
    parser.add_argument("--bridge-dir", default=None,
                        help="override the folder the panel writes job files to")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_mode(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", choices=("simulate", "engine"), default=None,
                       help="force a mode instead of the one in the job file")

    p = sub.add_parser("watch", help="poll for jobs until interrupted")
    add_mode(p)
    p.add_argument("--interval", type=float, default=1.0)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("once", help="drain pending jobs and exit")
    add_mode(p)
    p.set_defaults(func=cmd_once)

    p = sub.add_parser("print-commands", help="show the EffectCraft commands for a job")
    p.add_argument("--job", required=True, help="path to a job-*.json file")
    p.set_defaults(func=cmd_print_commands)

    p = sub.add_parser("sample-job", help="write a synthetic job file")
    p.add_argument("--mode", choices=("simulate", "engine"), default="simulate",
                   help="mode recorded in the job file (default: simulate)")
    p.add_argument("--source", required=True, help="video or image the job points at")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.set_defaults(func=cmd_sample_job)

    p = sub.add_parser("doctor", help="report what the adapter can reach")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
