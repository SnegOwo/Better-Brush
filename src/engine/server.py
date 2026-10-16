"""Localhost HTTP surface for the After Effects panel.

Standard-library ``http.server`` on purpose — no new dependency, and the whole
module imports without torch. It binds to ``127.0.0.1`` only: the loopback
socket *is* the security boundary, so there is no token to misplace.

Routes
------
``GET  /health``            capability + current-job snapshot (cheap)
``GET  /health?full=1``     adds the hardware/model description
``GET  /job/<id>``          job state / progress / result / error
``POST /job``               start a job -> 202 ``{"id": ...}`` (409 when busy)
                            also the brush-correction entry point: the body
                            carries ``resume_dir`` plus base64 ``correction_add``
                            / ``correction_remove`` strokes
``POST /job/<id>/cancel``   request cancellation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.engine import API_VERSION, ENGINE_NAME, ENGINE_VERSION
from src.engine.jobs import BusyError, JobRegistry
from src.engine.schemas import (
    ERR_BUSY,
    ERR_INTERNAL,
    RequestError,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_MAX_BYTES = 1_000_000
_LOG_BACKUPS = 2


def log_file_path() -> Path:
    """Where the engine keeps its log file.

    The panel starts the engine with ``pythonw.exe``, which has no console, so
    ``sys.stderr`` goes nowhere. Without a file, a failure would leave only the
    one line of text the panel happens to show as evidence.
    """
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "BetterBrush" / "engine.log"


def _stdio_is_detached() -> bool:
    """True when this process must not write to its inherited streams.

    ``pythonw.exe`` has no streams at all, and the panel starts the engine from
    After Effects with ``start /b``, so the engine also inherits the parent's
    output pipe. Windows blocks a write once such a pipe fills with nobody
    draining it, and libraries do print.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            return True
        try:
            if not stream.isatty():
                return True
        except (AttributeError, OSError, ValueError):
            return True
    return False


def _detach_stdio() -> bool:
    """Point a windowless engine's streams at the null device.

    This is the fix for a job that "runs" for fifteen minutes at 0% with the
    GPU idle: SAM 2 draws a ``tqdm`` bar while it loads video frames, the bar
    goes to an output pipe nobody reads, and the write blocks. The engine
    reports progress over HTTP, so it has nothing to say to a console it does
    not own — the log file is the diagnostic channel.

    Only called from :func:`main`, and only when the streams are not a console,
    so running the engine in a terminal still prints.
    """
    if not _stdio_is_detached():
        return False
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:  # pragma: no cover - devnull always opens
        return False
    try:
        for fd in (1, 2):
            os.dup2(devnull, fd)
    finally:
        os.close(devnull)
    try:
        # Deliberately not a context manager: this is the process-lifetime sink
        # for a windowless engine, closed when the process exits.
        sink = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    except OSError:  # pragma: no cover
        return False
    sys.stdout = sink
    sys.stderr = sink
    return True


def _configure_logging(quiet: bool) -> Path | None:
    """Attach a console handler and a rotating file handler.

    ``quiet`` silences the *console* only. The file always keeps INFO: the panel
    launches the engine windowless, so the file is the only evidence a run
    leaves behind, and "it hung for fifteen minutes" is impossible to answer
    from a log that recorded nothing.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Idempotent: tests call this more than once in a process.
    for existing in list(root.handlers):
        if getattr(existing, "_better_brush", False):
            root.removeHandler(existing)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING if quiet else logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    console._better_brush = True  # type: ignore[attr-defined]
    root.addHandler(console)

    path = log_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8"
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler._better_brush = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    except OSError as exc:
        logger.warning("could not open the engine log at %s: %s", path, exc)
        return None
    return path

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Private-Network": "true",
    "Access-Control-Max-Age": "600",
}


def _default_pipeline(request, on_progress=None, cancel_event=None):
    from src.engine.pipeline import run_job

    return run_job(request, on_progress=on_progress, cancel_event=cancel_event)


def _license_is_pro() -> bool:
    try:
        from src.core.licensing import is_pro

        return bool(is_pro())
    except Exception:  # noqa: BLE001 - licensing is best-effort telemetry here
        return False


class EngineServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        registry: JobRegistry,
        *,
        pro_probe: Callable[[], bool] = _license_is_pro,
    ) -> None:
        super().__init__(address, EngineHandler)
        self.registry = registry
        self.pro_probe = pro_probe

    def handle_error(self, request: object, client_address: tuple) -> None:
        """Keep a dropped keep-alive connection out of the console.

        HTTP/1.1 clients (fetch, urllib) routinely close a connection right
        after reading a response; socketserver's default handler prints a full
        traceback for that, which would spam the engine log for no reason.
        """
        error = sys.exc_info()[1]
        logger.debug("request error from %s: %r", client_address, error)


class EngineHandler(BaseHTTPRequestHandler):
    server: EngineServer
    server_version = f"BetterBrushEngine/{ENGINE_VERSION}"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in _CORS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in _CORS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError(f"invalid JSON body: {exc}") from exc

    @staticmethod
    def _route(path: str) -> tuple[str, list[str], dict[str, str]]:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(path)
        parts = [p for p in parsed.path.split("/") if p]
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return parsed.path, parts, query

    # ------------------------------------------------------------------ #
    # Verbs
    # ------------------------------------------------------------------ #
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for key, value in _CORS.items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path, parts, query = self._route(self.path)

        if path in ("/", "/health") or parts == ["health"]:
            full = query.get("full") in ("1", "true", "yes")
            self._send_json(200, self._health(full))
            return

        if len(parts) == 2 and parts[0] == "job":
            record = self.server.registry.get(parts[1])
            if record is None:
                self._send_json(404, {"error": {"kind": "not_found", "message": "unknown job"}})
                return
            self._send_json(200, record.to_dict())
            return

        self._send_json(404, {"error": {"kind": "not_found", "message": "no such route"}})

    def do_POST(self) -> None:
        _path, parts, _query = self._route(self.path)

        if parts == ["job"]:
            self._start_job()
            return

        if len(parts) == 3 and parts[0] == "job" and parts[2] == "cancel":
            ok = self.server.registry.cancel(parts[1])
            self._send_json(200 if ok else 404, {"ok": ok})
            return

        self._send_json(404, {"error": {"kind": "not_found", "message": "no such route"}})

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #
    def _health(self, full: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "app": ENGINE_NAME,
            "engine_version": ENGINE_VERSION,
            "api_version": API_VERSION,
            "pro": self.server.pro_probe(),
            "ready": True,
            "busy": self.server.registry.busy,
            "job": self.server.registry.snapshot(),
            "log_path": str(log_file_path()),
        }
        if full:
            try:
                from src.engine import model_setup

                payload["capabilities"] = model_setup.describe()
            except Exception as exc:  # noqa: BLE001
                payload["capabilities"] = {"error": str(exc)}
        return payload

    def _start_job(self) -> None:
        from src.engine.schemas import JobRequest

        try:
            request = JobRequest.from_dict(self._read_json())
        except RequestError as exc:
            self._send_json(
                400, {"error": {"kind": exc.kind, "message": str(exc)}}
            )
            return

        try:
            record = self.server.registry.submit(request)
        except BusyError as exc:
            self._send_json(
                409, {"error": {"kind": ERR_BUSY, "message": str(exc)}}
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                500, {"error": {"kind": ERR_INTERNAL, "message": str(exc)}}
            )
            return

        logger.info(
            "accepted job %s: %s (quality=%s, preset=%s, model=%s)",
            record.id,
            request.input_path,
            request.quality,
            request.preset,
            request.model or "auto",
        )
        self._send_json(202, {"id": record.id, "state": record.state})


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    pipeline_fn: Callable[..., Any] | None = None,
    registry: JobRegistry | None = None,
    pro_probe: Callable[[], bool] = _license_is_pro,
) -> EngineServer:
    """Build an :class:`EngineServer` (does not start serving)."""
    if registry is None:
        registry = JobRegistry(pipeline_fn or _default_pipeline)
    return EngineServer((host, port), registry, pro_probe=pro_probe)


def _say(message: str) -> None:
    """Print only when a console exists (pythonw.exe has no stdout)."""
    try:
        if sys.stdout is not None:
            print(message)
    except Exception:
        logger.debug("could not write to stdout", exc_info=True)


def _engine_running(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=1.0
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("app") == ENGINE_NAME
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="better-brush-engine",
        description="Better Brush headless engine (used by the After Effects panel).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # Order matters: the console handler captures sys.stderr when it is built,
    # so detach first or it keeps writing to the inherited pipe.
    detached = _detach_stdio()
    log_path = _configure_logging(args.quiet)
    if detached:
        logger.info(
            "stdout/stderr are not a console: detached to %s so nothing can "
            "block on an unread pipe (log: %s)",
            os.devnull,
            log_path,
        )

    if _engine_running(args.host, args.port):
        _say(f"Better Brush engine already running on http://{args.host}:{args.port}")
        return 0

    try:
        server = create_server(args.host, args.port)
    except OSError as exc:
        _say(
            f"Could not bind {args.host}:{args.port} ({exc}). "
            "Another program may be using the port."
        )
        return 2

    _say(f"Better Brush engine ready on http://{args.host}:{args.port}")
    if log_path is not None:
        _say(f"Logging to {log_path}")
    logger.info(
        "engine ready on http://%s:%s (log: %s)", args.host, args.port, log_path
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _say("Shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
