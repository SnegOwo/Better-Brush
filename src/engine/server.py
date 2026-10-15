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
``POST /job/<id>/cancel``   request cancellation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _say("Shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
