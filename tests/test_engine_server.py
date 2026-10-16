"""HTTP surface: real sockets on an ephemeral port with a stub pipeline."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.engine import API_VERSION, ENGINE_NAME, server
from src.engine.schemas import JobCancelled
from src.engine.server import create_server
from tests.engine_helpers import make_result


class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def get(self, path: str):
        return self._request(path, None, "GET")

    def post(self, path: str, body=None):
        return self._request(path, body if body is not None else {}, "POST")

    def _request(self, path: str, body, method: str):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.fixture
def start_server():
    servers = []

    def factory(pipeline):
        srv = create_server(
            "127.0.0.1", 0, pipeline_fn=pipeline, pro_probe=lambda: True
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        servers.append(srv)
        port = srv.server_address[1]
        return srv, Client(f"http://127.0.0.1:{port}")

    yield factory

    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _ok_pipeline(req, on_progress=None, cancel_event=None):
    if on_progress is not None:
        on_progress("export", 10, 10, "done")
    return make_result()


def test_health_is_cheap_and_reports_contract(start_server):
    _srv, client = start_server(_ok_pipeline)
    status, payload = client.get("/health")
    assert status == 200
    assert payload["app"] == ENGINE_NAME
    assert payload["api_version"] == API_VERSION
    assert payload["pro"] is True
    assert payload["busy"] is False
    assert payload["job"] is None


def test_root_also_answers_health(start_server):
    _srv, client = start_server(_ok_pipeline)
    status, payload = client.get("/")
    assert status == 200
    assert payload["app"] == ENGINE_NAME


def test_health_full_includes_capabilities(start_server):
    _srv, client = start_server(_ok_pipeline)
    status, payload = client.get("/health?full=1")
    assert status == 200
    assert "capabilities" in payload


def test_health_reports_the_log_path(start_server):
    """The panel points the user at the log when a job fails internally."""
    _srv, client = start_server(_ok_pipeline)
    _status, payload = client.get("/health")
    assert payload["log_path"].endswith("engine.log")


def test_post_job_then_poll_result(start_server):
    _srv, client = start_server(_ok_pipeline)
    status, payload = client.post("/job", {"input_path": "clip.mov"})
    assert status == 202
    job_id = payload["id"]

    status, job = client.get(f"/job/{job_id}")
    assert status == 200
    assert job["state"] == "done"
    assert job["result"]["output_kind"] == "mov_alpha"
    assert job["progress"]["current"] == 10


def test_bad_body_is_400(start_server):
    _srv, client = start_server(_ok_pipeline)
    status, payload = client.post("/job", {"input_path": ""})
    assert status == 400
    assert payload["error"]["kind"] == "bad_request"


def test_unknown_job_is_404(start_server):
    _srv, client = start_server(_ok_pipeline)
    status, _payload = client.get("/job/deadbeef")
    assert status == 404


def test_unknown_route_is_404(start_server):
    _srv, client = start_server(_ok_pipeline)
    status, _payload = client.get("/nope")
    assert status == 404


def test_busy_then_cancel(start_server):
    release = threading.Event()

    def blocking(req, on_progress=None, cancel_event=None):
        release.wait(5.0)
        if cancel_event.is_set():
            raise JobCancelled()
        return make_result()

    _srv, client = start_server(blocking)
    status, payload = client.post("/job", {"input_path": "clip.mov"})
    assert status == 202
    job_id = payload["id"]

    status, payload = client.post("/job", {"input_path": "clip.mov"})
    assert status == 409
    assert payload["error"]["kind"] == "busy"

    ok, cancelled = client.post(f"/job/{job_id}/cancel")
    assert ok == 200
    assert cancelled["ok"] is True
    release.set()


def test_dropped_connection_is_quiet(start_server, capsys):
    srv, _client = start_server(_ok_pipeline)
    srv.handle_error(None, ("127.0.0.1", 1234))
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_options_preflight(start_server):
    _srv, client = start_server(_ok_pipeline)
    req = urllib.request.Request(client.base + "/job", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"


# --------------------------------------------------------------------- #
# Windowless operation: the panel's engine has no console and no reader
# --------------------------------------------------------------------- #
def test_tiny_quality_is_accepted(start_server):
    """SAM 2.1 Tiny is the fastest tier; the request schema must allow it."""
    _srv, client = start_server(_ok_pipeline)
    status, payload = client.post("/job", {"input_path": "clip.mov", "quality": "tiny"})
    assert status == 202
    assert "id" in payload


def test_quiet_keeps_the_log_file_at_info(monkeypatch, tmp_path):
    """``--quiet`` silences the console, never the log.

    The panel starts the engine windowless, so the file is the only evidence a
    run leaves behind. With the level applied to the root logger instead, a
    quiet engine wrote a log with no job in it at all — which is how a fifteen
    minute hang produced zero diagnostics.
    """
    log_path = tmp_path / "engine.log"
    monkeypatch.setattr(server, "log_file_path", lambda: log_path)
    root = logging.getLogger()
    previous_level = root.level
    try:
        server._configure_logging(quiet=True)
        logging.getLogger("src.engine.test").info("a job stage worth keeping")
        for handler in root.handlers:
            handler.flush()

        console = [
            h
            for h in root.handlers
            if getattr(h, "_better_brush", False)
            # Ours, and not the log file: pytest's own capture handler is a
            # StreamHandler too.
            and not isinstance(h, logging.FileHandler)
        ]
        assert console and all(h.level == logging.WARNING for h in console)
    finally:
        root.setLevel(previous_level)
        for handler in list(root.handlers):
            if getattr(handler, "_better_brush", False):
                root.removeHandler(handler)
                handler.close()

    assert "a job stage worth keeping" in log_path.read_text(encoding="utf-8")


def test_a_job_survives_a_pipe_nobody_reads(tmp_path):
    """The regression test for the fifteen-minute hang.

    The panel starts the engine from After Effects with ``start /b``, so the
    engine inherits an output pipe that After Effects never drains. SAM 2 draws
    a tqdm bar while it loads frames; once the pipe's buffer filled, that write
    blocked forever and the job sat at 0% with the GPU idle. The child here does
    exactly that on purpose: it writes far more than a pipe buffer to both
    streams and must still exit.
    """
    script = tmp_path / "sink_probe.py"
    repo = Path(__file__).resolve().parents[1]
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{repo}')\n"
        "from src.engine.server import _detach_stdio\n"
        "detached = _detach_stdio()\n"
        "sys.stdout.write('x' * 200_000)\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('y' * 200_000)\n"
        "sys.stderr.flush()\n"
        "raise SystemExit(0 if detached else 2)\n",
        encoding="utf-8",
    )
    # Popen, not run(): run() would drain the pipes for the child and hide the
    # very bug this covers.
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
