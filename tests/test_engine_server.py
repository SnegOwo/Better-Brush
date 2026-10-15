"""HTTP surface: real sockets on an ephemeral port with a stub pipeline."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from src.engine import API_VERSION, ENGINE_NAME
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


def test_options_preflight(start_server):
    _srv, client = start_server(_ok_pipeline)
    req = urllib.request.Request(client.base + "/job", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
