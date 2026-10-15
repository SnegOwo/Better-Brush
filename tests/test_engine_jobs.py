"""Job registry: single-slot semantics, cancellation and error mapping."""

from __future__ import annotations

import time

import pytest

from src.engine.jobs import BusyError, JobRegistry
from src.engine.schemas import (
    CANCELLED,
    DONE,
    ERROR,
    JobCancelled,
    JobRequest,
)
from tests.engine_helpers import SyncStarter, make_result


def _request():
    return JobRequest.from_dict({"input_path": "clip.mov"})


def _wait(pred, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_sync_job_completes_and_records_result():
    registry = JobRegistry(
        lambda req, on_progress=None, cancel_event=None: make_result(),
        starter=SyncStarter,
    )
    record = registry.submit(_request())
    assert record.state == DONE
    assert record.result is not None
    assert registry.get(record.id).result.output_path.endswith("alpha.mov")
    assert registry.busy is False


def test_progress_callback_is_forwarded():
    def pipeline(req, on_progress=None, cancel_event=None):
        on_progress("propagate", 3, 10, "tracking")
        return make_result()

    registry = JobRegistry(pipeline, starter=SyncStarter)
    record = registry.submit(_request())
    assert record.progress.stage == "propagate"
    assert record.progress.current == 3
    assert record.progress.total == 10


def test_busy_while_running_then_cancel():
    def pipeline(req, on_progress=None, cancel_event=None):
        cancel_event.wait(3.0)
        if cancel_event.is_set():
            raise JobCancelled()
        return make_result()

    registry = JobRegistry(pipeline)
    first = registry.submit(_request())
    assert _wait(lambda: registry.busy)
    with pytest.raises(BusyError):
        registry.submit(_request())

    assert registry.cancel(first.id) is True
    assert _wait(lambda: registry.get(first.id).terminal)
    assert registry.get(first.id).state == CANCELLED
    assert registry.busy is False


def test_pipeline_error_kind_is_preserved():
    class PipeError(Exception):
        kind = "no_character"

    def pipeline(req, on_progress=None, cancel_event=None):
        raise PipeError("nothing here")

    registry = JobRegistry(pipeline, starter=SyncStarter)
    record = registry.submit(_request())
    assert record.state == ERROR
    assert record.error is not None
    assert record.error.kind == "no_character"


def test_unknown_exception_becomes_internal_error():
    def pipeline(req, on_progress=None, cancel_event=None):
        raise RuntimeError("boom")

    registry = JobRegistry(pipeline, starter=SyncStarter)
    record = registry.submit(_request())
    assert record.state == ERROR
    assert record.error.kind == "internal"


def test_history_is_trimmed():
    registry = JobRegistry(
        lambda req, on_progress=None, cancel_event=None: make_result(),
        history=2,
        starter=SyncStarter,
    )
    ids = [registry.submit(_request()).id for _ in range(4)]
    assert registry.get(ids[-1]) is not None
    assert registry.get(ids[0]) is None


def test_cancel_unknown_job_is_false():
    registry = JobRegistry(
        lambda req, on_progress=None, cancel_event=None: make_result(),
        starter=SyncStarter,
    )
    assert registry.cancel("nope") is False


def test_snapshot_none_when_idle():
    registry = JobRegistry(
        lambda req, on_progress=None, cancel_event=None: make_result(),
        starter=SyncStarter,
    )
    assert registry.snapshot() is None
