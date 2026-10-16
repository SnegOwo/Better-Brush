"""Single-slot job registry for the headless engine.

The panel only ever has one button, so the engine runs one job at a time and
answers ``busy`` for anything else. Keeping it single-slot is what makes
cancellation and progress trivial to reason about.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.engine.schemas import (
    CANCELLED,
    DONE,
    ERR_INTERNAL,
    ERROR,
    QUEUED,
    RUNNING,
    TERMINAL_STATES,
    JobCancelled,
    JobError,
    JobProgress,
    JobRequest,
    JobResult,
)

logger = logging.getLogger(__name__)


class BusyError(RuntimeError):
    """Raised when a job is requested while another is still running."""


@dataclass
class JobRecord:
    id: str
    request: JobRequest
    state: str = QUEUED
    progress: JobProgress = field(default_factory=JobProgress)
    result: JobResult | None = None
    error: JobError | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict() if self.result is not None else None,
            "error": self.error.to_dict() if self.error is not None else None,
        }


class _ThreadStarter:
    """Minimal seam so tests can run jobs synchronously."""

    def __init__(self, target: Callable[[], None], name: str) -> None:
        self._thread = threading.Thread(target=target, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)


class JobRegistry:
    def __init__(
        self,
        pipeline_fn: Callable[..., JobResult],
        *,
        history: int = 8,
        starter: Callable[[Callable[[], None], str], Any] | None = None,
    ) -> None:
        self._pipeline = pipeline_fn
        self._history = max(1, history)
        self._starter = starter or _ThreadStarter
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._active: str | None = None

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def busy(self) -> bool:
        return self.active_id is not None

    @property
    def active_id(self) -> str | None:
        with self._lock:
            if self._active is None:
                return None
            record = self._jobs.get(self._active)
            if record is None or record.terminal:
                return None
            return self._active

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> dict[str, Any] | None:
        """Compact view of the running job, for ``GET /health``."""
        active = self.active_id
        if active is None:
            return None
        record = self.get(active)
        return record.to_dict() if record is not None else None

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def submit(self, request: JobRequest) -> JobRecord:
        with self._lock:
            if self._active is not None:
                current = self._jobs.get(self._active)
                if current is not None and not current.terminal:
                    raise BusyError("a job is already running")
            job_id = uuid.uuid4().hex[:12]
            record = JobRecord(id=job_id, request=request)
            self._jobs[job_id] = record
            self._order.append(job_id)
            self._active = job_id
            self._trim_locked()
            starter = self._starter(lambda rec=record: self._run(rec), f"bb-job-{job_id}")
        starter.start()
        return record

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.terminal:
                return False
            record.cancel_event.set()
            record.progress.message = "cancelling…"
            return True

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _run(self, record: JobRecord) -> None:
        record.state = RUNNING

        def _progress(stage: str, current: int, total: int, message: str) -> None:
            record.progress.stage = stage
            record.progress.current = int(current)
            record.progress.total = int(total)
            record.progress.message = message

        try:
            result = self._pipeline(
                record.request,
                on_progress=_progress,
                cancel_event=record.cancel_event,
            )
            if record.cancel_event.is_set():
                record.state = CANCELLED
            else:
                record.result = result
                record.state = DONE
                logger.info("job %s finished: %s", record.id, result.output_path)
        except JobCancelled:
            record.state = CANCELLED
        except Exception as exc:  # noqa: BLE001 - every failure is a job error
            kind = getattr(exc, "kind", None) or ERR_INTERNAL
            record.error = JobError(kind=kind, message=str(exc))
            record.state = ERROR
            logger.warning("job %s failed (%s): %s", record.id, kind, exc)
        finally:
            with self._lock:
                if self._active == record.id:
                    self._active = None

    def _trim_locked(self) -> None:
        if len(self._order) <= self._history:
            return
        finished = [jid for jid in self._order if self._jobs[jid].terminal]
        while len(self._order) > self._history and finished:
            old = finished.pop(0)
            self._order.remove(old)
            self._jobs.pop(old, None)
