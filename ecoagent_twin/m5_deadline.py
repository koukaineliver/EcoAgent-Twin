"""Contain model wait time with one reusable worker and no inference backlog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from threading import Condition, Event, Thread
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .m3_supervisor import DecisionView


class WorkerUnavailable(RuntimeError):
    """The model worker is closed, still loading, or failed to initialize."""


class WorkerBusy(RuntimeError):
    """The model worker is still handling an earlier request."""


@dataclass
class _Request:
    view: DecisionView
    deadline: float
    done: Event = field(default_factory=Event)
    proposal: object = None
    scores: dict[str, float] = field(default_factory=dict)
    error: Exception | None = None


def _valid_seconds(value: float) -> bool:
    return type(value) in (float, int) and isfinite(value) and value >= 0


class DeadlineModel:
    """Use a prewarmed model without waiting beyond the selection allowance.

    The factory and model calls run on a single daemon thread. A timed-out
    native call can finish in the background; nothing queues behind it. This
    bounds the caller's wait, subject to normal interpreter and OS scheduling,
    and does not cancel native inference.
    """

    def __init__(self, factory, *, reserve_s: float = 0.1) -> None:
        if not callable(factory):
            raise ValueError("model factory must be callable")
        if not _valid_seconds(reserve_s):
            raise ValueError("deadline reserve must be finite and nonnegative")
        self._condition = Condition()
        self._ready = Event()
        self._available = False
        self._closed = False
        self._initialization_error: Exception | None = None
        self._request: _Request | None = None
        self._reserve_s = reserve_s
        self._attempt = 0
        self._last_status = "starting"
        self._last_scores: dict[str, float] = {}
        self._thread = Thread(
            target=self._worker, args=(factory,), daemon=True, name="ecoagent-model-worker"
        )
        self._thread.start()

    @property
    def last_status(self) -> str:
        with self._condition:
            return self._last_status

    @property
    def last_scores(self) -> dict[str, float]:
        with self._condition:
            return self._last_scores.copy()

    def accepted_scores(self, receipt: Mapping[str, object]) -> dict[str, float]:
        """Suppress scores when the supervisor rejects or falls back."""
        if receipt.get("status") != "selected" or receipt.get("fallback_used"):
            return {}
        return self.last_scores

    def _worker(self, factory) -> None:
        try:
            backend = factory()
            if not callable(backend):
                raise TypeError("model factory did not return a callable")
            with self._condition:
                if self._closed:
                    return
                self._available = True
                if self._last_status == "starting":
                    self._last_status = "ready"
                self._ready.set()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._request is not None)
                    if self._closed:
                        return
                    request = self._request
                self._execute(backend, request)
        except Exception as error:
            with self._condition:
                if not self._ready.is_set():
                    self._initialization_error = error
        finally:
            with self._condition:
                self._available = False
                self._last_status = "unavailable"
                self._last_scores = {}
                if self._request is not None and not self._request.done.is_set():
                    self._request.error = WorkerUnavailable("model worker stopped")
                    self._request.done.set()
                self._request = None
                self._ready.set()

    def _execute(self, backend, request: _Request) -> None:
        proposal, scores, error = None, {}, None
        try:
            remaining = request.deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError("model selection allowance expired")
            proposal = backend(request.view, timeout_s=remaining)
            if isinstance(proposal, Mapping):
                proposal = dict(proposal)
            scores = dict(getattr(backend, "last_scores", {}))
        except Exception as failure:
            error = failure
        with self._condition:
            if not request.done.is_set():
                if perf_counter() >= request.deadline:
                    error = TimeoutError("model selection allowance expired")
                request.error = error
                if error is None:
                    request.proposal, request.scores = proposal, scores
                request.done.set()
            self._request = None

    def warmup(self, view: DecisionView | None = None, *, timeout_s: float = 30.0) -> None:
        """Wait for initialization and optionally run one unrecorded model inference."""
        if not _valid_seconds(timeout_s) or timeout_s == 0:
            raise ValueError("warm-up timeout must be finite and positive")
        started = perf_counter()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("model worker did not finish initialization")
        with self._condition:
            if not self._available or self._closed:
                raise WorkerUnavailable("model worker initialization failed or was closed") from (
                    self._initialization_error
                )
        if view is not None:
            remaining = timeout_s - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("model warm-up allowance expired")
            self(view, timeout_s=remaining)
            with self._condition:
                self._last_scores = {}
                self._last_status = "ready"

    def __call__(self, view: DecisionView, *, timeout_s: float):
        started = perf_counter()
        with self._condition:
            self._attempt += 1
            attempt = self._attempt
            self._last_scores = {}
            if not _valid_seconds(timeout_s):
                raise ValueError("model timeout must be finite and nonnegative")
            allowance = timeout_s - self._reserve_s
            if allowance <= 0:
                self._last_status = "timeout"
                raise TimeoutError("remaining model budget is within the dispatch reserve")
            if not self._available or self._closed:
                self._last_status = "unavailable"
                raise WorkerUnavailable("model worker is unavailable")
            if self._request is not None:
                self._last_status = "busy"
                raise WorkerBusy("earlier inference is still running")
            request = _Request(view, started + allowance)
            self._request = request
            self._last_status = "running"
            self._condition.notify()

        completed = request.done.wait(max(0.0, request.deadline - perf_counter()))
        with self._condition:
            if not completed or perf_counter() >= request.deadline:
                error = TimeoutError("model selection allowance expired")
            else:
                error = request.error
            if attempt == self._attempt and not self._closed:
                if error is None:
                    self._last_scores = request.scores.copy()
                    self._last_status = "completed"
                elif isinstance(error, TimeoutError):
                    self._last_status = "timeout"
                elif isinstance(error, WorkerUnavailable):
                    self._last_status = "unavailable"
                else:
                    self._last_status = "error"
        if error is not None:
            raise error
        return request.proposal

    def close(self, wait_s: float = 0.0) -> bool:
        """Stop accepting calls and optionally wait briefly for the worker to exit."""
        if not _valid_seconds(wait_s):
            raise ValueError("shutdown wait must be finite and nonnegative")
        with self._condition:
            self._closed = True
            self._last_scores = {}
            self._last_status = "unavailable"
            if self._request is not None and not self._request.done.is_set():
                self._request.error = WorkerUnavailable("model worker was closed")
                self._request.done.set()
            self._ready.set()
            self._condition.notify_all()
        self._thread.join(wait_s)
        return not self._thread.is_alive()

    def __enter__(self) -> DeadlineModel:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
