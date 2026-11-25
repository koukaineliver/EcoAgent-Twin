"""Read-only LangGraph selection over a shared, approved intervention menu."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import isfinite
from time import perf_counter_ns
from types import MappingProxyType
from typing import Protocol

from typing_extensions import TypedDict


@dataclass(frozen=True, slots=True)
class Candidate:
    """Scores available at decision time for one approved intervention."""

    id: str
    label: str
    served: int
    travel_wh: int
    unserved: int
    restricted_recovery_min: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("candidate ID must be a nonempty string")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("candidate label must be a nonempty string")
        if any(type(value) is not int or value < 0 for value in (
            self.served, self.travel_wh, self.unserved,
        )):
            raise ValueError("candidate counts and travel watt-hours must be nonnegative integers")
        if (
            type(self.restricted_recovery_min) not in (int, float)
            or not isfinite(self.restricted_recovery_min)
            or self.restricted_recovery_min < 0
        ):
            raise ValueError("candidate restricted recovery must be finite and nonnegative")


def _freeze(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("alert keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if type(value) in (str, int, bool, type(None)) or (type(value) is float and isfinite(value)):
        return value
    raise ValueError("alert values must be JSON-compatible and finite")


def _to_json_value(value):
    if isinstance(value, Mapping):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DecisionView:
    """Snapshot shared by both backends for one alert and one menu."""

    alert: Mapping[str, object]
    candidates: tuple[Candidate, ...]

    def to_dict(self) -> dict:
        """Return a detached JSON-compatible payload for an injected adapter."""
        return {
            "alert": _to_json_value(self.alert),
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


class Chooser(Protocol):
    def __call__(
        self, view: DecisionView, *, timeout_s: float,
    ) -> str | Mapping[str, object]: ...


class _State(TypedDict, total=False):
    view: DecisionView
    model: Chooser | None
    deadline_ns: int
    deterministic_id: str
    proposed_id: str | None
    proposed_reason: str | None
    failure: str | None
    status: str
    decision_id: str
    reason: str
    fallback_used: bool


_FAILURE_REASONS = {
    "backend_error": "The model backend failed. Used the deterministic menu ranking.",
    "invalid_proposal": (
        "The model response was outside the approved menu or had an invalid format. "
        "Used the deterministic menu ranking."
    ),
    "timeout": (
        "No proposal was accepted within the decision budget. "
        "Used the deterministic menu ranking."
    ),
}
_RANKING_REASON = "Prioritized service, then traction energy, recovery time, and candidate ID."


def _propose(state: _State) -> dict:
    view, model = state["view"], state["model"]
    best = min(view.candidates, key=lambda item: (
        -item.served, item.travel_wh, item.restricted_recovery_min, item.id,
    ))
    update = {"deterministic_id": best.id}
    if model is None:
        return {**update, "proposed_id": best.id, "proposed_reason": _RANKING_REASON}

    remaining_ns = state["deadline_ns"] - perf_counter_ns()
    if remaining_ns <= 0:
        return {**update, "failure": "timeout"}
    try:
        proposal = model(view, timeout_s=remaining_ns / 1_000_000_000)
        if isinstance(proposal, str):
            candidate_id, reason = proposal, None
        elif isinstance(proposal, Mapping):
            candidate_id, reason = proposal.get("candidate_id"), proposal.get("reason")
        else:
            return {**update, "failure": "invalid_proposal"}
    except TimeoutError:
        return {**update, "failure": "timeout"}
    except Exception:
        return {**update, "failure": "backend_error"}
    if not isinstance(candidate_id, str) or (reason is not None and not isinstance(reason, str)):
        return {**update, "failure": "invalid_proposal"}
    return {**update, "proposed_id": candidate_id, "proposed_reason": reason}


def _validate(state: _State) -> dict:
    if perf_counter_ns() >= state["deadline_ns"]:
        return {"status": "timeout"}
    if failure := state.get("failure"):
        return {"status": failure}
    approved_ids = {candidate.id for candidate in state["view"].candidates}
    if state.get("proposed_id") in approved_ids:
        return {"status": "selected"}
    return {"status": "invalid_proposal"}


def _accept(state: _State) -> dict:
    reason = (state.get("proposed_reason") or "").strip()
    return {
        "decision_id": state["proposed_id"],
        "reason": reason[:512] or "The model selected an approved intervention.",
        "fallback_used": False,
    }


def _fallback(state: _State) -> dict:
    return {
        "decision_id": state["deterministic_id"],
        "reason": _FAILURE_REASONS[state["status"]],
        "fallback_used": True,
    }


@lru_cache(maxsize=1)
def _graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(_State)
    graph.add_node("propose", _propose)
    graph.add_node("validate", _validate)
    graph.add_node("accept", _accept)
    graph.add_node("fallback", _fallback)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda state: "accept" if state["status"] == "selected" else "fallback",
        {"accept": "accept", "fallback": "fallback"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def supervise(
    alert: dict, candidates: tuple[Candidate, ...], *,
    model: Chooser | None = None, budget_s: float = 1.0,
) -> dict:
    """Select one candidate without applying it.

    The deadline limits acceptance of proposals. The synchronous adapter receives
    the remaining budget and is responsible for interrupting its own work.
    An overrun is recorded and uses deterministic fallback on return.
    One-time graph setup is outside the per-decision clock.
    """
    from langsmith import tracing_context

    if not isinstance(alert, dict):
        raise ValueError("alert must be a dictionary")
    menu = tuple(candidates)
    if not menu or any(type(candidate) is not Candidate for candidate in menu):
        raise ValueError("an approved intervention menu must contain Candidate entries")
    if len({candidate.id for candidate in menu}) != len(menu):
        raise ValueError("candidate IDs must be unique")
    if len({candidate.served + candidate.unserved for candidate in menu}) != 1:
        raise ValueError("candidates must describe the same parcel cohort")
    if type(budget_s) not in (int, float) or not isfinite(budget_s) or budget_s <= 0:
        raise ValueError("decision budget must be finite and positive")
    if model is not None and not callable(model):
        raise ValueError("model must be a callable")
    graph = _graph()
    started_ns = perf_counter_ns()
    deadline_ns = started_ns + int(budget_s * 1_000_000_000)
    view = DecisionView(_freeze(alert), menu)
    with tracing_context(enabled=False):
        result = graph.invoke({"view": view, "model": model, "deadline_ns": deadline_ns})
    finished_ns = perf_counter_ns()
    deadline_exceeded = finished_ns >= deadline_ns
    if result["status"] == "selected" and deadline_exceeded:
        result["status"] = "timeout"
        result.update(_fallback(result))
    return {
        "status": result["status"],
        "decision_id": result["decision_id"],
        "proposed_id": result.get("proposed_id"),
        "reason": result["reason"],
        "fallback_used": result["fallback_used"],
        "elapsed_ms": (finished_ns - started_ns) / 1_000_000,
        "budget_ms": budget_s * 1_000,
        "deadline_exceeded": deadline_exceeded,
        "backend": "deterministic" if model is None else "model",
    }
