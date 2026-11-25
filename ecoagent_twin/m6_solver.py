"""Pluggable route repair with shared replay and a dependency-free search backend."""

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from time import perf_counter

from .m2_simulator import Action, OutageEnv
from .m3_experiment import _certify_record, _core
from .m3_route_repair import RepairPlan, _Dispatch, repair
from .model import Scenario


RepairBackend = Callable[[Scenario, dict], RepairPlan]
DEFAULT_TRANSITION_LIMIT = 20_000
_ACTION_RANK = {"deposit": 0, "charge": 1, "return": 2}


def cp_sat_backend(**limits) -> RepairBackend:
    def plan(scenario: Scenario, settings: dict) -> RepairPlan:
        return repair(scenario, **settings, **limits)

    return plan


def _action_order(action: Action) -> tuple:
    return (
        _ACTION_RANK[action.kind], action.completion_min, action.travel_min,
        action.energy_wh, action.parcel_id or "", action.locker_id or "",
    )


def _return_plan(environment: OutageEnv) -> tuple[tuple[_Dispatch, ...], dict]:
    sequence = []
    while (decision := environment.observation()) is not None:
        index = next(i for i, action in enumerate(decision.actions) if action.kind == "return")
        sequence.append(_Dispatch(decision.vehicle_id, decision.time_min, decision.actions[index]))
        environment.step(index)
    return tuple(sequence), environment.result()


def _objective(result: dict) -> tuple[int, int, int]:
    metrics = result["metrics"]
    return -metrics["served_parcels"], metrics["travel_min"], metrics["travel_wh"]


def _branches(environment, path, available, travel_min, travel_wh, view):
    order = sorted(range(len(view.actions)), key=lambda i: _action_order(view.actions[i]), reverse=True)
    return [(environment, path, available, travel_min, travel_wh, index) for index in order]


def search_plan(scenario: Scenario, settings: dict, *, transition_limit: int) -> RepairPlan:
    """Search the simulator's full legal action tree until proof or a work limit.

    A partial route could serve at most the parcels deposited, committed, or
    still in the custody of a vehicle which can act later. Accrued travel time
    and energy cannot decrease. Their lexicographic lower bound only prunes
    branches that cannot strictly improve the incumbent. Ties keep the first
    incumbent in a stable action order.
    """
    if type(transition_limit) is not int or transition_limit < 1:
        raise ValueError("search transition limit must be a positive integer")
    environment = OutageEnv(scenario, **settings)
    best_path, best_result = _return_plan(deepcopy(environment, memo={id(scenario): scenario}))
    best_objective = _objective(best_result)
    ownership = Counter(parcel.vehicle_id for parcel in scenario.parcels)
    transitions = pruned = completed = 0
    stack = _branches(environment, (), len(scenario.parcels), 0, 0, environment.observation())
    exact = True
    while stack:
        parent, previous, available, travel_min, travel_wh, index = stack.pop()
        view = parent.observation()
        action = view.actions[index]
        if action.kind == "return":
            already_deposited = sum(
                dispatch.vehicle_id == view.vehicle_id and dispatch.action.kind == "deposit"
                for dispatch in previous
            )
            available -= ownership[view.vehicle_id] - already_deposited
        travel_min += action.travel_min
        travel_wh += action.energy_wh
        if (-available, travel_min, travel_wh) >= best_objective:
            pruned += 1
            continue
        if transitions == transition_limit:
            exact = False
            break
        path = (*previous, _Dispatch(view.vehicle_id, view.time_min, action))
        environment = deepcopy(parent, memo={id(scenario): scenario})
        environment.step(index)
        transitions += 1
        if (decision := environment.observation()) is not None:
            stack.extend(_branches(environment, path, available, travel_min, travel_wh, decision))
        else:
            completed += 1
            result = environment.result()
            if (objective := _objective(result)) < best_objective:
                best_path, best_result, best_objective = path, result, objective
    return RepairPlan(best_path, {
        "engine": "python_simulator_branch_and_bound",
        "strategy": "simulator_action_search",
        "solver_status": "EXACT_SIMULATOR_ACTION_TREE" if exact else "TRANSITION_LIMIT",
        "optimal_within_simulator_action_tree": exact,
        "transition_limit": transition_limit,
        "transitions": transitions,
        "pruned_branches": pruned,
        "completed_search_leaves": completed,
        "incumbent_scope": "fixed offers, fixed custody, legal simulator actions",
        "certified_metrics": best_result["metrics"],
        "certified_unserved_parcel_ids": best_result["unserved_parcel_ids"],
    })


def simulator_backend(transition_limit: int = DEFAULT_TRANSITION_LIMIT) -> RepairBackend:
    return partial(search_plan, transition_limit=transition_limit)


def certified_repair(scenario: Scenario, settings: dict, backend: RepairBackend) -> dict:
    """Run any compatible planner and verify its output using the common replay."""
    started = perf_counter()
    plan = backend(scenario, settings)
    planning_ms = (perf_counter() - started) * 1000
    replay_started = perf_counter()
    record = _core(scenario, settings, plan.choose)
    _certify_record(scenario, settings, record)
    result = record["result"]
    if (result["metrics"] != plan.receipt["certified_metrics"]
            or result["unserved_parcel_ids"] != plan.receipt["certified_unserved_parcel_ids"]):
        raise ValueError("repair backend disagrees with shared replay")
    return {
        "record": record, "planner": plan.receipt, "planning_ms": planning_ms,
        "shared_replay_ms": (perf_counter() - replay_started) * 1000,
    }
