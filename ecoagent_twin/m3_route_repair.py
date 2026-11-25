"""Bounded CP-SAT route repair with certification in the common outage simulator."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from time import perf_counter

from .m2_rule import RuleBased
from .m2_simulator import Action, Decision, OutageEnv
from .model import Scenario, Vehicle


@dataclass(frozen=True)
class _Charge:
    request_min: int
    start_min: int
    end_min: int


@dataclass(frozen=True)
class _Column:
    actions: tuple[Action, ...]
    slots: Counter
    charge: _Charge | None
    latest_charge_start: int | None
    served: int
    travel_min: int
    travel_wh: int


@dataclass(frozen=True)
class _Dispatch:
    vehicle_id: str
    time_min: int
    action: Action


class _CertificateError(ValueError):
    pass


class RepairPlan:
    """Play one previously certified schedule against the same environment inputs."""

    def __init__(self, sequence: tuple[_Dispatch, ...], receipt: dict) -> None:
        self._sequence = sequence
        self._next = 0
        self.receipt = receipt

    def choose(self, decision: Decision) -> int:
        if self._next == len(self._sequence):
            raise ValueError("certified repair has no remaining action")
        expected = self._sequence[self._next]
        if (decision.vehicle_id, decision.time_min) != (expected.vehicle_id, expected.time_min):
            raise ValueError("live vehicle or timestamp differs from the certified repair")
        try:
            index = decision.actions.index(expected.action)
        except ValueError as exc:
            raise ValueError(f"certified repair action is masked for {decision.vehicle_id}") from exc
        self._next += 1
        return index


def _column(
    path: tuple[Action, ...], charge: _Charge | None, slots: dict[str, str],
    scenario: Scenario, deadlines: dict[str, int],
) -> _Column | None:
    deposits = [action for action in path if action.kind == "deposit"]
    latest_start = None
    if charge is not None:
        charge_index = next(i for i, action in enumerate(path) if action.kind == "charge")
        subsequent_deposits = [action for action in path[charge_index + 1:] if action.kind == "deposit"]
        if not subsequent_deposits:
            return None
        slack = scenario.horizon_min - path[-1].completion_min
        for action in subsequent_deposits:
            direct_return = scenario.travel_time[action.locker_id, scenario.depot]
            slack = min(
                slack, deadlines[action.parcel_id] - action.completion_min,
                scenario.horizon_min - action.completion_min - direct_return,
            )
        latest_start = charge.start_min + slack
    return _Column(
        path,
        Counter(action.locker_id for action in deposits if action.parcel_id not in slots),
        charge,
        latest_start,
        len(deposits),
        sum(action.travel_min for action in path),
        sum(action.energy_wh for action in path),
    )


def _vehicle_columns(
    scenario: Scenario, vehicle: Vehicle, options: dict, slots: dict[str, str], limit: int,
) -> tuple[list[_Column], bool]:
    parcels = tuple(parcel for parcel in scenario.parcels if parcel.vehicle_id == vehicle.id)
    if not parcels:
        back = vehicle.node, scenario.depot
        travel = scenario.travel_time[back]
        start = options["dispatch_after_min"]
        if start is None:
            start = scenario.outage_min
        action = Action("return", None, None, travel, start + travel, scenario.travel_energy[back])
        return [_column((action,), None, slots, scenario, {})], False

    owned = {parcel.id for parcel in parcels}
    other_holds = Counter(locker for parcel, locker in slots.items() if parcel not in owned)
    local = replace(
        scenario,
        vehicles=(vehicle,),
        parcels=parcels,
        lockers={
            locker_id: replace(locker, capacity=locker.capacity - other_holds[locker_id])
            for locker_id, locker in scenario.lockers.items()
        },
    )
    local_options = {
        **options,
        "preassigned_slots": {parcel: locker for parcel, locker in slots.items() if parcel in owned},
    }
    environment = OutageEnv(local, **local_options)
    stack = [(environment, (), None)]
    columns = []
    deadlines = {parcel.id: parcel.deadline_min for parcel in parcels}
    while stack and len(columns) < limit:
        environment, path, charge = stack.pop()
        decision = environment.observation()
        direct = next(action for action in decision.actions if action.kind == "return")
        column = _column((*path, direct), charge, slots, local, deadlines)
        if column is not None:
            columns.append(column)
        for index in reversed(range(len(decision.actions))):
            action = decision.actions[index]
            if action.kind == "return" or (action.kind == "charge" and charge is not None):
                continue
            selected_charge = charge
            if action.kind == "charge":
                selected_charge = _Charge(
                    decision.time_min, decision.time_min + action.travel_min, action.completion_min,
                )
            child = deepcopy(environment, memo={id(local): local})
            child.step(index)
            stack.append((child, (*path, action), selected_charge))
    columns.sort(key=lambda column: (
        len(column.actions),
        tuple((action.kind, action.parcel_id or "", action.locker_id or "") for action in column.actions),
    ))
    return columns, bool(stack)


def _select_columns(
    candidates: dict[str, list[_Column]], available: dict[str, int], horizon: int, timeout: float,
):
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    variables = {}
    for vehicle, columns in candidates.items():
        chosen = [model.new_bool_var(f"{vehicle}_route_{i}") for i in range(len(columns))]
        model.add_exactly_one(chosen)
        variables[vehicle] = chosen
    entries = [
        (vehicle, index, column, variables[vehicle][index])
        for vehicle, columns in candidates.items()
        for index, column in enumerate(columns)
    ]
    for locker, capacity in available.items():
        model.add(sum(column.slots[locker] * var for _, _, column, var in entries) <= capacity)

    groups = defaultdict(list)
    for vehicle, _, column, var in entries:
        if column.charge is not None:
            groups[vehicle, column.charge].append((var, column.latest_charge_start))
    booking_order = sorted(groups, key=lambda key: (key[1].request_min, key[0]))
    earlier_ends = []
    charge_starts = {}
    intervals = []
    for index, (vehicle, charge) in enumerate(booking_order):
        group = groups[vehicle, charge]
        presence = model.new_bool_var(f"charge_{index}")
        start = model.new_int_var(charge.start_min, horizon, f"arrival_or_queue_{index}")
        end = model.new_int_var(0, horizon, f"charge_end_{index}")
        model.add(presence == sum(var for var, _ in group))
        model.add_max_equality(start, [charge.start_min, *earlier_ends])
        model.add(end == 0).only_enforce_if(~presence)
        for route, latest_start in group:
            model.add(start <= latest_start).only_enforce_if(route)
        intervals.append(model.new_optional_interval_var(
            start, charge.end_min - charge.start_min, end, presence, f"plug_{index}",
        ))
        charge_starts[vehicle, charge] = start
        earlier_ends.append(end)
    if intervals:
        model.add_no_overlap(intervals)

    objectives = [
        ("served", True, sum(column.served * var for _, _, column, var in entries)),
        ("travel_min", False, sum(column.travel_min * var for _, _, column, var in entries)),
        ("travel_wh", False, sum(column.travel_wh * var for _, _, column, var in entries)),
    ]
    objectives.extend(
        (f"route_rank:{vehicle}", False, sum(rank * var for rank, var in enumerate(chosen)))
        for vehicle, chosen in variables.items()
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    end = perf_counter() + timeout
    selected = None
    shifts = {}
    stages = []
    for name, maximize, expression in objectives:
        remaining = end - perf_counter()
        if remaining <= 0:
            stages.append({"stage": name, "status": "TIME_LIMIT"})
            break
        if maximize:
            model.maximize(expression)
        else:
            model.minimize(expression)
        solver.parameters.max_time_in_seconds = remaining
        status = solver.solve(model)
        stage = {"stage": name, "status": solver.status_name(status)}
        stages.append(stage)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        optimum = int(solver.value(expression))
        stage["value"] = optimum
        selected = {
            vehicle: next(i for i, var in enumerate(chosen) if solver.boolean_value(var))
            for vehicle, chosen in variables.items()
        }
        shifts = {}
        for vehicle, index in selected.items():
            charge = candidates[vehicle][index].charge
            if charge is not None:
                shifts[vehicle] = solver.value(charge_starts[vehicle, charge]) - charge.start_min
        if status != cp_model.OPTIMAL:
            break
        model.add(expression == optimum)
    return selected, shifts, stages, all(stage["status"] == "OPTIMAL" for stage in stages)


def _certify(environment: OutageEnv, selected: dict[str, _Column] | None, shifts: dict[str, int]):
    sequence = []
    cursors = Counter()
    charged = set()
    rule = RuleBased(environment.scenario)
    while (decision := environment.observation()) is not None:
        vehicle = decision.vehicle_id
        if selected is None:
            index = rule.choose(decision)
        else:
            cursor = cursors[vehicle]
            route = selected[vehicle].actions
            if cursor >= len(route):
                raise _CertificateError(f"selected route ended before return for {vehicle}")
            action = route[cursor]
            if action.kind == "charge":
                charged.add(vehicle)
            if vehicle in charged and shifts.get(vehicle, 0):
                action = replace(action, completion_min=action.completion_min + shifts[vehicle])
            if action not in decision.actions:
                raise _CertificateError(
                    f"selected route action is absent at {decision.time_min} min for {vehicle}"
                )
            index = decision.actions.index(action)
            cursors[vehicle] += 1
        sequence.append(_Dispatch(vehicle, decision.time_min, decision.actions[index]))
        environment.step(index)
    result = environment.result()
    if selected is not None:
        expected = {
            "served_parcels": sum(column.served for column in selected.values()),
            "travel_min": sum(column.travel_min for column in selected.values()),
            "travel_wh": sum(column.travel_wh for column in selected.values()),
        }
        if (
            any(cursors[vehicle] != len(column.actions) for vehicle, column in selected.items())
            or any(result["metrics"][key] != value for key, value in expected.items())
        ):
            raise _CertificateError("completed rollout differs from the selected route columns")
    return tuple(sequence), result


def repair(
    scenario: Scenario, *, charger_rate_wh_per_min: int, charger_capacity_wh: int,
    preassigned_slots: dict[str, str] | None = None, dispatch_after_min: int | None = None,
    time_limit_s: float = 0.35, columns_per_vehicle: int = 512,
) -> RepairPlan:
    """Select certified routes under fixed custody and at most one depot charge per EV.

    Column generation has a separate per-vehicle cap. The wall-time limit is
    shared across all CP-SAT objectives, including deterministic rank ties.
    Rule-based fallback, where needed, uses the same full environment and is
    explicitly labeled in the receipt.
    """
    from ortools import __version__ as ortools_version

    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("CP-SAT time limit must be positive and finite")
    if type(columns_per_vehicle) is not int or columns_per_vehicle < 1:
        raise ValueError("column count limit must be a positive integer")
    options = {
        "charger_rate_wh_per_min": charger_rate_wh_per_min,
        "charger_capacity_wh": charger_capacity_wh,
        "dispatch_after_min": dispatch_after_min,
    }
    full_options = {**options, "preassigned_slots": preassigned_slots}
    environment = OutageEnv(scenario, **full_options)
    slots = {} if preassigned_slots is None else dict(preassigned_slots)
    holds = Counter(slots.values())
    available = {
        locker_id: locker.capacity - holds[locker_id]
        for locker_id, locker in scenario.lockers.items() if locker.available
    }
    candidates = {}
    capped = []
    for vehicle in sorted(scenario.vehicles, key=lambda vehicle: vehicle.id):
        columns, hit_limit = _vehicle_columns(scenario, vehicle, options, slots, columns_per_vehicle)
        candidates[vehicle.id] = columns
        if hit_limit:
            capped.append(vehicle.id)
    selected_ids, shifts, stages, all_optimal = _select_columns(
        candidates, available, scenario.horizon_min, time_limit_s,
    )
    selected = (
        {vehicle: candidates[vehicle][index] for vehicle, index in selected_ids.items()}
        if selected_ids is not None else None
    )
    fallback_reason = "cp_sat_no_incumbent" if selected is None else None
    try:
        sequence, result = _certify(environment, selected, shifts)
    except _CertificateError as exc:
        fallback_reason = str(exc)
        sequence, result = _certify(OutageEnv(scenario, **full_options), None, {})
    return RepairPlan(sequence, {
        "engine": "ortools_cp_sat",
        "ortools_version": ortools_version,
        "strategy": "cp_sat" if fallback_reason is None else "rule_based_fallback",
        "fallback_reason": fallback_reason,
        "solver_status": stages[-1]["status"],
        "optimization_stages": stages,
        "all_objectives_optimal_within_columns": all_optimal,
        "time_limit_s": time_limit_s,
        "columns_per_vehicle_limit": columns_per_vehicle,
        "candidate_counts": {vehicle: len(columns) for vehicle, columns in candidates.items()},
        "column_capped_vehicles": capped,
        "max_charges_per_vehicle_in_columns": 1,
        "selected_column_indices": selected_ids,
        "selected_charge_queue_wait_min": shifts,
        "certified_decision_count": len(sequence),
        "certified_metrics": result["metrics"],
        "certified_unserved_parcel_ids": result["unserved_parcel_ids"],
        "certified_dispatches": [
            {"vehicle_id": entry.vehicle_id, "time_min": entry.time_min, "action": asdict(entry.action)}
            for entry in sequence
        ],
    })
