"""Reference controllers for an outage after vehicles have left the depot."""

from collections import Counter
from math import factorial, prod

from .model import Parcel, Plan, RouteOutcome, Scenario, Stop, Vehicle, evaluate_route

MAX_EXACT_PARCELS = 8
MAX_EXACT_SEARCH_STATES = 250_000


def _try_route(
    scenario: Scenario, vehicle: Vehicle, stops: tuple[Stop, ...]
) -> RouteOutcome | None:
    try:
        return evaluate_route(scenario, vehicle, stops)
    except ValueError:
        return None


def _available_lockers(scenario: Scenario, parcel: Parcel) -> tuple[str, ...]:
    candidates = dict.fromkeys((parcel.original_locker, *parcel.alternatives))
    return tuple(
        locker
        for locker in candidates
        if scenario.lockers[locker].available and scenario.lockers[locker].capacity > 0
    )


def keep_plan(scenario: Scenario) -> Plan:
    """Visit available original lockers in dispatch order when still feasible."""
    vehicles = {vehicle.id: vehicle for vehicle in scenario.vehicles}
    plan: Plan = {vehicle_id: () for vehicle_id in vehicles}
    occupied: Counter[str] = Counter()
    for parcel in scenario.parcels:
        locker = scenario.lockers[parcel.original_locker]
        if not locker.available or occupied[parcel.original_locker] >= locker.capacity:
            continue
        stops = (*plan[parcel.vehicle_id], Stop(parcel.id, parcel.original_locker))
        if _try_route(scenario, vehicles[parcel.vehicle_id], stops) is not None:
            plan[parcel.vehicle_id] = stops
            occupied[parcel.original_locker] += 1
    return plan


def rule_based(scenario: Scenario) -> Plan:
    """Insert the earliest-deadline parcel at its shortest feasible locker and position."""
    vehicles = {vehicle.id: vehicle for vehicle in scenario.vehicles}
    plan: Plan = {vehicle_id: () for vehicle_id in vehicles}
    occupied: Counter[str] = Counter()
    for parcel in sorted(scenario.parcels, key=lambda parcel: (parcel.deadline_min, parcel.id)):
        current = plan[parcel.vehicle_id]
        best: tuple[RouteOutcome, str] | None = None
        for locker_id in _available_lockers(scenario, parcel):
            if occupied[locker_id] >= scenario.lockers[locker_id].capacity:
                continue
            for position in range(len(current) + 1):
                stops = (*current[:position], Stop(parcel.id, locker_id), *current[position:])
                outcome = _try_route(scenario, vehicles[parcel.vehicle_id], stops)
                if outcome is not None and (
                    best is None or outcome.travel_min < best[0].travel_min
                ):
                    best = outcome, locker_id
        if best is not None:
            outcome, locker_id = best
            plan[parcel.vehicle_id] = outcome.stops
            occupied[locker_id] += 1
    return plan


def _stop_ids(stops: tuple[Stop, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((stop.parcel_id, stop.locker_id) for stop in stops)


def _route_prefix_bound(scenario: Scenario, parcels: tuple[Parcel, ...]) -> int:
    """Count possible ordered prefixes before checking capacity or route feasibility."""
    assignment_counts = [1]
    for parcel in parcels:
        lockers = len(_available_lockers(scenario, parcel))
        assignment_counts.append(0)
        for size in range(len(assignment_counts) - 1, 0, -1):
            assignment_counts[size] += lockers * assignment_counts[size - 1]
    return sum(factorial(size) * count for size, count in enumerate(assignment_counts))


def _vehicle_routes(
    scenario: Scenario, vehicle: Vehicle, parcels: tuple[Parcel, ...]
) -> dict[tuple[tuple[str, int], ...], RouteOutcome]:
    options = {parcel.id: _available_lockers(scenario, parcel) for parcel in parcels}
    best_routes: dict[tuple[tuple[str, int], ...], RouteOutcome] = {}
    used: Counter[str] = Counter()

    def enumerate_routes(stops: tuple[Stop, ...], remaining: tuple[Parcel, ...]) -> None:
        outcome = _try_route(scenario, vehicle, stops)
        if outcome is not None:
            slots = tuple(sorted(used.items()))
            previous = best_routes.get(slots)
            if previous is None or (outcome.travel_min, _stop_ids(outcome.stops)) < (
                previous.travel_min,
                _stop_ids(previous.stops),
            ):
                best_routes[slots] = outcome

        # An infeasible return from this prefix can become feasible after a later stop.
        for index, parcel in enumerate(remaining):
            for locker_id in options[parcel.id]:
                if used[locker_id] >= scenario.lockers[locker_id].capacity:
                    continue
                used[locker_id] += 1
                enumerate_routes(
                    (*stops, Stop(parcel.id, locker_id)),
                    (*remaining[:index], *remaining[index + 1 :]),
                )
                used[locker_id] -= 1
                if not used[locker_id]:
                    del used[locker_id]

    enumerate_routes((), parcels)
    return best_routes


def exact_replan(scenario: Scenario) -> Plan:
    """Maximize service, then minimize travel, for a bounded exhaustive benchmark.

    Up to eight parcels are accepted. A second limit rejects instances with too
    many possible route prefixes or route combinations, even at that parcel count.
    """
    if len(scenario.parcels) > MAX_EXACT_PARCELS:
        raise ValueError(f"Exact replanning is limited to {MAX_EXACT_PARCELS} parcels")

    parcels_by_vehicle: dict[str, list[Parcel]] = {}
    for parcel in scenario.parcels:
        parcels_by_vehicle.setdefault(parcel.vehicle_id, []).append(parcel)
    active = [
        (vehicle, tuple(parcels_by_vehicle[vehicle.id]))
        for vehicle in scenario.vehicles
        if vehicle.id in parcels_by_vehicle
    ]
    if sum(_route_prefix_bound(scenario, parcels) for _, parcels in active) > MAX_EXACT_SEARCH_STATES:
        raise ValueError(f"Exact replanning exceeds {MAX_EXACT_SEARCH_STATES:,} possible route prefixes")

    routes_by_vehicle = [_vehicle_routes(scenario, vehicle, parcels) for vehicle, parcels in active]
    if prod(len(routes) for routes in routes_by_vehicle) > MAX_EXACT_SEARCH_STATES:
        raise ValueError(f"Exact replanning exceeds {MAX_EXACT_SEARCH_STATES:,} route combinations")
    occupied: Counter[str] = Counter()
    best_key = None
    best_routes: tuple[RouteOutcome, ...] = ()

    def combine_routes(routes: tuple[RouteOutcome, ...], served: int, travel: int) -> None:
        nonlocal best_key, best_routes
        if len(routes) == len(routes_by_vehicle):
            key = (-served, travel, tuple(_stop_ids(route.stops) for route in routes))
            if best_key is None or key < best_key:
                best_key, best_routes = key, routes
            return

        for slots, outcome in routes_by_vehicle[len(routes)].items():
            if any(
                occupied[locker_id] + count > scenario.lockers[locker_id].capacity
                for locker_id, count in slots
            ):
                continue
            for locker_id, count in slots:
                occupied[locker_id] += count
            combine_routes(
                (*routes, outcome), served + len(outcome.stops), travel + outcome.travel_min
            )
            for locker_id, count in slots:
                occupied[locker_id] -= count

    combine_routes((), 0, 0)
    if best_key is None:
        raise ValueError("No feasible combination of vehicle routes")
    selected = {vehicle.id: route.stops for (vehicle, _), route in zip(active, best_routes)}
    return {vehicle.id: selected.get(vehicle.id, ()) for vehicle in scenario.vehicles}
