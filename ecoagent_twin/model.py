"""A post-dispatch locker outage with explicit route feasibility checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Locker:
    capacity: int
    available: bool


@dataclass(frozen=True)
class Vehicle:
    id: str
    node: str
    battery_wh: int
    reserve_wh: int


@dataclass(frozen=True)
class Parcel:
    id: str
    vehicle_id: str
    original_locker: str
    alternatives: tuple[str, ...]
    deadline_min: int


@dataclass(frozen=True)
class Stop:
    parcel_id: str
    locker_id: str


Plan = dict[str, tuple[Stop, ...]]


@dataclass(frozen=True)
class Scenario:
    id: str
    depot: str
    outage_locker: str
    outage_min: int
    horizon_min: int
    service_min: int
    vehicles: tuple[Vehicle, ...]
    parcels: tuple[Parcel, ...]
    lockers: dict[str, Locker]
    travel_time: dict[tuple[str, str], int]
    travel_energy: dict[tuple[str, str], int]

    @classmethod
    def from_dict(cls, raw: dict) -> Scenario:
        if raw.get("schema") != 1 or raw.get("data_basis") != "synthetic":
            raise ValueError("expected a version 1 synthetic scenario")
        outage_locker = raw["outage"]["locker_id"]
        outage_min = raw["outage"]["time_min"]
        horizon = raw["horizon_min"]
        service_min = raw["service_min_per_parcel"]
        depot = raw["depot"]
        lockers = {
            name: Locker(capacity, name != outage_locker)
            for name, capacity in raw["locker_free_capacity"].items()
        }
        vehicles = tuple(Vehicle(**row) for row in raw["vehicles"])
        parcels = tuple(
            Parcel(
                row["id"], row["vehicle_id"], row["original_locker"],
                tuple(row["alternatives"]), row["deadline_min"],
            )
            for row in raw["parcels_in_original_plan_order"]
        )
        vehicle_ids = {vehicle.id for vehicle in vehicles}
        if len(vehicle_ids) != len(vehicles) or not vehicles:
            raise ValueError("vehicle IDs must be unique and nonempty")
        if len({parcel.id for parcel in parcels}) != len(parcels) or not parcels:
            raise ValueError("parcel IDs must be unique and nonempty")
        if not all(type(n) is int for n in (outage_min, horizon, service_min)):
            raise ValueError("scenario times must use integer minutes")
        if not 0 <= outage_min < horizon or service_min <= 0:
            raise ValueError("invalid outage time, horizon, or service duration")
        if depot in lockers or outage_locker not in lockers:
            raise ValueError("specify a distinct depot and a known outage locker")
        if any(type(locker.capacity) is not int or locker.capacity < 0 for locker in lockers.values()):
            raise ValueError("locker capacity must be a nonnegative integer")
        for vehicle in vehicles:
            if any(type(n) is not int for n in (vehicle.battery_wh, vehicle.reserve_wh)):
                raise ValueError("battery and reserve must use integer watt-hours")
            if not 0 <= vehicle.reserve_wh <= vehicle.battery_wh:
                raise ValueError(f"invalid battery snapshot for {vehicle.id}")
        for parcel in parcels:
            if parcel.vehicle_id not in vehicle_ids:
                raise ValueError(f"unknown carrying vehicle for {parcel.id}")
            if not {parcel.original_locker, *parcel.alternatives} <= lockers.keys():
                raise ValueError(f"unknown locker for {parcel.id}")
            if len({parcel.original_locker, *parcel.alternatives}) != len(parcel.alternatives) + 1:
                raise ValueError(f"duplicate locker option for {parcel.id}")
            if type(parcel.deadline_min) is not int or not outage_min <= parcel.deadline_min <= horizon:
                raise ValueError(f"invalid deadline for {parcel.id}")

        nodes = {depot, *lockers, *(vehicle.node for vehicle in vehicles)}
        scenario = cls(
            raw["id"], depot, outage_locker, outage_min, horizon, service_min,
            vehicles, parcels, lockers,
            _arc_matrix(raw["travel_minutes"], nodes, "travel_minutes"),
            _arc_matrix(raw["travel_wh"], nodes, "travel_wh"),
        )
        for vehicle in vehicles:
            evaluate_route(scenario, vehicle, ())
        return scenario


def _arc_matrix(rows: dict, nodes: set[str], name: str) -> dict[tuple[str, str], int]:
    if rows.keys() != nodes or any(row.keys() != nodes for row in rows.values()):
        raise ValueError(f"{name} must contain every directed node pair")
    matrix = {}
    for start, row in rows.items():
        for end, value in row.items():
            if type(value) is not int or value < 0 or (start == end and value != 0):
                raise ValueError(f"invalid arc in {name}: {start} -> {end}")
            matrix[start, end] = value
    return matrix


@dataclass(frozen=True)
class RouteOutcome:
    stops: tuple[Stop, ...]
    deposits_min: tuple[int, ...]
    travel_min: int
    travel_wh: int
    return_min: int
    remaining_wh: int


def evaluate_route(scenario: Scenario, vehicle: Vehicle, stops: tuple[Stop, ...]) -> RouteOutcome:
    """Validate a route from the outage snapshot through its return to depot."""
    parcels = {parcel.id: parcel for parcel in scenario.parcels if parcel.vehicle_id == vehicle.id}
    if len({stop.parcel_id for stop in stops}) != len(stops):
        raise ValueError(f"route duplicates a parcel for {vehicle.id}")

    travel_min = travel_wh = 0
    now, node = scenario.outage_min, vehicle.node
    deposits = []
    use = Counter()
    for stop in stops:
        parcel, locker = parcels.get(stop.parcel_id), scenario.lockers.get(stop.locker_id)
        if parcel is None or locker is None or not locker.available:
            raise ValueError(f"infeasible parcel or locker: {stop}")
        if stop.locker_id not in (parcel.original_locker, *parcel.alternatives):
            raise ValueError(f"locker is ineligible for {parcel.id}")
        use[stop.locker_id] += 1
        if use[stop.locker_id] > locker.capacity:
            raise ValueError(f"route exceeds capacity at {stop.locker_id}")
        arc = node, stop.locker_id
        travel_min += scenario.travel_time[arc]
        travel_wh += scenario.travel_energy[arc]
        now += scenario.travel_time[arc] + scenario.service_min
        if now > parcel.deadline_min:
            raise ValueError(f"parcel deadline missed: {parcel.id}")
        deposits.append(now)
        node = stop.locker_id

    return_arc = node, scenario.depot
    travel_min += scenario.travel_time[return_arc]
    travel_wh += scenario.travel_energy[return_arc]
    now += scenario.travel_time[return_arc]
    remaining = vehicle.battery_wh - travel_wh
    if now > scenario.horizon_min or remaining < vehicle.reserve_wh:
        raise ValueError(f"return deadline or battery reserve violated for {vehicle.id}")
    return RouteOutcome(stops, tuple(deposits), travel_min, travel_wh, now, remaining)


def evaluate_plan(scenario: Scenario, plan: Plan) -> dict[str, RouteOutcome]:
    if plan.keys() != {vehicle.id for vehicle in scenario.vehicles}:
        raise ValueError("plan must contain exactly one route for each vehicle")
    use = Counter(stop.locker_id for stops in plan.values() for stop in stops)
    outcomes = {
        vehicle.id: evaluate_route(scenario, vehicle, plan[vehicle.id])
        for vehicle in scenario.vehicles
    }
    if any(count > scenario.lockers[locker_id].capacity for locker_id, count in use.items()):
        raise ValueError("plan exceeds shared locker capacity")
    return outcomes
