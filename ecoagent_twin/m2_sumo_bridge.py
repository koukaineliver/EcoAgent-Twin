"""Convert directed SUMO replay trips into scenario travel inputs."""

from copy import deepcopy

from .model import Scenario


def apply_travel_matrix(raw: dict, trips: list[dict]) -> dict:
    """Replace the travel matrices, filling unprobed self-arcs with zero."""
    start_nodes = {vehicle["node"] for vehicle in raw["vehicles"]}
    nodes = sorted({raw["depot"], *raw["locker_free_capacity"], *start_nodes})
    expected = {(origin, destination) for origin in nodes for destination in nodes}
    arcs: dict[tuple[str, str], tuple[int, int]] = {}
    required = {"origin", "destination", "travel_min", "travel_wh"}
    for trip in trips:
        if not isinstance(trip, dict) or not required <= trip.keys():
            raise ValueError(
                "each trip must provide origin, destination, travel_min, and travel_wh"
            )
        origin, destination = trip["origin"], trip["destination"]
        arc = origin, destination
        if not isinstance(origin, str) or not isinstance(destination, str) or arc not in expected:
            raise ValueError(f"unexpected travel arc: {origin!r} -> {destination!r}")
        if arc in arcs:
            raise ValueError(f"duplicate travel arc: {origin} -> {destination}")
        minutes, watt_hours = trip["travel_min"], trip["travel_wh"]
        if type(minutes) is not int or type(watt_hours) is not int:
            raise ValueError(
                f"travel minutes and watt-hours must be integers: {origin} -> {destination}"
            )
        if origin == destination:
            if minutes != 0 or watt_hours != 0:
                raise ValueError(f"diagonal travel values must be zero: {origin} -> {destination}")
        elif minutes <= 0 or watt_hours <= 0:
            raise ValueError(
                f"off-diagonal travel values must be positive: {origin} -> {destination}"
            )
        arcs[arc] = minutes, watt_hours

    for node in nodes:
        arcs.setdefault((node, node), (0, 0))
    missing = expected - arcs.keys()
    if missing:
        origin, destination = min(missing)
        raise ValueError(f"missing travel arc: {origin} -> {destination}")

    result = deepcopy(raw)
    result["travel_minutes"] = {
        origin: {destination: arcs[origin, destination][0] for destination in nodes}
        for origin in nodes
    }
    result["travel_wh"] = {
        origin: {destination: arcs[origin, destination][1] for destination in nodes}
        for origin in nodes
    }
    Scenario.from_dict(result)
    return result
