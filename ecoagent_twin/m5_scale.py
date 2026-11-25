"""Scale a fixed synthetic outage workload on the existing site-cost matrix."""

from __future__ import annotations

from copy import deepcopy

from .model import Scenario


FLEET_SIZES = (2, 4, 8)
LOCKER_LAYOUTS = ("replicated", "pooled")


def _clone_id(prefix: str, original: str, occupied: set[str]) -> str:
    candidate = f"{prefix}-{original}"
    while candidate in occupied:
        candidate = f"copy-{candidate}"
    occupied.add(candidate)
    return candidate


def _bindings(originals, occupied: set[str], copy_index: int, *, replicate: bool) -> dict[str, str]:
    if not copy_index or not replicate:
        return {original: original for original in originals}
    prefix = f"rep{copy_index + 1:02d}"
    return {original: _clone_id(prefix, original, occupied) for original in originals}


def scale_workload(
    raw: dict, fleet_size: int, *, locker_layout: str = "replicated",
) -> tuple[dict, dict]:
    """Clone onboard workloads and capacities while keeping their observed costs.

    Replicated banks are distinct resources at the original physical sites.
    Parcels may visit the corresponding banks in their own copy, with one
    closed locker, shared depot, and unchanged charger model. Pooled banks
    instead preserve the original IDs and let all copies share scaled capacity.
    Neither layout simulates new geography or traffic interactions.
    """
    if locker_layout not in LOCKER_LAYOUTS:
        raise ValueError(f"unknown locker layout: {locker_layout}")
    source = Scenario.from_dict(raw)
    if type(fleet_size) is not int or fleet_size < len(source.vehicles) or fleet_size % len(source.vehicles):
        raise ValueError("fleet size must be a positive integer multiple of the source fleet size")
    copies = fleet_size // len(source.vehicles)
    snapshot = deepcopy(raw)
    snapshot["id"] = f"{source.id}-fleet{fleet_size}-{locker_layout}"
    available = tuple(locker_id for locker_id, locker in source.lockers.items() if locker.available)
    base_nodes = tuple(raw["travel_minutes"])
    node_origins = dict(zip(base_nodes, base_nodes, strict=True))
    used_nodes = set(base_nodes)
    used_vehicles = {vehicle.id for vehicle in source.vehicles}
    used_parcels = {parcel.id for parcel in source.parcels}
    capacities = snapshot["locker_free_capacity"]
    snapshots_vehicles = []
    snapshots_parcels = []
    replicas = []
    if locker_layout == "pooled":
        for locker_id in available:
            capacities[locker_id] *= copies
    for index in range(copies):
        vehicles = _bindings((v.id for v in source.vehicles), used_vehicles, index, replicate=True)
        parcels = _bindings((p.id for p in source.parcels), used_parcels, index, replicate=True)
        lockers = _bindings(available, used_nodes, index, replicate=locker_layout == "replicated")
        lockers[source.outage_locker] = source.outage_locker
        for original, replicated in lockers.items():
            if original != replicated:
                capacities[replicated] = source.lockers[original].capacity
                node_origins[replicated] = original
        for row in raw["vehicles"]:
            vehicle = deepcopy(row)
            vehicle["id"] = vehicles[row["id"]]
            snapshots_vehicles.append(vehicle)
        for row in raw["parcels_in_original_plan_order"]:
            parcel = deepcopy(row)
            parcel["id"] = parcels[row["id"]]
            parcel["vehicle_id"] = vehicles[row["vehicle_id"]]
            parcel["original_locker"] = lockers[row["original_locker"]]
            parcel["alternatives"] = [lockers[locker_id] for locker_id in row["alternatives"]]
            snapshots_parcels.append(parcel)
        replicas.append({
            "replica": index + 1, "vehicles": vehicles, "parcels": parcels, "lockers": lockers,
        })
    snapshot["vehicles"] = snapshots_vehicles
    snapshot["parcels_in_original_plan_order"] = snapshots_parcels
    for field in ("travel_minutes", "travel_wh"):
        matrix = raw[field]
        snapshot[field] = {
            start: {end: matrix[base_start][base_end] for end, base_end in node_origins.items()}
            for start, base_start in node_origins.items()
        }
    result = Scenario.from_dict(snapshot)
    return snapshot, {
        "source_case_id": source.id,
        "fleet_size": len(result.vehicles),
        "parcel_count": len(result.parcels),
        "locker_count": len(result.lockers),
        "open_locker_count": sum(locker.available for locker in result.lockers.values()),
        "open_slots": sum(locker.capacity for locker in result.lockers.values() if locker.available),
        "scale_factor": copies,
        "locker_layout": locker_layout,
        "site_profile": "unchanged_directed_site_costs_with_colocated_logical_banks",
        "shared_infrastructure": {"depot": source.depot, "closed_locker": source.outage_locker},
        "node_origins": node_origins,
        "replicas": replicas,
    }
