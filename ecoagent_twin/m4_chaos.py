"""Deterministic alert-time stresses on the synthetic SUMO-derived OD snapshot."""

from __future__ import annotations

from copy import deepcopy
from random import Random

from .m2_simulator import OutageEnv
from .model import Scenario


KINDS = (
    "locker_outage",
    "locker_saturation",
    "charger_outage",
    "road_blockage",
    "demand_surge",
    "compound",
)
SEVERITIES = ("moderate", "severe")
COMPOUND_COMPONENTS = KINDS[1:-1]


def _saturate(raw: dict, severity: str, rng: Random) -> dict:
    capacities = raw["locker_free_capacity"]
    closed = raw["outage"]["locker_id"]
    choices = sorted(locker for locker, capacity in capacities.items() if locker != closed and capacity)
    count = min(len(choices), 1 if severity == "moderate" else 2)
    reduction = 1 if severity == "moderate" else 2
    changes = []
    for locker_id in rng.sample(choices, count):
        before = capacities[locker_id]
        capacities[locker_id] = max(0, before - reduction)
        changes.append({
            "locker_id": locker_id,
            "free_slots_before": before,
            "free_slots_after": capacities[locker_id],
        })
    return {"capacity_changes": changes}


def _disable_charger(raw: dict, settings: dict, severity: str, rng: Random) -> dict:
    previous_rate = settings["charger_rate_wh_per_min"]
    settings["charger_rate_wh_per_min"] = 0
    depot = raw["depot"]
    headroom = (140, 220) if severity == "moderate" else (15, 75)
    battery = []
    for vehicle in sorted(raw["vehicles"], key=lambda row: row["id"]):
        safe_return_wh = vehicle["reserve_wh"] + raw["travel_wh"][vehicle["node"]][depot]
        before = vehicle["battery_wh"]
        vehicle["battery_wh"] = min(before, safe_return_wh + rng.randint(*headroom))
        battery.append({
            "vehicle_id": vehicle["id"],
            "battery_before_wh": before,
            "battery_after_wh": vehicle["battery_wh"],
            "direct_return_and_reserve_wh": safe_return_wh,
        })
    return {
        "charger_rate_before_wh_per_min": previous_rate,
        "charger_rate_after_wh_per_min": 0,
        "charger_capacity_wh": settings["charger_capacity_wh"],
        "initial_battery_context": "low_battery_with_direct_depot_return_preserved",
        "battery_changes": battery,
    }


def _detour(raw: dict, severity: str, rng: Random) -> dict:
    times = raw["travel_minutes"]
    energy = raw["travel_wh"]
    closed = raw["outage"]["locker_id"]
    positions = {vehicle["id"]: vehicle["node"] for vehicle in raw["vehicles"]}
    lockers = {
        locker_id for locker_id, capacity in raw["locker_free_capacity"].items()
        if locker_id != closed and capacity > 0
    }
    primary = {
        (positions[parcel["vehicle_id"]], locker_id)
        for parcel in raw["parcels_in_original_plan_order"]
        for locker_id in (parcel["original_locker"], *parcel["alternatives"])
        if locker_id in lockers and locker_id != positions[parcel["vehicle_id"]]
    }
    secondary = {(start, end) for start in lockers for end in lockers if start != end} - primary
    first, second = sorted(primary), sorted(secondary)
    rng.shuffle(first)
    rng.shuffle(second)
    limit = 1 if severity == "moderate" else 2
    vias = sorted(set(times) - {closed})
    replacements = []
    replaced = set()
    components = set()
    for origin, destination in (*first, *second):
        if (origin, destination) in components:
            continue
        before_min = times[origin][destination]
        routes = []
        for via in vias:
            if via in (origin, destination) or {(origin, via), (via, destination)} & replaced:
                continue
            after_min = times[origin][via] + times[via][destination]
            if after_min > before_min:
                after_wh = energy[origin][via] + energy[via][destination]
                routes.append((after_min, after_wh, via))
        if not routes:
            continue
        after_min, after_wh, via = min(routes)
        replacements.append({
            "origin": origin,
            "destination": destination,
            "via": via,
            "travel_before_min": before_min,
            "travel_after_min": after_min,
            "traction_before_wh": energy[origin][destination],
            "traction_after_wh": after_wh,
        })
        replaced.add((origin, destination))
        components.update(((origin, via), (via, destination)))
        if len(replacements) == limit:
            break
    for detour in replacements:
        origin, destination = detour["origin"], detour["destination"]
        times[origin][destination] = detour["travel_after_min"]
        energy[origin][destination] = detour["traction_after_wh"]
    return {
        "representation": "synthetic_directed_od_detour_over_measured_sites",
        "requested_replacements": limit,
        "replacements": replacements,
    }


def _surge(raw: dict, severity: str, rng: Random) -> dict:
    parcels = raw["parcels_in_original_plan_order"]
    before = len(parcels)
    used_ids = {parcel["id"] for parcel in parcels}
    by_vehicle = {
        vehicle_id: sorted(
            (parcel for parcel in parcels if parcel["vehicle_id"] == vehicle_id),
            key=lambda row: row["id"],
        )
        for vehicle_id in sorted({parcel["vehicle_id"] for parcel in parcels})
    }
    carriers = list(by_vehicle)
    rng.shuffle(carriers)
    added = []
    number = 1
    for index in range(1 if severity == "moderate" else 2):
        carrier = carriers[index % len(carriers)]
        parcel = deepcopy(rng.choice(by_vehicle[carrier]))
        while f"surge-{number:02d}" in used_ids:
            number += 1
        parcel["id"] = f"surge-{number:02d}"
        used_ids.add(parcel["id"])
        parcels.append(parcel)
        added.append(deepcopy(parcel))
    return {
        "onboard_before": before,
        "onboard_after": len(parcels),
        "denominator_scope": "fixed_at_alert_including_newly_disclosed_onboard_parcels",
        "added_parcels": added,
    }


def inject(
    raw: dict, kind: str, severity: str, seed: int, *,
    charger_rate_wh_per_min: int, charger_capacity_wh: int,
) -> tuple[dict, dict, list[dict]]:
    """Produce a paired synthetic treatment world and the alerts visible at dispatch."""
    if kind not in KINDS:
        raise ValueError(f"unknown chaos kind: {kind}")
    if severity not in SEVERITIES:
        raise ValueError(f"unknown chaos severity: {severity}")
    if type(seed) is not int:
        raise ValueError("chaos seed must be an integer")
    if (
        type(charger_rate_wh_per_min) is not int or type(charger_capacity_wh) is not int
        or charger_rate_wh_per_min < 0 or charger_capacity_wh < 0
        or (charger_rate_wh_per_min > 0 and charger_capacity_wh == 0)
    ):
        raise ValueError("charger rate and capacity must be compatible nonnegative integers")
    base = Scenario.from_dict(raw)
    world = deepcopy(raw)
    world["id"] = f"{base.id}-{kind}-{severity}-d{seed}"
    settings = {
        "charger_rate_wh_per_min": charger_rate_wh_per_min,
        "charger_capacity_wh": charger_capacity_wh,
    }
    events = []

    def alert(event_kind: str, payload: dict) -> None:
        events.append({
            "time_min": base.outage_min,
            "kind": event_kind,
            "payload": {"severity": severity, **payload},
        })

    alert("locker_outage", {
        "locker_id": base.outage_locker, "baseline_fixture": True,
    })
    if kind == "compound":
        alert("compound", {"components": list(COMPOUND_COMPONENTS)})
    parts = COMPOUND_COMPONENTS if kind == "compound" else (kind,)
    for component in parts:
        rng = Random(f"ecoagent-m4-v1:{component}:{seed}")
        if component == "locker_saturation":
            alert(component, _saturate(world, severity, rng))
        elif component == "charger_outage":
            alert(component, _disable_charger(world, settings, severity, rng))
        elif component == "road_blockage":
            alert(component, _detour(world, severity, rng))
        elif component == "demand_surge":
            alert(component, _surge(world, severity, rng))
    OutageEnv(Scenario.from_dict(world), **settings)
    return world, settings, events
