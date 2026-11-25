"""Replay matched modeled locker disruptions on actual Amazon route windows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from statistics import fmean, median
from time import perf_counter
from typing import Callable

from .empirical_amazon import file_sha256, read_records
from .empirical_energy import EVProfile, PROFILES, assumptions, carbon_receipt, leg_wh, road_proxy_km
from .m2_offer_experiment import DEFAULT_PROTOCOL as OFFER_PROTOCOL
from .m2_offers import UtilityProfile
from .m2_simulator import OutageEnv
from .m3_experiment import DEFAULT_MODEL, _menu
from .m3_route_repair import repair
from .m3_supervisor import DecisionView, _freeze, _graph
from .m4_experiment import _take_decision
from .m5_deadline import DeadlineModel
from .m5_experiment import _verify_selections
from .m5_local_model import OrderInvariantMenuModel
from .m6_incentive import negotiate
from .model import Scenario


METHODS = ("preserve_observed_order", "frozen_mappo", "agentic_supervisor", "deterministic_supervisor")
BATTERY_CONTEXTS = ("standard_departure_state", "low_soc_at_alert")
CP_LIMITS = {"time_limit_s": 0.35, "columns_per_vehicle": 1024}
DECISION_CONFIG = {"supervisor_budget_s": 2.0}
ARTIFACTS = ("worlds.jsonl", "observations.jsonl", "summary.json", "policy.pt", "offer_protocol.json")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def route_window(record: dict) -> tuple[dict[str, str], list[str], int]:
    stops = record["route"]["stops"]
    actual = record["actual_sequence"]["actual"]
    ordered = sorted(actual, key=actual.__getitem__)
    if set(ordered) != set(stops) or len(set(actual.values())) != len(ordered):
        raise ValueError("observed sequence does not cover the official route stops exactly once")
    if len(ordered) < 9 or stops[ordered[0]]["type"] != "Station":
        raise ValueError("expected the official sequence to begin at a station and have a delivery prefix")
    packages = record["packages"]
    alert_index = max(1, len(ordered) // 3)
    start = ordered[alert_index]
    following = [
        stop for stop in ordered[alert_index + 1:]
        if stops[stop]["type"] == "Dropoff" and packages[stop]
    ][:4]
    if stops[start]["type"] != "Dropoff" or len(following) != 4:
        raise ValueError("the registered one-third route window is unavailable")
    source_stops = {"depot": ordered[0], "start": start,
                    **{f"site-{index}": stop for index, stop in enumerate(following, start=1)}}
    return source_stops, ordered, alert_index


def make_case(record: dict, profile: EVProfile, context: str, road_distance_m: dict | None = None) -> tuple[dict, dict, dict]:
    """Project one actual route into a bounded post-alert counterfactual."""
    if context not in BATTERY_CONTEXTS:
        raise ValueError(f"unknown modeled battery context: {context}")
    route = record["route"]
    stops = route["stops"]
    packages = record["packages"]
    travel = record["travel_seconds"]
    source_stops, ordered, alert_index = route_window(record)
    start = source_stops["start"]
    projected_prefix_seconds = sum(
        travel[first][second] + sum(
            float(package["planned_service_time_seconds"] or 0) for package in packages[second].values()
        )
        for first, second in zip(ordered[:alert_index], ordered[1:alert_index + 1])
    )
    departure_utc = _utc(f'{route["date_YYYY_MM_DD"]}T{route["departure_time_utc"]}')
    projected_alert_utc = departure_utc + timedelta(seconds=projected_prefix_seconds)
    closed = "site-1"
    selected = [
        (site, package_id, packages[stop][package_id])
        for site, stop in source_stops.items() if site.startswith("site-")
        for package_id in sorted(packages[stop])[:2]
    ]
    service_min = max(1, math.ceil(median(
        float(package["planned_service_time_seconds"] or 0) for _, _, package in selected
    ) / 60))
    travel_seconds = {
        a: {b: travel[first][second] for b, second in source_stops.items()}
        for a, first in source_stops.items()
    }
    travel_minutes = {
        a: {b: math.ceil(seconds / 60) for b, seconds in row.items()}
        for a, row in travel_seconds.items()
    }
    distance_field = "estimated_distance_proxy_km"
    if road_distance_m is None:
        coords = {local_id: (stops[source_id]["lat"], stops[source_id]["lng"])
                  for local_id, source_id in source_stops.items()}
        distance = {a: {b: road_proxy_km(first, second) for b, second in coords.items()}
                    for a, first in coords.items()}
    else:
        if set(road_distance_m) != source_stops.keys() or any(set(row) != source_stops.keys()
                                                            for row in road_distance_m.values()):
            raise ValueError("OSMnx distances must cover every pair of the six window sites")
        if any(not math.isfinite(value) or value < 0 or (a == b and value != 0)
               for a, row in road_distance_m.items() for b, value in row.items()):
            raise ValueError("Invalid OSMnx road distance")
        distance = {a: {b: meters / 1000 for b, meters in row.items()} for a, row in road_distance_m.items()}
        distance_field = "osmnx_road_distance_km"
    travel_wh = {
        a: {b: leg_wh(distance[a][b], seconds, profile) for b, seconds in row.items()}
        for a, row in travel_seconds.items()
    }
    historical_window = ("start", "site-1", "site-2", "site-3", "site-4", "depot")
    horizon = max(75, sum(travel_minutes[a][b] for a, b in zip(historical_window, historical_window[1:]))
                  + service_min * len(selected) + 35)
    reserve = profile.nominal_pack_wh // 10
    initial = profile.nominal_pack_wh * 65 // 100
    if context == "low_soc_at_alert":
        initial = reserve + travel_wh["start"]["depot"] + 300
    if not reserve + travel_wh["start"]["depot"] <= initial <= profile.nominal_pack_wh:
        raise ValueError("modeled EV profile does not support immediate reserve-safe return")
    charger = {
        "charger_rate_wh_per_min": 833,
        "charger_capacity_wh": min(profile.nominal_pack_wh, reserve + 14_000),
    }
    alternatives = sorted(
        set(source_stops) - {"depot", "start", closed},
        key=lambda site: (travel_seconds[closed][site], site),
    )[:2]
    source_packages = {}
    parcels = []
    for site, package_id, package in selected:
        end = package["time_window"]["end_time_utc"]
        latest = horizon if end is None else max(0, min(horizon, math.floor(
            (_utc(end) - projected_alert_utc).total_seconds() / 60
        )))
        parcels.append({
            "id": package_id, "vehicle_id": "ev-1", "original_locker": site,
            "alternatives": alternatives if site == closed else [], "deadline_min": latest,
        })
        source_packages[package_id] = {
            "source_stop_id": source_stops[site], "latest_utc_from_source": end,
            "earliest_utc_from_source": package["time_window"]["start_time_utc"],
            "planned_service_seconds_from_source": package["planned_service_time_seconds"],
        }
    demand = Counter(parcel["original_locker"] for parcel in parcels)
    case_id = f'{record["route_id"]}-{profile.id}-{context}'
    raw = {
        "schema": 1, "id": case_id, "data_basis": "synthetic",
        "outage": {"time_min": 0, "locker_id": closed}, "depot": "depot",
        "horizon_min": horizon, "service_min_per_parcel": service_min,
        "vehicles": [{"id": "ev-1", "node": "start", "battery_wh": initial, "reserve_wh": reserve}],
        "locker_free_capacity": {site: demand[site] + 2 for site in sorted(demand)},
        "parcels_in_original_plan_order": parcels,
        "travel_minutes": travel_minutes, "travel_wh": travel_wh,
    }
    metadata = {
        "route_id": record["route_id"], "station_code": route["station_code"],
        "observed_route_date": route["date_YYYY_MM_DD"], "observed_departure_utc": departure_utc.isoformat(),
        "projected_alert_utc": projected_alert_utc.isoformat(),
        "source_route_stop_count": len(stops), "source_alert_stop_rank": record["actual_sequence"]["actual"][start],
        "source_stops": source_stops, "source_packages": source_packages,
        "estimated_directed_travel_seconds_from_source": travel_seconds,
        distance_field: distance,
        "battery_context": context, "ev_profile": asdict(profile),
    }
    Scenario.from_dict(raw)
    return raw, charger, metadata


def _pickup_round_trip_km(metadata: dict, origin: str, destination: str) -> float:
    if "osmnx_road_distance_km" in metadata:
        distance = metadata["osmnx_road_distance_km"]
        return distance[origin][destination] + distance[destination][origin]
    return 2 * metadata["estimated_distance_proxy_km"][origin][destination]


def _preserve_observed(scenario: Scenario) -> Callable:
    original = {parcel.id: (index, parcel.original_locker) for index, parcel in enumerate(scenario.parcels)}

    def choose(decision) -> int:
        eligible = [
            (original[action.parcel_id][0], index)
            for index, action in enumerate(decision.actions)
            if action.kind == "deposit" and action.locker_id == original[action.parcel_id][1]
        ]
        if eligible:
            return min(eligible)[1]
        return next(i for i, action in enumerate(decision.actions) if action.kind == "return")

    return choose


def _rollout(scenario: Scenario, charger: dict, choose: Callable) -> tuple[list[int], dict]:
    environment = OutageEnv(scenario, **charger)
    actions = []
    while (decision := environment.observation()) is not None:
        action = choose(decision)
        actions.append(action)
        environment.step(action)
    return actions, environment.result()


def _offer_state(world: dict, protocol: dict) -> tuple[Scenario, dict, dict, list[dict], dict]:
    raw, charger, metadata = world["snapshot"], world["charger"], world["metadata"]
    original = Scenario.from_dict(raw)
    pickup_distance = {site: _pickup_round_trip_km(metadata, original.outage_locker, site)
                       for site in original.lockers}
    names = sorted(protocol["profiles"])
    profiles = {
        parcel.id: UtilityProfile(**protocol["profiles"][names[
            int.from_bytes(sha256(parcel.id.encode()).digest()[:4], "big") % len(names)
        ]])
        for parcel in original.parcels if parcel.original_locker == original.outage_locker
    }
    info = {
        **protocol, **charger,
        "locker_pickup": {
            site: {"pickup_min": round(pickup_distance[site] * 60 / 4.5),
                   "pickup_cost_cents": round(pickup_distance[site] * 50)}
            for site in original.lockers
        },
        "fallback": {
            **protocol["fallback"], "availability_after_alert_min": original.horizon_min + 15,
        },
    }
    route_id = metadata["route_id"]
    case = {"day": metadata["observed_route_date"],
            "seed": int.from_bytes(sha256(route_id.encode()).digest()[:4], "big"),
            "scenario": "amazon_route_window_locker_outage"}
    routed, dispatch, offers, fallback = negotiate(original, info, case, profiles, arm="adaptive")
    settings = {**charger, "preassigned_slots": offers["assignments"], "dispatch_after_min": dispatch}
    alert = {
        "kind": "locker_outage", "severity": "severe" if metadata["battery_context"] == BATTERY_CONTEXTS[1]
        else "moderate",
        "onboard_parcels": len(original.parcels), "accepted_reassignments": offers["metrics"]["reassignments"],
        "alert_min": 0, "dispatch_min": dispatch, "horizon_min": original.horizon_min,
    }
    return routed, settings, offers, fallback, alert


def _carbon(scenario: Scenario, result: dict, metadata: dict, profile: EVProfile) -> dict:
    # A dedicated common later home-delivery policy removes credits for leaving
    # parcels unserved. Its route follows the source order and starts/ends at the depot.
    parcel_site = {parcel.id: parcel.original_locker for parcel in scenario.parcels}
    unserved = set(result["unserved_parcel_ids"])
    fallback_stops = tuple(dict.fromkeys(
        parcel.original_locker for parcel in scenario.parcels if parcel.id in unserved
    ))
    future_route = (scenario.depot, *fallback_stops, scenario.depot)
    fallback_wh = sum(scenario.travel_energy[a, b] for a, b in zip(future_route, future_route[1:]))
    pickup_pairs = {
        (parcel_site[event["parcel_id"]], event["locker_id"]) for event in result["events"]
        if event["kind"] == "parcel_deposited" and event["locker_id"] != parcel_site[event["parcel_id"]]
    }
    if "osmnx_road_distance_km" in metadata:
        pickup_km = sum(_pickup_round_trip_km(metadata, a, b) for a, b in sorted(pickup_pairs))
    else:
        distance = metadata["estimated_distance_proxy_km"]
        pickup_km = 2 * sum(distance[a][b] for a, b in sorted(pickup_pairs))
    terminal = next(event["remaining_wh"] for event in result["events"] if event["kind"] == "vehicle_returned")
    metrics = result["metrics"]
    return carbon_receipt(
        traction_wh=metrics["travel_wh"], fallback_wh=fallback_wh,
        charged_wh=metrics["charged_wh"], initial_wh=scenario.vehicles[0].battery_wh,
        terminal_wh=terminal, all_car_pickup_km=pickup_km, profile=profile,
    )


def _cluster_delta(case_rows: list[dict], method: str, reference: str, field: str) -> dict:
    by_route = defaultdict(list)
    by_case = defaultdict(dict)
    for row in case_rows:
        by_case[row["case_id"]][row["method"]] = row
    for rows in by_case.values():
        intervention, comparator = rows[method], rows[reference]
        by_route[intervention["route_id"]].append(intervention[field] - comparator[field])
    effects = [fmean(values) for _, values in sorted(by_route.items())]
    rng = Random(20261125)
    sample = sorted(fmean(rng.choices(effects, k=len(effects))) for _ in range(2000))
    return {"mean_change": fmean(effects), "route_cluster_bootstrap_95": [sample[49], sample[1949]]}


def summarize(rows: list[dict], road_provenance: dict | None = None) -> dict:
    by_method = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        sorted_latency = sorted(row["post_offer_decision_ms"] for row in subset)
        by_method[method] = {
            "episodes": len(subset), "mean_by_deadline_service": fmean(row["service_rate"] for row in subset),
            "parcels_served_by_deadline": sum(row["served_parcels"] for row in subset),
            "parcels_recovered_in_deferred_home_model": sum(row["unserved_parcels"] for row in subset),
            "episodes_with_onboard_charge": sum(row["charged_wh"] > 0 for row in subset),
            "recovery_observed": sum(row["recovery_95_min"] is not None for row in subset),
            "mean_restricted_recovery_minutes": fmean(row["restricted_recovery_min"] for row in subset),
            "post_offer_decision_p95_ms": sorted_latency[math.ceil(0.95 * len(subset)) - 1],
            "decision_statuses": dict(Counter(row["selection_status"] for row in subset)),
            "selected_interventions": dict(Counter(row["selected_intervention"] for row in subset)),
            "modeled_offers_issued": sum(row["offers_issued"] for row in subset),
            "modeled_offers_accepted": sum(row["offers_accepted"] for row in subset),
            "modeled_credit_spend_cents": sum(row["credit_spent_cents"] for row in subset),
            "mean_carrier_operational_kg_co2e": fmean(row["carrier_operational_kg_co2e"] for row in subset),
            "mean_operational_kg_co2e_at_assumed_share": fmean(
                row["operational_kg_co2e_at_assumed_share"] for row in subset
            ),
            "mean_operational_kg_co2e_no_car": fmean(row["operational_kg_co2e_no_car"] for row in subset),
            "mean_operational_kg_co2e_all_car": fmean(row["operational_kg_co2e_all_car"] for row in subset),
            "solver_statuses": dict(Counter(row["solver_status"] for row in subset if row["solver_status"])),
            "solver_fallbacks": sum(row["solver_fallback"] for row in subset),
            "column_cap_cases": sum(row["column_cap_hit"] for row in subset),
        }
        references = [reference for reference in ("preserve_observed_order", "frozen_mappo") if reference != method]
        if method == "agentic_supervisor":
            references.append("deterministic_supervisor")
        for reference in references:
            by_method[method][f"paired_vs_{reference}"] = {
                field: _cluster_delta(rows, method, reference, field)
                for field in ("service_rate", "operational_kg_co2e_at_assumed_share")
            }
    report = {
        "routes": len({row["route_id"] for row in rows}),
        "matched_cases": len({row["case_id"] for row in rows}), "episode_rows": len(rows),
        "assumptions": assumptions(),
        "evidence": {
            "observed": "official route IDs/dates, stop types and anonymized coordinates, executed stop order, package records",
            "supplied_estimate": "Amazon directed point-to-point travel seconds and planned service seconds",
            "modeled": (
                "four-stop window with first two package IDs at each stop, projected alert clock, latest-date/horizon "
                "deadlines, designated candidate lockers/outage and capacity, EV profiles, grid loss and pickup mode, "
                "common adaptive simulated offers and subsequent home delivery. "
                "No earliest-window enforcement or observed EV/customer outcomes."
            ),
        },
        "latency": "common offer time excluded from reported post-offer compute; 2-second supervisor acceptance window",
        "by_method": by_method,
    }
    if road_provenance is not None:
        report["assumptions"]["movement_model"] = (
            "Wh per km of directed OSMnx road distance plus moving auxiliary Wh from Amazon supplied duration"
        )
        report["evidence"]["road_distance"] = road_provenance
    return report


def _write(stream, row: dict) -> None:
    stream.write(json.dumps(row, allow_nan=False) + "\n")


def _row(
    world: dict, method: str, result: dict, solver: dict | None, profile: EVProfile,
    choice: dict, offers: dict, offer_ms: float,
) -> dict:
    raw, metadata = world["snapshot"], world["metadata"]
    recovery = result["metrics"]["recovery_95_min"]
    return {
        "case_id": raw["id"], "route_id": metadata["route_id"], "method": method,
        "profile": profile.id, "context": metadata["battery_context"],
        "restricted_recovery_min": raw["horizon_min"] if recovery is None else recovery,
        "selection_status": choice["status"], "selected_intervention": choice["decision_id"],
        "post_offer_decision_ms": choice["end_to_end_decision_ms"], "offer_ms": offer_ms,
        "offers_issued": offers["metrics"]["issued"], "offers_accepted": offers["metrics"]["accepted"],
        "credit_spent_cents": offers["metrics"]["spent_cents"],
        **result["metrics"], **_carbon(Scenario.from_dict(raw), result, metadata, profile),
        "solver_status": None if solver is None else solver["solver_status"],
        "solver_fallback": solver is not None and solver["fallback_reason"] is not None,
        "column_cap_hit": solver is not None and bool(solver["column_capped_vehicles"]),
    }


def _load_route_distances(data: Path, path: Path | None) -> dict | None:
    if path is None:
        return None
    road = json.loads(path.read_bytes())
    if (road["schema"] != 1 or road["source"] != "osmnx" or
            road["source_records_sha256"] != file_sha256(data / "records.jsonl")):
        raise ValueError("OSMnx road distances do not match the official route input")
    return road


def _worlds(data: Path, road: dict | None = None):
    records = read_records(data)
    if road is not None and set(road["by_route"]) != {record["route_id"] for record in records}:
        raise ValueError("Missing or unexpected route in the OSMnx distance table")
    for record in records:
        distance = None
        if road is not None:
            table = road["by_route"][record["route_id"]]
            if table["source_stops"] != route_window(record)[0]:
                raise ValueError("OSMnx distance table uses different source stops")
            distance = table["distance_m"]
        for profile in PROFILES:
            for context in BATTERY_CONTEXTS:
                raw, charger, meta = make_case(record, profile, context, distance)
                yield {"snapshot": raw, "charger": charger, "metadata": meta}, profile


def _evaluate(world: dict, profile: EVProfile, offer_protocol: dict, policy, model) -> tuple[dict, list[dict]]:
    started = perf_counter()
    scenario, settings, offers, fallback, alert = _offer_state(world, offer_protocol)
    offer_ms = 1000 * (perf_counter() - started)
    started = perf_counter()
    indices, result = _rollout(scenario, settings, _preserve_observed(scenario))
    preserved = {"action_indices": indices, "result": result}
    preserve_ms = 1000 * (perf_counter() - started)
    started = perf_counter()
    indices, result = _rollout(scenario, settings, policy.choose)
    baseline = {"action_indices": indices, "result": result}
    solo_ms = 1000 * (perf_counter() - started)
    program = repair(scenario, **settings, **CP_LIMITS)
    indices, result = _rollout(scenario, settings, program.choose)
    if result["metrics"] != program.receipt["certified_metrics"]:
        raise ValueError("empirical repair rollout differs from its certificate")
    candidates = {
        "continue_mappo": baseline, "route_repair": {"action_indices": indices, "result": result},
    }
    menu = _menu(scenario, candidates)
    planning_ms = 1000 * (perf_counter() - started)
    budget = DECISION_CONFIG["supervisor_budget_s"]
    agentic = _take_decision(alert, menu, budget, planning_ms, model=model)
    deterministic = _take_decision(alert, menu, budget, planning_ms)
    trace = {
        "offers": offers, "fallbacks": fallback, "alert": alert, "preserve": preserved,
        "preserve_ms": preserve_ms, "repair_receipt": program.receipt, "candidates": candidates,
        "menu": [asdict(candidate) for candidate in menu],
        "timing": {"offer_ms": offer_ms, "solo_ms": solo_ms, "planning_ms": planning_ms},
        "mappo_alone": {"status": "selected", "decision_id": "continue_mappo", "end_to_end_decision_ms": solo_ms},
        "selections": {"agentic_supervisor": agentic, "deterministic_supervisor": deterministic},
        "preselection_planning_ms": planning_ms, "model_scores": model.accepted_scores(agentic),
        "worker_status": "not_called" if agentic["status"] == "planning_timeout" else model.last_status,
    }
    _verify_selections(trace, DECISION_CONFIG)
    return trace, _rows(world, profile, trace)


def _rows(world: dict, profile: EVProfile, trace: dict) -> list[dict]:
    choices = {
        "preserve_observed_order": {
            "status": "selected", "decision_id": "preserve_observed_order",
            "end_to_end_decision_ms": trace["preserve_ms"],
        },
        "frozen_mappo": trace["mappo_alone"], **trace["selections"],
    }
    rows = []
    for method in METHODS:
        choice = choices[method]
        selected = choice["decision_id"]
        record = trace["preserve"] if selected == "preserve_observed_order" else trace["candidates"][selected]
        solver = trace["repair_receipt"] if selected == "route_repair" else None
        rows.append(_row(world, method, record["result"], solver, profile, choice,
                         trace["offers"], trace["timing"]["offer_ms"]))
    return rows


def run(data: Path, checkpoint: Path, output: Path, model_path: Path = DEFAULT_MODEL,
        road_distances: Path | None = None) -> dict:
    from .m3_rllib import RLlibMAPPO
    import torch

    torch.set_num_threads(1)
    policy = RLlibMAPPO.load(checkpoint)
    protocol = json.loads(OFFER_PROTOCOL.read_bytes())
    road = _load_route_distances(data, road_distances)
    artifacts = (*ARTIFACTS, "osmnx_distances.json") if road is not None else ARTIFACTS
    model = DeadlineModel(partial(OrderInvariantMenuModel, model_path, threads=2), reserve_s=0.1)
    rows = []
    try:
        model.warmup(timeout_s=90)
        _graph()
        worlds = _worlds(data, road)
        first, _ = next(worlds)
        scenario, settings, _, _, alert = _offer_state(first, protocol)
        _, baseline = _rollout(scenario, settings, policy.choose)
        menu = _menu(scenario, {identifier: {"result": baseline} for identifier in ("continue_mappo", "route_repair")})
        model(DecisionView(_freeze(alert), menu), timeout_s=20)
        output.mkdir(parents=True, exist_ok=False)
        (output / "policy.pt").write_bytes(checkpoint.read_bytes())
        (output / "offer_protocol.json").write_bytes(OFFER_PROTOCOL.read_bytes())
        if road is not None:
            (output / "osmnx_distances.json").write_bytes(road_distances.read_bytes())
        with (output / "worlds.jsonl").open("x", encoding="utf-8", newline="\n") as cases, (
            output / "observations.jsonl"
        ).open("x", encoding="utf-8", newline="\n") as traces:
            for index, (world, profile) in enumerate(_worlds(data, road), 1):
                trace, matched = _evaluate(world, profile, protocol, policy, model)
                rows.extend(matched)
                _write(cases, world)
                _write(traces, {**trace, "rows": matched})
                if index % 20 == 0:
                    cases.flush()
                    traces.flush()
                    print(f"{index} empirical matched cases complete", flush=True)
    finally:
        model.close()
    summary = summarize(rows, road["provenance"] if road is not None else None)
    with (output / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({
            "schema": 2 if road is not None else 1, "source_manifest_sha256": file_sha256(data / "manifest.json"),
            "source_records_sha256": file_sha256(data / "records.jsonl"),
            "model": json.loads((model_path / "source.json").read_bytes()),
            "sha256": {name: file_sha256(output / name) for name in artifacts},
            "source_sha256": {
                name: file_sha256(Path(__file__).with_name(name)) for name in (
                    "m6_empirical_benchmark.py", "empirical_amazon.py", "empirical_energy.py",
                    "m2_simulator.py", "m3_route_repair.py", "m3_rllib.py", "m5_local_model.py",
                    "m5_deadline.py", "m6_incentive.py",
                )
            },
        }, stream, indent=2)
    return summary


def verify(data: Path, output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest["schema"] not in (1, 2):
        raise ValueError("unknown empirical experiment schema")
    road_path = output / "osmnx_distances.json" if manifest["schema"] == 2 else None
    artifacts = (*ARTIFACTS, road_path.name) if road_path is not None else ARTIFACTS
    if (
        manifest["source_manifest_sha256"] != file_sha256(data / "manifest.json")
        or manifest["source_records_sha256"] != file_sha256(data / "records.jsonl")
        or any(file_sha256(output / name) != manifest["sha256"][name] for name in artifacts)
    ):
        raise ValueError("empirical run fingerprints do not match")
    protocol = json.loads((output / "offer_protocol.json").read_bytes())
    road = _load_route_distances(data, road_path)
    rows = []
    with (output / "worlds.jsonl").open(encoding="utf-8") as cases, (
        output / "observations.jsonl"
    ).open(encoding="utf-8") as traces:
        for (world, profile), line in zip(_worlds(data, road), cases, strict=True):
            if world != json.loads(line):
                raise ValueError("saved case does not regenerate from the official records")
            trace = json.loads(next(traces))
            scenario, settings, offers, fallback, alert = _offer_state(world, protocol)
            if (trace["offers"], trace["fallbacks"], trace["alert"]) != (offers, fallback, alert):
                raise ValueError("empirical customer commitments do not regenerate")
            for record in (trace["preserve"], *trace["candidates"].values()):
                indices = iter(record["action_indices"])
                _, result = _rollout(scenario, settings, lambda _: next(indices))
                if next(indices, None) is not None or result != record["result"]:
                    raise ValueError("saved empirical decisions do not replay")
            if (
                trace["menu"] != [asdict(candidate) for candidate in _menu(scenario, trace["candidates"])]
                or trace["repair_receipt"]["certified_metrics"] != trace["candidates"]["route_repair"]["result"]["metrics"]
                or not math.isfinite(trace["preserve_ms"]) or trace["preserve_ms"] < 0
            ):
                raise ValueError("empirical candidate integrity mismatch")
            _verify_selections(trace, DECISION_CONFIG)
            recomputed = _rows(world, profile, trace)
            if recomputed != trace["rows"]:
                raise ValueError("saved empirical metrics differ from the replay")
            rows.extend(recomputed)
        if next(traces, None) is not None:
            raise ValueError("unexpected additional traces")
    summary = summarize(rows, road["provenance"] if road is not None else None)
    if summary != json.loads((output / "summary.json").read_bytes()):
        raise ValueError("saved aggregate differs from the certified episode outcomes")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    execute = subs.add_parser("run")
    execute.add_argument("data", type=Path)
    execute.add_argument("checkpoint", type=Path)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    execute.add_argument("--osmnx-distances", type=Path, help="pinned OSMnx distance tables for the exact official route windows")
    check = subs.add_parser("verify")
    check.add_argument("data", type=Path)
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            summary = run(args.data, args.checkpoint, args.output, args.model, args.osmnx_distances)
        else:
            summary = verify(args.data, args.output)
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(args.output.resolve(), summary["routes"], summary["matched_cases"], summary["episode_rows"])


if __name__ == "__main__":
    main()
