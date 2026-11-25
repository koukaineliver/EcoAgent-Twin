"""Replay issued GNN and trailing-mean staging quotas through certified routes."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .model import Scenario, Stop, evaluate_plan


METHODS = ("gnn", "trailing_mean")
DEPOT = "depot"
PACKAGES_PER_BATCH = 100
POSITIONS_AT_EACH_SITE = 4
VEHICLE_COUNT = 6
BATTERY_WH = 65_000
RESERVE_WH = 8_000
WH_PER_KM = 220
DEADLINE_MIN = 180
HORIZON_MIN = 240
SERVICE_MIN = 3
SOURCE_FILES = {
    "demand": "la_locker_demand_daily.csv",
    "gnn_roads": "la_locker_road_time_matrix_api.csv",
    "depot_roads": "la_depot_locker_road_matrix_api.csv",
    "locations": "la_phase5_top30_candidate_lockers.csv",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def read_roads(path: Path, nodes: tuple[str, ...], *, source: str = "osrm_table_api") -> dict[tuple[str, str], tuple[Decimal, Decimal]]:
    roads = {}
    for row in csv_rows(path):
        arc = row["from_locker_id"], row["to_locker_id"]
        if arc in roads or row["matrix_source"] != source or row["true_road_matrix"] != "True":
            raise ValueError(f"duplicate arc or unexpected road provenance in {path.name}: {arc}")
        minutes = Decimal(row["travel_time_min"])
        kilometers = Decimal(row["distance_km"])
        if not minutes.is_finite() or not kilometers.is_finite() or min(minutes, kilometers) < 0:
            raise ValueError(f"invalid road length or duration in {path.name}: {arc}")
        if arc[0] == arc[1] and (minutes or kilometers):
            raise ValueError(f"nonzero diagonal in {path.name}: {arc}")
        roads[arc] = minutes, kilometers
    if roads.keys() != {(start, end) for start in nodes for end in nodes}:
        raise ValueError(f"{path.name} does not cover the specified directed node pairs")
    return roads


def read_inputs(gnn_dir: Path, processed: Path) -> tuple[dict, list[dict], dict[str, Path]]:
    paths = {name: processed / filename for name, filename in SOURCE_FILES.items()}
    paths.update(gnn_summary=gnn_dir / "summary.json", forecasts=gnn_dir / "forecasts.json")
    summary = json.loads(paths["gnn_summary"].read_bytes())
    forecasts = json.loads(paths["forecasts"].read_bytes())
    if summary["forecast_sha256"] != file_digest(paths["forecasts"]):
        raise ValueError("forecast bundle fails the GNN run's SHA-256 check")
    prior_hashes = {
        path.replace("\\", "/").rsplit("/", 1)[-1]: expected
        for path, expected in summary["inputs_sha256"].items()
    }
    for name in ("demand", "gnn_roads"):
        if file_digest(paths[name]) != prior_hashes[paths[name].name]:
            raise ValueError(f"{paths[name].name} differs from the GNN run's verified input")
    return summary, forecasts, paths


def check_holdout(summary: dict, forecasts: list[dict], nodes: tuple[str, ...], demand: Path) -> None:
    split = summary["dates"]
    first_date = date.fromisoformat(split["test_from"])
    expected_days = [first_date + timedelta(days=offset) for offset in range(split["holdout_days"])]
    if not expected_days or expected_days[-1].isoformat() != split["test_through"]:
        raise ValueError("GNN holdout summary does not define a complete dated period")
    if [row["target_date"] for row in forecasts] != [day.isoformat() for day in expected_days]:
        raise ValueError("forecast records do not match the GNN holdout period")
    observed = {}
    target_days = {row["target_date"] for row in forecasts}
    for row in csv_rows(demand):
        if row["date"] in target_days and row["locker_id"] in nodes:
            key = row["date"], row["locker_id"]
            if key in observed:
                raise ValueError(f"duplicate archived locker-day record: {key}")
            observed[key] = int(row["packages_assigned"])
    for record, target in zip(forecasts, expected_days, strict=True):
        if record["issue_date"] != (target - timedelta(days=1)).isoformat():
            raise ValueError(f"expected prior-day forecast issue for {target}")
        if record["observed_by_locker"] != {
            node: observed.get((target.isoformat(), node), 0) for node in nodes
        }:
            raise ValueError(f"archived observations differ from GNN record for {target}")
        if any(type(value) is not int or value < 0 for value in record["observed_by_locker"].values()):
            raise ValueError(f"invalid package count for {target}")
        for method in METHODS:
            weights = record["staged_slots_by_locker"][method]
            if (
                weights.keys() != set(nodes)
                or any(type(value) is not int or value < 0 for value in weights.values())
                or sum(weights.values()) != summary["assumed_staging_slots_per_day"]
            ):
                raise ValueError(f"invalid issued quota for {target} {method}")


def package_batches(observed: dict[str, int]) -> dict[str, int]:
    """Preserve the day's rounded workload and apportion fractional batches by site."""
    batches = {node: packages // PACKAGES_PER_BATCH for node, packages in observed.items()}
    total = (sum(observed.values()) + PACKAGES_PER_BATCH // 2) // PACKAGES_PER_BATCH
    ordered = sorted(observed, key=lambda node: (-(observed[node] % PACKAGES_PER_BATCH), node))
    for node in ordered[:total - sum(batches.values())]:
        batches[node] += 1
    return batches


def stage_positions(
    weights: dict[str, int], free: dict[str, int], budget: int,
) -> dict[str, int]:
    """Bound the integer apportionment by predeclared open positions at each site."""
    if budget <= 0 or budget > sum(free.values()):
        raise ValueError("staging budget must fit the open position supply")
    scores = {node: weights[node] if capacity else 0 for node, capacity in free.items()}
    if not sum(scores.values()):
        scores = {node: int(bool(capacity)) for node, capacity in free.items()}
    total = sum(scores.values())
    allocations = {node: min(free[node], budget * scores[node] // total) for node in free}
    for _ in range(budget - sum(allocations.values())):
        candidate = min(
            (node for node in free if allocations[node] < free[node]),
            key=lambda node: (total * allocations[node] - budget * scores[node], node),
        )
        allocations[candidate] += 1
    return allocations


def route(raw: dict) -> dict:
    scenario = Scenario.from_dict(raw)
    environment = OutageEnv(scenario, charger_rate_wh_per_min=0, charger_capacity_wh=0)
    controller = RuleBased(scenario)
    actions = []
    while (decision := environment.observation()) is not None:
        index = controller.choose(decision)
        action = decision.actions[index]
        actions.append({
            "vehicle_id": decision.vehicle_id,
            "time_min": decision.time_min,
            "action_index": index,
            "kind": action.kind,
            "parcel_id": action.parcel_id,
            "locker_id": action.locker_id,
            "travel_min": action.travel_min,
            "completion_min": action.completion_min,
            "energy_wh": action.energy_wh,
        })
        environment.step(index)
    result = environment.result()
    plan = {
        vehicle.id: tuple(
            Stop(action["parcel_id"], action["locker_id"])
            for action in actions if action["vehicle_id"] == vehicle.id and action["kind"] == "deposit"
        )
        for vehicle in scenario.vehicles
    }
    certification = evaluate_plan(scenario, plan)
    metrics = result["metrics"]
    if (
        sum(len(outcome.stops) for outcome in certification.values()) != metrics["served_parcels"]
        or sum(outcome.travel_wh for outcome in certification.values()) != metrics["travel_wh"]
        or sum(outcome.travel_min for outcome in certification.values()) != metrics["travel_min"]
    ):
        raise RuntimeError("independent route certificate disagrees with the simulator")
    return {"actions": actions, "result": result}


def evaluate(gnn_dir: Path, processed: Path) -> tuple[dict, list[dict]]:
    summary, forecasts, paths = read_inputs(gnn_dir, processed)
    sites = {}
    for row in csv_rows(paths["locations"]):
        node = row["locker_id"]
        if node in sites:
            raise ValueError(f"duplicate candidate coordinates: {node}")
        sites[node] = (Decimal(row["lng"]), Decimal(row["lat"]))
    nodes = tuple(sorted(sites))
    if len(nodes) != 30 or len(nodes) != summary["locker_count"] or DEPOT in sites:
        raise ValueError("expected exactly the GNN's 30 archived candidate lockers and a distinct depot")
    check_holdout(summary, forecasts, nodes, paths["demand"])
    source = "osmnx" if summary["scope"].endswith("OSMnx geometry") else "osrm_table_api"
    engine_name = "OSMnx" if source == "osmnx" else "OSRM"
    gnn_roads = read_roads(paths["gnn_roads"], nodes, source=source)
    all_nodes = (DEPOT, *nodes)
    roads = read_roads(paths["depot_roads"], all_nodes, source=source)
    if any(roads[arc][0] != minutes for arc, (minutes, _) in gnn_roads.items()):
        raise ValueError("the depot road archive disagrees with GNN directed locker travel times")

    minutes = {
        start: {
            end: int(roads[start, end][0].to_integral_value(rounding=ROUND_CEILING))
            for end in all_nodes
        }
        for start in all_nodes
    }
    energy = {
        start: {
            end: int((roads[start, end][1] * WH_PER_KM).to_integral_value(rounding=ROUND_CEILING))
            for end in all_nodes
        }
        for start in all_nodes
    }
    closed = nodes[0]
    occupied = {node: 1 + index % 2 for index, node in enumerate(nodes)}
    free = {node: POSITIONS_AT_EACH_SITE - occupied[node] if node != closed else 0 for node in nodes}
    if summary["assumed_staging_slots_per_day"] % PACKAGES_PER_BATCH:
        raise ValueError("archived slot budget must convert to a whole number of batch positions")
    budget = summary["assumed_staging_slots_per_day"] // PACKAGES_PER_BATCH
    vehicle_ids = tuple(f"ev-{index + 1:02d}" for index in range(VEHICLE_COUNT))
    geographic_order = sorted(nodes, key=lambda node: (*sites[node], node))
    owner = {
        node: vehicle_ids[index * VEHICLE_COUNT // len(nodes)]
        for index, node in enumerate(geographic_order)
    }
    common = {
        "schema": 1,
        "data_basis": "synthetic",
        "depot": DEPOT,
        "outage": {"time_min": 0, "locker_id": closed},
        "horizon_min": HORIZON_MIN,
        "service_min_per_parcel": SERVICE_MIN,
        "vehicles": [
            {"id": vehicle, "node": DEPOT, "battery_wh": BATTERY_WH, "reserve_wh": RESERVE_WH}
            for vehicle in vehicle_ids
        ],
        "travel_minutes": minutes,
        "travel_wh": energy,
    }
    traces = []
    for record in forecasts:
        issued_positions = {
            method: stage_positions(record["staged_slots_by_locker"][method], free, budget)
            for method in METHODS
        }
        realized_batches = package_batches(record["observed_by_locker"])
        matched = {
            **common,
            "id": f"m6-predictive-{record['target_date']}",
            "parcels_in_original_plan_order": [
                {
                    "id": f"{node}-{batch + 1:02d}",
                    "vehicle_id": owner[node],
                    "original_locker": node,
                    "alternatives": [],
                    "deadline_min": DEADLINE_MIN,
                }
                for node in nodes for batch in range(realized_batches[node])
            ],
        }
        traces.append({
            "target_date": record["target_date"],
            "issue_date": record["issue_date"],
            "matched_scenario_sha256": digest(json_bytes(matched)),
            "batches_by_locker": realized_batches,
            "arms": {
                method: {
                    "staged_positions_by_locker": issued_positions[method],
                    **route({**matched, "locker_free_capacity": issued_positions[method]}),
                }
                for method in METHODS
            },
        })
    return {
        "scope": "Retrospective batch-routing sensitivity on the GNN's fixed 30-candidate holdout",
        "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "inputs_sha256": {name: file_digest(path) for name, path in paths.items()},
        "protocol": {
            "holdout_days": len(traces),
            "test_from": traces[0]["target_date"],
            "test_through": traces[-1]["target_date"],
            "closed_locker": closed,
            "occupancy_observation": "Declared, identical and fixed across both arms and all dates",
            "declared_occupied_positions": occupied,
            "open_free_positions_before_staging": free,
            "physical_positions_per_locker": POSITIONS_AT_EACH_SITE,
            "staged_positions_total": budget,
            "packages_per_symbolic_batch": PACKAGES_PER_BATCH,
            "batch_rounding": "Daily total nearest whole batch, half up; largest site remainders",
            "allocation": "Prior-day issued quotas, renormalized over open lockers and capped by declared free capacity",
            "vehicle_snapshots": common["vehicles"],
            "locker_custody_vehicle": owner,
            "deadline_min": DEADLINE_MIN,
            "horizon_min": HORIZON_MIN,
            "service_min": SERVICE_MIN,
            "charging": "disabled in both arms",
            "reassignment": "disabled in both arms",
            "travel_time": f"Same archived directed {engine_name} minutes for both arms, rounded upward",
            "travel_energy": f"Same archived directed {engine_name} distance for both arms, multiplied by {WH_PER_KM} Wh/km and rounded upward",
            "controller": "EcoAgent-Twin M2 deterministic deadline then shortest feasible action",
            "validation": "OutageEnv sequential feasibility and an independent evaluate_plan route certificate",
        },
        "limits": [
            "Archived target-day totals instantiate the same realized batch manifest at dispatch after quotas are fixed.",
            "No live locker occupancy, actual fleet manifests, payload, or empirical Wh calibration enters the scenarios.",
            "Candidate sites were already fixed retrospectively. Quota, service and energy results are simulated.",
            "Trajectories can differ as a consequence of staging quotas. The two arms use identical directed roads and conditions.",
        ],
        **aggregate(traces),
    }, traces


def aggregate(traces: list[dict]) -> dict:
    metric_names = ("total_parcels", "served_parcels", "unserved_parcels", "travel_min", "travel_wh")
    totals = {}
    for method in METHODS:
        rows = [trace["arms"][method]["result"]["metrics"] for trace in traces]
        counts = {name: sum(row[name] for row in rows) for name in metric_names}
        totals[method] = {
            "batches": counts["total_parcels"],
            "served_batches": counts["served_parcels"],
            "unserved_batches": counts["unserved_parcels"],
            "pooled_service_rate": counts["served_parcels"] / counts["total_parcels"],
            "travel_min": counts["travel_min"],
            "travel_wh": counts["travel_wh"],
        }
    paired = {
        "days_with_different_staged_positions": 0,
        "days_with_different_routed_actions": 0,
        "days_gnn_served_more": 0,
        "days_trailing_mean_served_more": 0,
        "days_tied_on_service": 0,
    }
    for trace in traces:
        gnn, trailing = (trace["arms"][method] for method in METHODS)
        paired["days_with_different_staged_positions"] += (
            gnn["staged_positions_by_locker"] != trailing["staged_positions_by_locker"]
        )
        fields = ("vehicle_id", "kind", "parcel_id", "locker_id", "completion_min")
        signatures = [
            [tuple(action[field] for field in fields) for action in arm["actions"]]
            for arm in (gnn, trailing)
        ]
        paired["days_with_different_routed_actions"] += signatures[0] != signatures[1]
        first, second = (arm["result"]["metrics"]["served_parcels"] for arm in (gnn, trailing))
        paired["days_gnn_served_more"] += first > second
        paired["days_trailing_mean_served_more"] += first < second
        paired["days_tied_on_service"] += first == second
    return {"totals": totals, "paired": paired}


def trace_bytes(traces: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in traces).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "verify"))
    parser.add_argument("--gnn", type=Path, required=True, help="verified GNN run directory")
    parser.add_argument("--processed", type=Path, required=True, help="archived LA processed CSV directory")
    parser.add_argument("--output", type=Path, help="new run directory, or existing directory when verifying")
    args = parser.parse_args()
    if args.operation == "verify" and args.output is None:
        parser.error("verify requires --output pointing to the saved run")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("runs") / f"m6_predictive_routing_{stamp}"
    summary, traces = evaluate(args.gnn, args.processed)
    serialized = trace_bytes(traces)
    summary["traces_sha256"] = digest(serialized)
    if args.operation == "run":
        output.mkdir(parents=True, exist_ok=False)
        with (output / "traces.jsonl").open("xb") as trace_file:
            trace_file.write(serialized)
        with (output / "summary.json").open("xb") as summary_file:
            summary_file.write(json_bytes(summary))
    else:
        saved_summary = json.loads((output / "summary.json").read_bytes())
        saved_summary.pop("input_paths")
        summary.pop("input_paths")
        if saved_summary != summary or (output / "traces.jsonl").read_bytes() != serialized:
            raise SystemExit("Saved results, input hashes or route traces differ from replay")
    print(json.dumps({
        "directory": str(output.resolve()),
        "operation": args.operation,
        "episodes": len(traces) * len(METHODS),
        **aggregate(traces),
    }, indent=2))


if __name__ == "__main__":
    main()
