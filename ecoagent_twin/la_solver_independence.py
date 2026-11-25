"""Compare independent route repair engines on the two-EV LA matched cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .la_controller_experiment import _input_hash
from .m3_cases import digest, read_json, write_json
from .m3_experiment import _certify_record, _restricted
from .m5_experiment import _state
from .m6_solver import DEFAULT_TRANSITION_LIMIT, certified_repair, cp_sat_backend, simulator_backend
from .m6_solver_experiment import METHODS, _summary


RUN = Path("runs/la_osmnx_controller_distinct_20261125")
ROADS = Path("runs/osmnx_la_university_park/road_network_scale.json")
OUTPUT = Path("runs/la_osmnx_solver_independence_20261125.json")
KEYS = ("case_id", "traffic", "traffic_seed", "chaos_kind", "severity", "input_sha256")


def _source(run: Path, roads: Path) -> tuple[dict, dict, dict, list[dict]]:
    receipt = read_json(run / "manifest.json")
    if receipt["kind"] != "la_osmnx_frozen_controller_transfer" or receipt["road_network_sha256"] != digest(roads):
        raise ValueError("Expected the verified matched LA controller run and its OSMnx inputs")
    for name in ("protocol.json", "offer_protocol.json", "traces.jsonl"):
        if digest(run / name) != receipt["sha256"][name]:
            raise ValueError(f"LA input fingerprint differs: {name}")
    config = read_json(run / "protocol.json")
    if config["geometry_layout"] != "distinct":
        raise ValueError("Expected the physically distinct LA transfer campaign")
    with (run / "traces.jsonl").open(encoding="utf-8") as stream:
        traces = [row for line in stream if (row := json.loads(line))["world"]["fleet_size"] == 2]
    declared = Counter((day, profile, disruption, severity) for day in config["traffic_seeds"]
                       for profile in ("light_traffic", "heavy_traffic") for disruption in config["chaos_kinds"]
                       for severity in config["case_seeds"])
    observed = Counter((row["traffic_seed"], row["traffic"], row["chaos_kind"], row["severity"]) for row in traces)
    if observed != declared or len({row["case_id"] for row in traces}) != len(traces):
        raise ValueError("Two-EV cases do not form the expected complete matched design")
    provenance = {"run_manifest_sha256": digest(run / "manifest.json"),
                  "road_network_sha256": receipt["road_network_sha256"],
                  "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in (
                      "la_solver_independence.py", "m6_solver.py", "m3_route_repair.py", "m2_simulator.py",
                  )}}
    return provenance, config, read_json(run / "offer_protocol.json"), traces


def _state_for(trace: dict, protocol: dict):
    world = trace["world"]
    scenario, dispatch, offers, fallbacks, alert = _state(world, protocol)
    if ((dispatch, offers, fallbacks, alert) != tuple(trace[key] for key in ("dispatch", "offers", "fallbacks", "alert"))
            or _input_hash(world, dispatch, offers) != trace["input_sha256"]):
        raise ValueError("Rebuilt Mesa offers or dispatch differ from the recorded common state")
    _certify_record(scenario, dispatch, trace["candidates"]["route_repair"])
    return scenario, dispatch, offers["metrics"]


def _order(index: int) -> tuple:
    return METHODS if index % 2 == 0 else METHODS[::-1]


def run(source: Path, roads: Path, output: Path, transition_limit: int) -> dict:
    provenance, config, protocol, traces = _source(source, roads)
    cp, search = (cp_sat_backend(**config["repair"]), simulator_backend(transition_limit))
    planners = dict(zip(METHODS, (cp, search), strict=True))
    rows = []
    for i, trace in enumerate(traces):
        scenario, dispatch, metrics = _state_for(trace, protocol)
        arms = {}
        for method in _order(i):
            arm = certified_repair(scenario, dispatch, planners[method])
            arm["restricted_recovery_min"] = _restricted(scenario, arm["record"]["result"]["metrics"])
            arms[method] = arm
        rows.append({**{key: trace[key] for key in KEYS}, "dispatch": dispatch, "offer_metrics": metrics,
                     "execution_order": list(_order(i)), "backends": arms,
                     "cp_sat_record_matches_archived": arms["cp_sat"]["record"] == trace["candidates"]["route_repair"]})
    summary = _summary(rows)
    write_json(output, {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "provenance": provenance, "repair_limits": config["repair"],
                        "search_transition_limit": transition_limit, "summary": summary, "rows": rows,
                        "scope": "Two-EV synthetic disruption scenarios with modeled costs on the real University Park OSMnx graph. Both repairs share Mesa offers, fixed custody, and certified event replay.",
                        "objective": "Lexicographic maximum on-time deposits, minimum travel minutes, minimum traction watt-hours. Independently apply the bounded CP-SAT and Python action-search backends.",
                        "timing": "Single runs on this host in alternating backend order. Planning follows shared offers and includes each backend's own certification. The common additional replay is timed separately."})
    return summary


def verify(source: Path, roads: Path, output: Path) -> dict:
    provenance, config, protocol, traces = _source(source, roads)
    archive = read_json(output)
    if archive["schema"] != 1 or archive["provenance"] != provenance or archive["repair_limits"] != config["repair"]:
        raise ValueError("Solver audit inputs or code differ from their receipt")
    search = simulator_backend(archive["search_transition_limit"])
    for i, (trace, row) in enumerate(zip(traces, archive["rows"], strict=True)):
        scenario, dispatch, metrics = _state_for(trace, protocol)
        if (any(row[key] != trace[key] for key in KEYS) or row["dispatch"] != dispatch or row["offer_metrics"] != metrics
                or row["execution_order"] != list(_order(i)) or set(row["backends"]) != set(METHODS)):
            raise ValueError("A paired case or dispatch differs from the archive")
        for arm in row["backends"].values():
            record = arm["record"]
            _certify_record(scenario, dispatch, record)
            actual = record["result"]
            if (arm["planner"]["certified_metrics"] != actual["metrics"]
                    or arm["planner"]["certified_unserved_parcel_ids"] != actual["unserved_parcel_ids"]
                    or arm["restricted_recovery_min"] != _restricted(scenario, actual["metrics"])):
                raise ValueError("A saved solver certificate differs from feasible-action replay")
        cp, alternative = (row["backends"][method] for method in METHODS)
        rebuilt = certified_repair(scenario, dispatch, search)
        if (row["cp_sat_record_matches_archived"] != (cp["record"] == trace["candidates"]["route_repair"])
                or any(rebuilt[key] != alternative[key] for key in ("record", "planner"))):
            raise ValueError("The independent action search or historical CP-SAT link did not verify")
    if _summary(archive["rows"]) != archive["summary"]:
        raise ValueError("The stored outcome totals differ from certified case rows")
    return archive["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--source", type=Path, default=RUN)
    parser.add_argument("--roads", type=Path, default=ROADS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--transition-limit", type=int, default=DEFAULT_TRANSITION_LIMIT)
    args = parser.parse_args()
    outcome = (run(args.source, args.roads, args.output, args.transition_limit) if args.command == "run"
               else verify(args.source, args.roads, args.output))
    print(json.dumps(outcome, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
