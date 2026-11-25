"""Compare modeled single and follow-up credits on matched road cost tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from hashlib import file_digest, sha256
import json
from math import ceil
from pathlib import Path
from statistics import fmean

from .m1_benchmark import iter_snapshots
from .m2_experiment import _rollout
from .m2_offer_experiment import DEFAULT_PROTOCOL
from .m2_offers import UtilityProfile
from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .m2_sumo_experiment import verify_measurement
from .m2_sumo_network import CONDITIONS
from .m6_incentive import ARMS, negotiate
from .model import Scenario
from .replay import DEFAULT_SCENARIO


INPUTS = "inputs.json"
TRACE = "negotiation_traces.jsonl"
SUMMARY = "summary.json"
SEEDS = (821, 822, 823, 824)
DAYS = (531, 532, 533, 534)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _write(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _profile_assignment(scenario: Scenario, case: dict, profile_table: dict) -> dict[str, str]:
    names = tuple(sorted(profile_table))
    if not names:
        raise ValueError("The protocol must contain modeled preference profiles")
    return {
        parcel.id: names[int.from_bytes(sha256(
            f"m6-profile:{case['day']}:{case['seed']}:{parcel.id}".encode()
        ).digest()[:8], "big") % len(names)]
        for parcel in scenario.parcels
    }


def _traces(inputs: dict):
    protocol = inputs["protocol"]
    table = {name: UtilityProfile(**values) for name, values in protocol["profiles"].items()}
    for traffic in CONDITIONS:
        base = inputs["scenario_by_traffic"][traffic]
        traffic_protocol = {**protocol, "locker_pickup": inputs["road_pickup_by_traffic"][traffic]} if (
            "road_pickup_by_traffic" in inputs) else protocol
        for case, raw in iter_snapshots(traffic_protocol, base, [("incentive_pilot", DAYS, SEEDS)]):
            scenario = Scenario.from_dict(raw)
            names = _profile_assignment(scenario, case, table)
            profiles = {parcel: table[name] for parcel, name in names.items()}
            outcomes = {}
            for arm in ARMS:
                routed, dispatch, offers, fallback = negotiate(scenario, traffic_protocol, case, profiles, arm=arm)
                factory = partial(
                    OutageEnv, routed, charger_rate_wh_per_min=protocol["charger_rate_wh_per_min"],
                    charger_capacity_wh=protocol["charger_capacity_wh"],
                    preassigned_slots=offers["assignments"], dispatch_after_min=dispatch,
                )
                fleet = _rollout(factory, RuleBased(routed).choose, measure=False)
                outcomes[arm] = {
                    "dispatch_min": dispatch, "offers": offers,
                    "fallbacks": fallback, "fleet_actions": fleet["actions"], "fleet": fleet["result"],
                }
            if outcomes["personalized"]["offers"]["rounds"][0] != outcomes["adaptive"]["offers"]["rounds"][0]:
                raise ValueError("Adaptive and personalized choices differed before any first response")
            yield {
                "case": {**case, "traffic": traffic},
                "synthetic_profile_by_parcel": names,
                "outcomes": outcomes,
            }


def _summarize(traces: list[dict], road_basis: dict | None = None) -> dict:
    groups = defaultdict(list)
    for trace in traces:
        matches = trace["outcomes"]
        if set(matches) != set(ARMS):
            raise ValueError("A matched case is missing an incentive arm")
        if len({outcome["dispatch_min"] for outcome in matches.values()}) != 1:
            raise ValueError("The incentive arms used different dispatch times")
        for arm, outcome in matches.items():
            groups[arm].append(outcome)
    aggregate = {}
    for arm, outcomes in groups.items():
        results = [value["offers"]["metrics"] for value in outcomes]
        fleet = [value["fleet"]["metrics"] for value in outcomes]
        metric = {name: sum(result[name] for result in results) for name in (
            "issued", "accepted", "reassignments", "spent_cents", "followup_issued", "followup_accepted",
        )}
        metric["modeled_acceptances_per_offer"] = (
            metric["accepted"] / metric["issued"] if metric["issued"] else None
        )
        aggregate[arm] = {
            **metric, "episodes": len(outcomes), "served": sum(row["served_parcels"] for row in fleet),
            "mean_on_time_service": fmean(row["service_rate"] for row in fleet),
        }
    report = {
        "cases": len(traces), "arm_episodes": len(traces) * len(ARMS),
        "basis": "SUMO-derived synthetic traffic matrices, with simulated Mesa customer choices and assumed preference profiles",
        "total_credit_scope": "Same 400-cent cap per case; spent amounts are simulated",
        "by_arm": aggregate,
    }
    if road_basis is not None:
        report["basis"] = "Actual University Park OSM road distances with modeled EV traffic and customer preferences"
        report["road_basis"] = road_basis
    return report


def _sumo_inputs(measurement: Path) -> dict:
    verify_measurement(measurement)
    protocol = json.loads(DEFAULT_PROTOCOL.read_bytes())
    protocol.update({
        "id": "m6-incentive-sumo-pilot-v1",
        "traffic_delay_max_min": 0,
        "delay_energy_wh_per_min": 0,
    })
    return {
        "measurement_manifest_sha256": _digest(measurement / "manifest.json"),
        "protocol": protocol,
        "scenario_by_traffic": {
            condition: json.loads((measurement / condition / "scenario.json").read_bytes())
            for condition in CONDITIONS
        },
    }


def _osmnx_inputs(roads: Path) -> dict:
    network = json.loads(roads.read_bytes())
    if network["provenance"]["osmnx_version"] == "":
        raise ValueError("Missing OSMnx network provenance")
    base = json.loads(DEFAULT_SCENARIO.read_bytes())
    protocol = json.loads(DEFAULT_PROTOCOL.read_bytes())
    protocol.update({"id": "m6-incentive-la-osmnx-v1", "traffic_delay_max_min": 0, "delay_energy_wh_per_min": 0})
    origin = base["outage"]["locker_id"]
    scenarios, pickup_profiles = {}, {}
    for condition in CONDITIONS:
        profile = network["profiles"][condition]
        distances = profile["distance_m"]
        scenarios[condition] = {**base, "travel_minutes": profile["travel_minutes"], "travel_wh": profile["travel_wh"]}
        pickup_profiles[condition] = {}
        for site in protocol["locker_pickup"]:
            meters = distances[origin][site] + distances[site][origin]
            pickup_profiles[condition][site] = {
                "pickup_min": ceil(meters / 1000 * 60 / 4.5),
                "pickup_cost_cents": protocol["locker_pickup"][site]["pickup_cost_cents"],
            }
    return {
        "protocol": protocol, "scenario_by_traffic": scenarios, "road_pickup_by_traffic": pickup_profiles,
        "road_basis": {"sha256": _digest(roads), "provider": network["provenance"]["attribution"],
                       "pickup": "Directed shared-road round trip at an assumed effective 4.5 km/h. Costs use the prior assumptions."},
    }


def run(measurement: Path, output: Path, *, osmnx: bool = False) -> dict:
    inputs = _osmnx_inputs(measurement) if osmnx else _sumo_inputs(measurement)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / INPUTS, inputs)
    traces = []
    with (output / TRACE).open("x", encoding="utf-8", newline="\n") as stream:
        for trace in _traces(inputs):
            stream.write(json.dumps(trace, allow_nan=False) + "\n")
            traces.append(trace)
    summary = _summarize(traces, inputs.get("road_basis"))
    _write(output / SUMMARY, summary)
    _write(output / "manifest.json", {
        "schema": 2 if osmnx else 1,
        **({"road_network_sha256": inputs["road_basis"]["sha256"]} if osmnx else
           {"measurement_manifest_sha256": inputs["measurement_manifest_sha256"]}),
        "sha256": {name: _digest(output / name) for name in (INPUTS, TRACE, SUMMARY)},
    })
    return summary


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest["schema"] not in (1, 2) or any(
        _digest(output / name) != manifest["sha256"][name] for name in (INPUTS, TRACE, SUMMARY)
    ):
        raise ValueError("Saved incentive artifacts failed manifest checks")
    inputs = json.loads((output / INPUTS).read_bytes())
    traces = []
    with (output / TRACE).open(encoding="utf-8") as stream:
        for rerun, saved in zip(_traces(inputs), stream, strict=True):
            if rerun != json.loads(saved):
                raise ValueError("Stored incentive or downstream fleet events did not replay")
            traces.append(rerun)
    if manifest["schema"] == 2 and inputs["road_basis"]["sha256"] != manifest["road_network_sha256"]:
        raise ValueError("Incentive road input provenance mismatch")
    report = _summarize(traces, inputs.get("road_basis"))
    if report != json.loads((output / SUMMARY).read_bytes()):
        raise ValueError("Saved incentive summary differs from replay")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trial = commands.add_parser("run")
    trial.add_argument("measurement", type=Path)
    trial.add_argument("--output", type=Path)
    trial.add_argument("--osmnx", action="store_true", help="use the pinned University Park OSMnx route JSON")
    replay = commands.add_parser("verify")
    replay.add_argument("output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("runs") / f"m6_incentive_{stamp}"
    try:
        report = run(args.measurement, output, osmnx=args.osmnx) if args.command == "run" else verify(output)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(output.resolve())
    print(f"{report['cases']} matched cases and {report['arm_episodes']} arm episodes")


if __name__ == "__main__":
    main()
