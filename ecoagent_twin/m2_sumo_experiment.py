"""Measure traffic and EV energy with SUMO, then replay matched locker outages."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

from .m1_benchmark import check_world_protocol, iter_snapshots
from .m2_experiment import _rollout
from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .m2_sumo_bridge import apply_travel_matrix
from .m2_sumo_network import CONDITIONS, write_demand, write_network
from .m2_sumo_runner import read_trips, run_sumo
from .model import Scenario
from .replay import DEFAULT_SCENARIO


DEFAULT_PROTOCOL = DEFAULT_SCENARIO.with_name("m2_sumo_pilot.json")
ROWS = (
    "case_id", "day", "seed", "condition", "traffic", "controller", "demand", "served",
    "recovery_min", "restricted_recovery_min", "vehicle_min", "traction_wh", "charged_wh",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _measure_summary(trips_by_traffic: dict[str, list[dict]]) -> dict:
    pairs = {traffic: {(row["origin"], row["destination"]): row for row in trips}
             for traffic, trips in trips_by_traffic.items()}
    light, heavy = (pairs[traffic] for traffic in CONDITIONS)
    if light.keys() != heavy.keys():
        raise ValueError("traffic regimes use different origin-destination pairs")
    return {
        "data_basis": "microscopic SUMO simulation on a synthetic road network",
        "completed_od_probes": {traffic: len(trips) for traffic, trips in pairs.items()},
        "profiles": {
            traffic: {
                "mean_elapsed_with_departure_delay_s": fmean(
                    row["duration_s"] + row["depart_delay_s"] for row in trips.values()
                ),
                "mean_gross_traction_wh": fmean(row["consumed_wh"] for row in trips.values()),
                "mean_regenerated_wh": fmean(row["regenerated_wh"] for row in trips.values()),
                "mean_budgeted_travel_min": fmean(row["travel_min"] for row in trips.values()),
            }
            for traffic, trips in pairs.items()
        },
        "changed_travel_min_pairs": sum(heavy[pair]["travel_min"] != row["travel_min"] for pair, row in light.items()),
        "changed_traction_wh_pairs": sum(heavy[pair]["travel_wh"] != row["travel_wh"] for pair, row in light.items()),
    }


def measure(base_path: Path, output: Path, seed: int) -> dict:
    if type(seed) is not int or not 0 <= seed <= 2_147_483_647:
        raise ValueError("SUMO seed must be a nonnegative signed 32-bit integer")
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    Scenario.from_dict(base)
    output.mkdir(parents=True, exist_ok=False)
    (output / "base_scenario.json").write_bytes(base_bytes)
    trip_profiles = {}
    for traffic in CONDITIONS:
        directory = output / traffic
        directory.mkdir()
        write_network(directory)
        probes = write_demand(directory, traffic, seed)
        _write(directory / "probes.json", probes)
        receipt = run_sumo(directory, seed)
        _write(directory / "execution.json", receipt)
        trips = read_trips(directory / "tripinfo.xml", probes)
        _write(directory / "trips.json", trips)
        _write(directory / "scenario.json", apply_travel_matrix(base, trips))
        trip_profiles[traffic] = trips
    summary = _measure_summary(trip_profiles)
    _write(output / "summary.json", summary)
    _write(output / "manifest.json", {
        "schema": 1, "kind": "sumo_measurement", "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {path.relative_to(output).as_posix(): _digest(path)
                   for path in sorted(output.rglob("*")) if path.is_file()},
        "source_sha256": {name: _digest(Path(__file__).with_name(name)) for name in (
            "m2_sumo_experiment.py", "m2_sumo_network.py", "m2_sumo_runner.py", "m2_sumo_bridge.py", "model.py",
        )},
    })
    return summary


def _verify_fingerprints(output: Path, kind: str) -> dict:
    manifest = _read(output / "manifest.json")
    if manifest["schema"] != 1 or manifest["kind"] != kind:
        raise ValueError(f"expected a {kind} run directory")
    for name, digest in manifest["sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or _digest(path) != digest:
            raise ValueError(f"artifact fingerprint mismatch: {name}")
    return manifest


def verify_measurement(output: Path) -> dict:
    _verify_fingerprints(output, "sumo_measurement")
    base = _read(output / "base_scenario.json")
    trips_by_traffic = {}
    for traffic in CONDITIONS:
        directory = output / traffic
        probes = {key: tuple(pair) for key, pair in _read(directory / "probes.json").items()}
        trips = read_trips(directory / "tripinfo.xml", probes)
        if trips != _read(directory / "trips.json"):
            raise ValueError(f"normalized trips differ from raw SUMO output: {traffic}")
        if apply_travel_matrix(base, trips) != _read(directory / "scenario.json"):
            raise ValueError(f"scenario matrices differ from raw SUMO output: {traffic}")
        trips_by_traffic[traffic] = trips
    summary = _measure_summary(trips_by_traffic)
    if summary != _read(output / "summary.json"):
        raise ValueError("SUMO measurement summary differs from the trip records")
    return summary


def _scenario_iter(protocol: dict, bases: dict):
    if protocol["traffic_delay_max_min"] != 0 or protocol["delay_energy_wh_per_min"] != 0:
        raise ValueError("SUMO replay uses measured arc tables without added travel or energy jitter")
    for field in ("days", "seeds"):
        values = protocol[field]
        if not values or any(type(value) is not int or value <= 0 for value in values) or len(values) != len(set(values)):
            raise ValueError(f"{field} must contain unique positive integers")
    for base in bases.values():
        check_world_protocol(protocol, base)
    return zip(*(iter_snapshots(protocol, bases[traffic], [("pilot", protocol["days"], protocol["seeds"])])
                 for traffic in CONDITIONS), strict=True)


def _read_policies(directory: Path) -> dict:
    if not directory.is_dir():
        return {}
    candidates = sorted(directory.glob("mappo_*.pt"))
    if not candidates:
        raise ValueError("MAPPO checkpoint directory has no checkpoint files")
    import torch
    from .m2_mappo import MAPPO

    torch.set_num_threads(1)
    return {path.stem: MAPPO.load(path) for path in candidates}


def _comparison(protocol: dict, bases: dict, policies: dict):
    for traffic_cases in _scenario_iter(protocol, bases):
        key = traffic_cases[0][0]
        rows, records = [], {}
        for traffic, (paired_key, raw) in zip(CONDITIONS, traffic_cases, strict=True):
            if key != paired_key:
                raise ValueError("light and heavy traffic snapshots are not paired")
            scenario = Scenario.from_dict(raw)
            factory = partial(OutageEnv, scenario, charger_rate_wh_per_min=protocol["charger_rate_wh_per_min"],
                              charger_capacity_wh=protocol["charger_capacity_wh"])
            controllers = {"rule_based": RuleBased(scenario).choose, **{name: policy.choose for name, policy in policies.items()}}
            traces = {}
            for name, choose in controllers.items():
                replay = _rollout(factory, choose, measure=False)
                traces[name] = {field: replay[field] for field in ("actions", "result")}
                result = replay["result"]["metrics"]
                recovery = result["recovery_95_min"]
                rows.append({
                    "case_id": scenario.id, "day": key["day"], "seed": key["seed"], "condition": key["scenario"],
                    "traffic": traffic, "controller": name, "demand": result["total_parcels"],
                    "served": result["served_parcels"], "recovery_min": recovery,
                    "restricted_recovery_min": scenario.horizon_min - scenario.outage_min if recovery is None else recovery,
                    "vehicle_min": result["travel_min"], "traction_wh": result["travel_wh"], "charged_wh": result["charged_wh"],
                })
            records[traffic] = {"snapshot": raw, "controllers": traces}
        yield {"case": key, "traffic": records}, rows


def _replay_summary(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["traffic"], row["controller"]].append(row)
    metrics = {}
    for (traffic, method), group in sorted(groups.items()):
        metrics.setdefault(traffic, {})[method] = {
            "episodes": len(group),
            "mean_on_time_service": fmean(row["served"] / row["demand"] for row in group),
            "reached_95_percent_service": sum(row["recovery_min"] is not None for row in group),
            "mean_restricted_recovery_min": fmean(row["restricted_recovery_min"] for row in group),
            "mean_traction_wh": fmean(row["traction_wh"] for row in group),
        }
    return {
        "data_basis": "synthetic outage replays with frozen SUMO travel and EV energy matrices",
        "matched_outage_snapshots": len({row["case_id"] for row in rows}), "controller_episodes": len(rows),
        "profiles": metrics,
    }


def replay(measurement: Path, protocol_path: Path, output: Path, checkpoint_run: Path | None) -> dict:
    verify_measurement(measurement)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_bytes)
    bases = {}
    for traffic in CONDITIONS:
        data = (measurement / traffic / "scenario.json").read_bytes()
        (output / f"{traffic}.json").write_bytes(data)
        bases[traffic] = json.loads(data)
    checkpoint_dir = output / "checkpoints"
    if checkpoint_run is not None:
        source = _read(checkpoint_run / "manifest.json")["artifacts_sha256"]
        checkpoint_dir.mkdir()
        for name, digest in source.items():
            path = Path(name)
            if len(path.parts) != 2 or path.parent.name != "checkpoints" or not path.name.startswith("mappo_") or path.suffix != ".pt":
                continue
            data = (checkpoint_run / path).read_bytes()
            if sha256(data).hexdigest() != digest:
                raise ValueError(f"checkpoint fingerprint mismatch: {path.name}")
            (checkpoint_dir / path.name).write_bytes(data)
    policies = _read_policies(checkpoint_dir)
    rows = []
    with (output / "traces.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for trace, episodes in _comparison(protocol, bases, policies):
            stream.write(json.dumps(trace, allow_nan=False) + "\n")
            rows.extend(episodes)
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROWS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _replay_summary(rows)
    _write(output / "summary.json", summary)
    _write(output / "manifest.json", {
        "schema": 1, "kind": "sumo_outage_replay", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sumo_measurement_manifest_sha256": _digest(measurement / "manifest.json"),
        "mode": "frozen MAPPO transfer and rule baseline" if policies else "rule baseline",
        "sha256": {path.relative_to(output).as_posix(): _digest(path)
                   for path in sorted(output.rglob("*")) if path.is_file()},
        "source_sha256": {name: _digest(Path(__file__).with_name(name)) for name in (
            "m2_sumo_experiment.py", "m1_benchmark.py", "m2_experiment.py", "m2_simulator.py", "m2_rule.py", "m2_mappo.py", "model.py",
        )},
    })
    return summary


def verify_replay(output: Path) -> dict:
    _verify_fingerprints(output, "sumo_outage_replay")
    protocol = _read(output / "protocol.json")
    bases = {traffic: _read(output / f"{traffic}.json") for traffic in CONDITIONS}
    policies = _read_policies(output / "checkpoints")
    rows = []
    with (output / "traces.jsonl").open(encoding="utf-8") as stream:
        for line, (trace, episodes) in zip(stream, _comparison(protocol, bases, policies), strict=True):
            if json.loads(line) != trace:
                raise ValueError(f"saved fleet trace differs from replay: {trace['case']}")
            rows.extend(episodes)
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(ROWS):
            raise ValueError("comparison columns differ from replay schema")
        for saved, result in zip(reader, rows, strict=True):
            if saved != {name: "" if value is None else str(value) for name, value in result.items()}:
                raise ValueError("comparison row differs from fleet replay")
    summary = _replay_summary(rows)
    if summary != _read(output / "summary.json"):
        raise ValueError("fleet summary differs from replay")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probes = commands.add_parser("measure")
    probes.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    probes.add_argument("--seed", type=int, default=47)
    probes.add_argument("--output", type=Path)
    commands.add_parser("verify-measurement").add_argument("output", type=Path)
    experiment = commands.add_parser("replay")
    experiment.add_argument("measurement", type=Path)
    experiment.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    experiment.add_argument("--checkpoint-run", type=Path)
    experiment.add_argument("--output", type=Path)
    commands.add_parser("verify-replay").add_argument("output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        if args.command == "measure":
            output = args.output or Path("runs") / f"m2_sumo_measure_{stamp}"
            summary = measure(args.scenario, output, args.seed)
        elif args.command == "replay":
            output = args.output or Path("runs") / f"m2_sumo_replay_{stamp}"
            summary = replay(args.measurement, args.protocol, output, args.checkpoint_run)
        elif args.command == "verify-measurement":
            output, summary = args.output, verify_measurement(args.output)
        else:
            output, summary = args.output, verify_replay(args.output)
        print(output.resolve())
        print(json.dumps(summary, indent=2, allow_nan=False))
    except (OSError, ImportError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
