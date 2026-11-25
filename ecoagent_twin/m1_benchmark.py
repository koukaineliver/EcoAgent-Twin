"""Run and verify the matched synthetic M1 outage benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from random import Random
import re
from statistics import fmean

from . import __version__
from .model import Scenario
from .replay import DEFAULT_SCENARIO, METHODS, compare, replay_saved


DEFAULT_PROTOCOL = DEFAULT_SCENARIO.with_name("m1_benchmark.json")
ARTIFACTS = ("protocol.json", "base_scenario.json", "cases.jsonl", "episodes.csv", "summary.json")
EPISODE_COLUMNS = (
    "phase", "scenario", "day", "seed", "method", "demand_packages", "served_packages",
    "failed_packages", "service_rate", "recovery_minutes", "censoring_horizon_minutes",
    "restricted_recovery_minutes", "travel_minutes", "traction_wh", "decision_latency_ms",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_protocol(protocol: dict, base: dict) -> None:
    for field in ("development_days", "evaluation_days", "seeds"):
        values = protocol[field]
        if (not values or any(type(value) is not int or value <= 0 for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{field} must contain unique positive integers")
    if set(protocol["development_days"]) & set(protocol["evaluation_days"]):
        raise ValueError("development and evaluation day IDs overlap")
    check_world_protocol(protocol, base)


def check_world_protocol(protocol: dict, base: dict) -> None:
    """Validate the synthetic variation shared by the M1 and M2 benchmarks."""
    if protocol["schema"] != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", protocol["id"]):
        raise ValueError("invalid benchmark ID or schema")
    for field in ("deadline_jitter_min", "traffic_delay_max_min", "delay_energy_wh_per_min"):
        if type(protocol[field]) is not int or protocol[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    conditions = protocol["conditions"]
    identifiers = [condition["id"] for condition in conditions]
    if (not identifiers or "all_conditions" in identifiers or len(identifiers) != len(set(identifiers))
            or not all(re.fullmatch(r"[a-z][a-z0-9_]*", identifier) for identifier in identifiers)):
        raise ValueError("conditions must have distinct slug IDs")
    for condition in conditions:
        for locker, cap in condition["locker_capacity_caps"].items():
            if locker not in base["locker_free_capacity"] or type(cap) is not int or cap < 0:
                raise ValueError("capacity caps must name lockers and use nonnegative integers")
        extra = condition["extra_battery_above_direct_return_wh"]
        if extra is not None and (type(extra) is not int or extra < 0):
            raise ValueError("battery cap must be null or a nonnegative integer")
    Scenario.from_dict(base)


def _perturb(base: dict, protocol: dict, day: int, seed: int) -> dict:
    """Generate common synthetic inputs before applying the condition."""
    raw = deepcopy(base)
    rng = Random(f"{protocol['id']}:{day}:{seed}")
    day_delay_max = day % (protocol["traffic_delay_max_min"] + 1)
    for start in sorted(raw["travel_minutes"]):
        for end in sorted(raw["travel_minutes"][start]):
            if start != end:
                delay = rng.randint(0, day_delay_max)
                raw["travel_minutes"][start][end] += delay
                raw["travel_wh"][start][end] += delay * protocol["delay_energy_wh_per_min"]
    jitter = protocol["deadline_jitter_min"]
    for parcel in raw["parcels_in_original_plan_order"]:
        proposal = parcel["deadline_min"] + rng.randint(-jitter, jitter)
        parcel["deadline_min"] = min(raw["horizon_min"], max(raw["outage"]["time_min"], proposal))
    return raw


def _cases(protocol: dict, base: dict):
    partitions = [(phase, protocol[f"{phase}_days"], protocol["seeds"])
                  for phase in ("development", "evaluation")]
    yield from iter_snapshots(protocol, base, partitions)


def iter_snapshots(protocol: dict, base: dict, partitions):
    for phase, days, seeds in partitions:
        for day in days:
            for seed in seeds:
                common = _perturb(base, protocol, day, seed)
                for condition in protocol["conditions"]:
                    raw = deepcopy(common)
                    condition_id = condition["id"]
                    raw["id"] = f"{protocol['id']}-{phase}-{day:02d}-{seed:02d}-{condition_id}"
                    for locker, cap in condition["locker_capacity_caps"].items():
                        raw["locker_free_capacity"][locker] = min(raw["locker_free_capacity"][locker], cap)
                    extra = condition["extra_battery_above_direct_return_wh"]
                    if extra is not None:
                        for vehicle in raw["vehicles"]:
                            return_wh = raw["travel_wh"][vehicle["node"]][raw["depot"]]
                            max_wh = vehicle["reserve_wh"] + return_wh + extra
                            vehicle["battery_wh"] = min(vehicle["battery_wh"], max_wh)
                    yield {"phase": phase, "scenario": condition_id, "day": day, "seed": seed}, raw


def _episode_row(key: dict, scenario: Scenario, method: str, record: dict, latency_ms: float) -> dict:
    metric = record["metrics"]
    recovery = metric["recovery_95_min"]
    cutoff = scenario.horizon_min - scenario.outage_min
    return {
        **key,
        "method": method,
        "demand_packages": metric["total_parcels"],
        "served_packages": metric["served_parcels"],
        "failed_packages": metric["unserved_parcels"],
        "service_rate": metric["service_rate"],
        "recovery_minutes": recovery,
        "censoring_horizon_minutes": cutoff,
        "restricted_recovery_minutes": cutoff if recovery is None else recovery,
        "travel_minutes": metric["travel_min"],
        "traction_wh": metric["travel_wh"],
        "decision_latency_ms": latency_ms,
    }


def _summary(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["phase"], "all_conditions", row["method"]].append(row)
        groups[row["phase"], row["scenario"], row["method"]].append(row)
    summary = {}
    for (phase, condition, method), group in sorted(groups.items()):
        by_phase = summary.setdefault(phase, {}).setdefault(condition, {})
        recoveries = [row["recovery_minutes"] for row in group if row["recovery_minutes"] is not None]
        by_phase[method] = {
            "episodes": len(group),
            "mean_service_rate": fmean(row["service_rate"] for row in group),
            "mean_failed_packages": fmean(row["failed_packages"] for row in group),
            "observed_recoveries": len(recoveries),
            "right_censored": len(group) - len(recoveries),
            "mean_observed_recovery_minutes": fmean(recoveries) if recoveries else None,
            "restricted_mean_recovery_minutes": fmean(row["restricted_recovery_minutes"] for row in group),
            "mean_traction_wh": fmean(row["traction_wh"] for row in group),
        }
    return {
        "data_basis": "synthetic engineering benchmark",
        "matched_snapshots": len(rows) // len(METHODS),
        "episode_rows": len(rows),
        "recovery_clock": "Minutes from alert to the deposit of ceil(0.95 × fixed onboard demand) unique parcels.",
        "restricted_recovery_clock": "Observed recovery or the post-alert horizon for a right-censored episode.",
        "by_phase": summary,
    }


def run(protocol_path: Path, scenario_path: Path, output: Path) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    scenario_bytes = scenario_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    base = json.loads(scenario_bytes)
    _check_protocol(protocol, base)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "base_scenario.json").write_bytes(scenario_bytes)
    rows = []
    with (output / "cases.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for key, raw in _cases(protocol, base):
            scenario = Scenario.from_dict(raw)
            decisions = compare(scenario)
            records = {name: record for name, (record, _) in decisions.items()}
            stream.write(json.dumps({"case": key, "snapshot": raw, "controllers": records}, allow_nan=False) + "\n")
            rows.extend(_episode_row(key, scenario, method, record, latency)
                        for method, (record, latency) in decisions.items())
    with (output / "episodes.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifest.json", {
        "schema": 1,
        "protocol_id": protocol["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "prototype_version": __version__,
        "sha256": {name: _digest(output / name) for name in ARTIFACTS},
        "source_sha256": {
            name: _digest(Path(__file__).with_name(name))
            for name in ("m1_benchmark.py", "replay.py", "model.py", "controllers.py")
        },
    })
    return summary


def verify(output: Path) -> dict:
    manifest = _json(output / "manifest.json")
    for name in ARTIFACTS:
        if _digest(output / name) != manifest["sha256"][name]:
            raise ValueError(f"artifact fingerprint mismatch: {name}")
    protocol, base = (_json(output / name) for name in ("protocol.json", "base_scenario.json"))
    _check_protocol(protocol, base)
    if manifest["schema"] != 1 or manifest["protocol_id"] != protocol["id"]:
        raise ValueError("manifest does not identify the saved benchmark protocol")
    rows = []
    with (output / "episodes.csv").open(encoding="utf-8", newline="") as csv_stream, (
        output / "cases.jsonl"
    ).open(encoding="utf-8") as cases_stream:
        csv_rows = csv.DictReader(csv_stream)
        if csv_rows.fieldnames != list(EPISODE_COLUMNS):
            raise ValueError("episode columns differ from the benchmark schema")
        for (key, raw), line in zip(_cases(protocol, base), cases_stream, strict=True):
            case = json.loads(line)
            if case["case"] != key or case["snapshot"] != raw or list(case["controllers"]) != list(METHODS):
                raise ValueError(f"snapshot or methods differ from the protocol: {key}")
            scenario = Scenario.from_dict(raw)
            rerun = compare(scenario)
            for method, record in case["controllers"].items():
                replay = replay_saved(scenario, record)
                if replay != record:
                    raise ValueError(f"saved events do not replay for {key}, {method}")
                if rerun[method][0] != record:
                    raise ValueError(f"controller rerun differs from saved plan for {key}, {method}")
                original = next(csv_rows, None)
                if original is None:
                    raise ValueError("missing episode row")
                latency = float(original["decision_latency_ms"])
                if not math.isfinite(latency) or latency < 0:
                    raise ValueError("invalid recorded decision latency")
                row = _episode_row(key, scenario, method, replay, latency)
                if original != {name: "" if value is None else str(value) for name, value in row.items()}:
                    raise ValueError(f"episode row differs from replay: {key}, {method}")
                rows.append(row)
        if next(csv_rows, None) is not None:
            raise ValueError("unexpected extra episode row")
    summary = _summary(rows)
    if summary != _json(output / "summary.json"):
        raise ValueError("summary differs from replayed episodes")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    run_parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    run_parser.add_argument("--output", type=Path)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or Path("runs") / f"m1_{stamp}"
            result = run(args.protocol, args.scenario, output)
            print(f"Saved benchmark to {output.resolve()}")
        else:
            result = verify(args.output)
        print(f"{result['matched_snapshots']} matched snapshots, {result['episode_rows']} controller episodes")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
