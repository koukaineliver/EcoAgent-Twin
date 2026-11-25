"""Train MAPPO and compare it with deadline-first recovery in a shared simulator."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from statistics import fmean
from time import perf_counter_ns

from .m1_benchmark import check_world_protocol, iter_snapshots
from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .model import Scenario
from .replay import DEFAULT_SCENARIO


DEFAULT_PROTOCOL = DEFAULT_SCENARIO.with_name("m2_outages.json")
PHASES = ("training", "validation", "evaluation")
EPISODE_COLUMNS = (
    "case_id", "scenario", "day", "seed", "method", "policy_seed", "demand_packages",
    "served_packages", "failed_packages", "service_rate", "recovery_minutes",
    "censoring_horizon_minutes", "restricted_recovery_minutes", "travel_minutes", "traction_wh",
    "charged_wh", "decisions", "decision_latency_ms",
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _positive_ints(values: list, name: str) -> None:
    if (not values or any(type(value) is not int or value <= 0 for value in values)
            or len(values) != len(set(values))):
        raise ValueError(f"{name} must contain distinct positive integers")


def _worlds(protocol: dict, base: dict) -> dict[str, list[dict]]:
    check_world_protocol(protocol, base)
    if set(protocol["partitions"]) != set(PHASES):
        raise ValueError("provide training, validation, and evaluation partitions")
    _positive_ints(protocol["training"]["policy_seeds"], "policy_seeds")
    for name in ("updates", "episodes_per_update"):
        _positive_ints([protocol["training"][name]], name)
    _positive_ints([protocol["charger_capacity_wh"]], "charger_capacity_wh")
    rate = protocol["charger_rate_wh_per_min"]
    if type(rate) is not int or rate < 0:
        raise ValueError("charger_rate_wh_per_min must be a nonnegative integer")

    partitions = []
    used_days = set()
    for phase in PHASES:
        days = protocol["partitions"][phase]["days"]
        seeds = protocol["partitions"][phase]["seeds"]
        _positive_ints(days, f"{phase} days")
        _positive_ints(seeds, f"{phase} seeds")
        if used_days.intersection(days):
            raise ValueError("synthetic day IDs overlap across partitions")
        used_days.update(days)
        partitions.append((phase, days, seeds))

    worlds = {phase: [] for phase in PHASES}
    fingerprints = {}
    for key, raw in iter_snapshots(protocol, base, partitions):
        phase = key["phase"]
        state = {name: value for name, value in raw.items() if name != "id"}
        digest = sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()
        if fingerprints.setdefault(digest, phase) != phase:
            raise ValueError(f"identical initial state across partitions: {raw['id']}")
        worlds[phase].append({"case": key, "snapshot": raw})
    return worlds


def _factory(scenario: Scenario, protocol: dict):
    return partial(
        OutageEnv, scenario,
        charger_rate_wh_per_min=protocol["charger_rate_wh_per_min"],
        charger_capacity_wh=protocol["charger_capacity_wh"],
    )


def _rollout(factory, choose, measure: bool) -> dict:
    env = factory()
    actions = []
    elapsed_ns = 0
    while (view := env.observation()) is not None:
        start = perf_counter_ns() if measure else 0
        index = choose(view)
        if measure:
            elapsed_ns += perf_counter_ns() - start
        actions.append({"vehicle_id": view.vehicle_id, "time_min": view.time_min, "action_index": index})
        env.step(index)
    return {"actions": actions, "result": env.result(), "decision_latency_ms": elapsed_ns / 1_000_000}


def _methods(scenario: Scenario, checkpoints: dict):
    yield "rule_based", None, RuleBased(scenario).choose
    for seed, policy in checkpoints.items():
        yield f"mappo_{seed}", seed, policy.choose


def _row(case: dict, scenario: Scenario, policy_seed: int | None, record: dict) -> dict:
    metric = record["result"]["metrics"]
    cutoff = scenario.horizon_min - scenario.outage_min
    recovery = metric["recovery_95_min"]
    return {
        "case_id": scenario.id,
        "scenario": case["scenario"],
        "day": case["day"],
        "seed": case["seed"],
        "method": "rule_based" if policy_seed is None else "mappo",
        "policy_seed": policy_seed,
        "demand_packages": metric["total_parcels"],
        "served_packages": metric["served_parcels"],
        "failed_packages": metric["unserved_parcels"],
        "service_rate": metric["service_rate"],
        "recovery_minutes": recovery,
        "censoring_horizon_minutes": cutoff,
        "restricted_recovery_minutes": cutoff if recovery is None else recovery,
        "travel_minutes": metric["travel_min"],
        "traction_wh": metric["travel_wh"],
        "charged_wh": metric["charged_wh"],
        "decisions": len(record["actions"]),
        "decision_latency_ms": record["decision_latency_ms"],
    }


def _summary(rows: list[dict], worlds: dict[str, list[dict]]) -> dict:
    rule_by_case = {row["case_id"]: row for row in rows if row["method"] == "rule_based"}
    groups = defaultdict(list)
    for row in rows:
        method = row["method"] if row["policy_seed"] is None else f"mappo_{row['policy_seed']}"
        groups["all_conditions", method].append(row)
        groups[row["scenario"], method].append(row)

    report = {}
    for (condition, method), group in sorted(groups.items()):
        observed = [row["recovery_minutes"] for row in group if row["recovery_minutes"] is not None]
        metrics = {
            "episodes": len(group),
            "mean_service_rate": fmean(row["service_rate"] for row in group),
            "mean_failed_packages": fmean(row["failed_packages"] for row in group),
            "right_censored": len(group) - len(observed),
            "mean_observed_recovery_minutes": fmean(observed) if observed else None,
            "restricted_mean_recovery_minutes": fmean(row["restricted_recovery_minutes"] for row in group),
            "mean_travel_minutes": fmean(row["travel_minutes"] for row in group),
            "mean_traction_wh": fmean(row["traction_wh"] for row in group),
            "mean_charged_wh": fmean(row["charged_wh"] for row in group),
            "mean_decision_latency_ms_per_episode": fmean(row["decision_latency_ms"] for row in group),
        }
        if method != "rule_based":
            delta = [row["served_packages"] - rule_by_case[row["case_id"]]["served_packages"] for row in group]
            metrics["paired_against_rule"] = {
                "mean_service_delta_percentage_points": 100 * fmean(
                    difference / row["demand_packages"] for difference, row in zip(delta, group, strict=True)
                ),
                "mappo_wins": sum(difference > 0 for difference in delta),
                "ties": sum(difference == 0 for difference in delta),
                "rule_wins": sum(difference < 0 for difference in delta),
            }
        report.setdefault(condition, {})[method] = metrics
    return {
        "data_basis": "synthetic shared post-dispatch locker-outage snapshots",
        "world_counts_by_partition": {phase: len(cases) for phase, cases in worlds.items()},
        "evaluation_episode_rows": len(rows),
        "rule_baseline_reuse": "one rule episode per case reused only as the paired comparator for each checkpoint",
        "timing_scope": "sum of next-action selection latency, excluding the common event simulator",
        "by_condition": report,
    }


def run(protocol_path: Path, base_path: Path, output: Path) -> dict:
    from .m2_mappo import MAPPO, train
    import torch

    inputs = {"protocol.json": protocol_path.read_bytes(), "base_scenario.json": base_path.read_bytes()}
    protocol, base = map(json.loads, inputs.values())
    worlds = _worlds(protocol, base)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    for name, data in inputs.items():
        (output / name).write_bytes(data)
    setup = {
        phase: [_factory(Scenario.from_dict(case["snapshot"]), protocol) for case in worlds[phase]]
        for phase in ("training", "validation")
    }
    checkpoints, training_log = {}, {}
    for seed in protocol["training"]["policy_seeds"]:
        print(f"Training MAPPO seed {seed}", flush=True)
        policy, receipt = train(
            setup["training"][0](), setup["training"], setup["validation"], seed,
            protocol["training"]["updates"], protocol["training"]["episodes_per_update"],
        )
        checkpoint_path = checkpoint_dir / f"mappo_{seed}.pt"
        policy.save(checkpoint_path)
        checkpoints[seed] = MAPPO.load(checkpoint_path)
        training_log[str(seed)] = receipt
    _write_json(output / "training.json", training_log)

    rows = []
    with (output / "evaluation_traces.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for world in worlds["evaluation"]:
            scenario = Scenario.from_dict(world["snapshot"])
            factory = _factory(scenario, protocol)
            records = {}
            for name, seed, choose in _methods(scenario, checkpoints):
                records[name] = _rollout(factory, choose, measure=True)
                rows.append(_row(world["case"], scenario, seed, records[name]))
            stream.write(json.dumps({**world, "controllers": records}, allow_nan=False) + "\n")
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, worlds)
    _write_json(output / "summary.json", summary)
    artifacts = ["protocol.json", "base_scenario.json", "training.json", "evaluation_traces.jsonl",
                 "comparison.csv", "summary.json", *(f"checkpoints/mappo_{seed}.pt" for seed in checkpoints)]
    sources = ("m2_experiment.py", "m2_simulator.py", "m2_rule.py", "m2_mappo.py", "m1_benchmark.py", "model.py")
    _write_json(output / "manifest.json", {
        "schema": 1,
        "protocol_id": protocol["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "partition_counts": summary["world_counts_by_partition"],
        "sources_sha256": {name: _digest(Path(__file__).with_name(name)) for name in sources},
        "artifacts_sha256": {name: _digest(output / name) for name in artifacts},
    })
    return summary


def verify(output: Path) -> dict:
    from .m2_mappo import MAPPO
    import torch

    torch.set_num_threads(1)
    manifest = _read_json(output / "manifest.json")
    for name, digest in manifest["artifacts_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or _digest(path) != digest:
            raise ValueError(f"artifact fingerprint mismatch: {name}")
    protocol, base = (_read_json(output / name) for name in ("protocol.json", "base_scenario.json"))
    if manifest["schema"] != 1 or manifest["protocol_id"] != protocol["id"]:
        raise ValueError("manifest does not identify the saved experiment protocol")
    worlds = _worlds(protocol, base)
    checkpoints = {
        seed: MAPPO.load(output / "checkpoints" / f"mappo_{seed}.pt")
        for seed in protocol["training"]["policy_seeds"]
    }
    rows = []
    with (output / "evaluation_traces.jsonl").open(encoding="utf-8") as stream:
        for world, line in zip(worlds["evaluation"], stream, strict=True):
            trace = json.loads(line)
            if trace["case"] != world["case"] or trace["snapshot"] != world["snapshot"]:
                raise ValueError("saved evaluation snapshot differs from the protocol")
            scenario = Scenario.from_dict(world["snapshot"])
            factory = _factory(scenario, protocol)
            seen = set()
            for name, seed, choose in _methods(scenario, checkpoints):
                record = trace["controllers"][name]
                replay = _rollout(factory, choose, measure=False)
                latency = record["decision_latency_ms"]
                if not math.isfinite(latency) or latency < 0:
                    raise ValueError("invalid saved decision latency")
                if {**replay, "decision_latency_ms": latency} != record:
                    raise ValueError(f"policy replay differs for {scenario.id}, {name}")
                rows.append(_row(world["case"], scenario, seed, record))
                seen.add(name)
            if trace["controllers"].keys() != seen:
                raise ValueError("unexpected controllers in the saved trace")
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = csv.DictReader(stream)
        if csv_rows.fieldnames != list(EPISODE_COLUMNS):
            raise ValueError("comparison columns differ from the schema")
        for original, row in zip(csv_rows, rows, strict=True):
            if original != {name: "" if value is None else str(value) for name, value in row.items()}:
                raise ValueError("comparison CSV differs from policy replay")
    result = _summary(rows, worlds)
    if result != _read_json(output / "summary.json"):
        raise ValueError("summary differs from policy replay")
    return result


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
            output = args.output or Path("runs") / f"m2_{stamp}"
            result = run(args.protocol, args.scenario, output)
            print(f"Saved experiment to {output.resolve()}")
        else:
            result = verify(args.output)
        print(f"{result['world_counts_by_partition']['evaluation']} matched evaluation snapshots, "
              f"{result['evaluation_episode_rows']} controller episodes")
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
