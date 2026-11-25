"""Fit a separate LA-cost MAPPO policy and run the original M1 benchmark on LA paths."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean

from . import m1_benchmark
from .la_controller_experiment import PROFILES, WORKLOAD_SEEDS, _apply_profile, _protocol, _worlds
from .m1_benchmark import iter_snapshots
from .m2_offer_experiment import DEFAULT_PROTOCOL as OFFER_PROTOCOL
from .m2_rule import RuleBased
from .m3_cases import DEFAULT_PROTOCOL, digest, read_json, write_json
from .m3_experiment import _core, _environment_settings, _restricted
from .m5_experiment import _state
from .model import Scenario
from .replay import DEFAULT_SCENARIO


TRAIN = (5011, 5012, 5013, 5014, 5015, 5016)
VALIDATION = (5111, 5112)
TEST = (5211, 5212, 5213, 5214)
SOURCES = ("la_policy_training.py", "la_controller_experiment.py", "m3_rllib.py", "m2_simulator.py",
           "m1_benchmark.py", "m5_experiment.py", "m3_experiment.py", "model.py")
ARTIFACTS = ("policy.pt", "training.json", "protocol.json", "base_scenario.json", "heldout.jsonl", "summary.json")


def _training_cases(network, config, base):
    for split, workloads in (("training", TRAIN), ("validation", VALIDATION), ("evaluation", TEST)):
        for profile in PROFILES:
            raw = _apply_profile(network, base, profile)
            for key, snapshot in iter_snapshots(config, raw, [(split, workloads, config["case_seeds"][split])]):
                snapshot["id"] += f"-{profile}"
                yield {"profile": profile, "partition": split, "case": key, "snapshot": snapshot}


def _paired_inputs(network, training, config, base, offer_protocol):
    settings = _environment_settings(config)
    for world in _training_cases(network, config, base):
        if world["partition"] == "evaluation":
            yield {**world, "panel": "m3_no_offer", "dispatch": settings, "offers": None}, Scenario.from_dict(world["snapshot"])
    m5 = _protocol(training, "colocated")
    for world in _worlds(network, m5):
        if world["fleet_size"] != 2:
            continue
        scenario, dispatch, offers, _, _ = _state(world, offer_protocol)
        yield {**world, "panel": "m5_paid_offer", "dispatch": dispatch, "offers": offers}, scenario


def _statistics(rows):
    panels = defaultdict(list)
    for row in rows:
        panels[row["world"]["panel"]].append(row)
    result = {}
    for name, group in panels.items():
        arms = {}
        for controller in ("rule", "sumo_frozen_mappo", "la_frozen_mappo"):
            metrics = [row["records"][controller]["result"]["metrics"] for row in group]
            arms[controller] = {
                "mean_service_rate": fmean(x["service_rate"] for x in metrics),
                "on_time_parcels": sum(x["served_parcels"] for x in metrics),
                "mean_travel_wh": fmean(x["travel_wh"] for x in metrics),
                "mean_restricted_recovery_min": fmean(row["restricted_recovery"][controller] for row in group),
            }
        paired = [row["records"]["la_frozen_mappo"]["result"]["metrics"]["served_parcels"] -
                  row["records"]["sumo_frozen_mappo"]["result"]["metrics"]["served_parcels"] for row in group]
        result[name] = {"matched_cases": len(group), "by_controller": arms,
                        "la_minus_sumo_policy_parcels": {"wins": sum(value > 0 for value in paired),
                                                          "losses": sum(value < 0 for value in paired),
                                                          "ties": sum(value == 0 for value in paired),
                                                          "mean_per_case": fmean(paired)}}
    return result


def _rollouts(network, frozen_training, config, base, offer_protocol, fresh, frozen):
    for world, scenario in _paired_inputs(network, frozen_training, config, base, offer_protocol):
        options = {"rule": RuleBased(scenario).choose, "sumo_frozen_mappo": frozen.choose, "la_frozen_mappo": fresh.choose}
        records = {arm: _core(scenario, world["dispatch"], choose) for arm, choose in options.items()}
        restricted = {arm: _restricted(scenario, record["result"]["metrics"]) for arm, record in records.items()}
        yield {"world": world, "records": records, "restricted_recovery": restricted}


def _source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def _receipt_inputs(network_path, frozen_path):
    frozen_training = read_json(frozen_path / "training.json")
    if digest(frozen_path / "policy.pt") != frozen_training["checkpoint_sha256"]:
        raise ValueError("frozen SUMO checkpoint fingerprint differs")
    split = set(TRAIN) | set(VALIDATION) | set(TEST) | set(WORKLOAD_SEEDS)
    if len(split) != len(TRAIN) + len(VALIDATION) + len(TEST) + len(WORKLOAD_SEEDS):
        raise ValueError("LA workload partition seeds overlap")
    return read_json(network_path), frozen_training


def run(network_path, frozen_path, output):
    from .m3_rllib import RLlibMAPPO, train

    network, frozen_training = _receipt_inputs(network_path, frozen_path)
    base = read_json(DEFAULT_SCENARIO)
    config = {**read_json(DEFAULT_PROTOCOL), "id": "la-osmnx-rllib-ppo-v1",
              "traffic_seeds": {"training": list(TRAIN), "validation": list(VALIDATION), "evaluation": list(TEST)}}
    cases = list(_training_cases(network, config, base))
    training = [w["snapshot"] for w in cases if w["partition"] == "training"]
    validation = [w["snapshot"] for w in cases if w["partition"] == "validation"]
    if (len(training), len(validation)) != (108, 24):
        raise ValueError("LA training or validation population is incomplete")
    fresh, receipt = train(training, validation, **_environment_settings(config), **config["training"])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "policy.pt"
    fresh.save(checkpoint)
    write_json(output / "training.json", {
        "data_basis": "one real OSMnx road graph with modeled workload and traffic seed partitions",
        "checkpoint_sha256": digest(checkpoint), "training_workload_seeds": list(TRAIN),
        "validation_workload_seeds": list(VALIDATION), "evaluation_workload_seeds": list(TEST),
        "additional_paid_offer_test_seeds": list(WORKLOAD_SEEDS), "receipt": receipt,
    })
    write_json(output / "protocol.json", config)
    write_json(output / "base_scenario.json", base)
    frozen = RLlibMAPPO.load(frozen_path / "policy.pt")
    rows = list(_rollouts(network, frozen_training, config, base, read_json(OFFER_PROTOCOL),
                          RLlibMAPPO.load(checkpoint), frozen))
    with (output / "heldout.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = _statistics(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {
        "schema": 1, "kind": "la_osmnx_new_policy_training", "created_utc": datetime.now(timezone.utc).isoformat(),
        "road_network_sha256": digest(network_path), "frozen_sumo_checkpoint_sha256": frozen_training["checkpoint_sha256"],
        "offer_protocol_sha256": digest(OFFER_PROTOCOL), "sources": _source_hashes(),
        "sha256": {name: digest(output / name) for name in ARTIFACTS},
    })
    return summary


def verify(network_path, frozen_path, output):
    from .m3_rllib import RLlibMAPPO

    network, frozen_training = _receipt_inputs(network_path, frozen_path)
    manifest = read_json(output / "manifest.json")
    if (manifest["kind"] != "la_osmnx_new_policy_training" or manifest["sources"] != _source_hashes()
            or manifest["road_network_sha256"] != digest(network_path)
            or manifest["offer_protocol_sha256"] != digest(OFFER_PROTOCOL)
            or manifest["frozen_sumo_checkpoint_sha256"] != frozen_training["checkpoint_sha256"]
            or any(manifest["sha256"][name] != digest(output / name) for name in ARTIFACTS)):
        raise ValueError("LA training source or artifact fingerprint differs")
    training = read_json(output / "training.json")
    receipt = training["receipt"]
    chosen = max(receipt["history"], key=lambda item: (item["validation_mean_service"],
                                                       item["validation_mean_episode_return"], -item["iteration"]))
    if (training["checkpoint_sha256"] != digest(output / "policy.pt")
            or chosen["iteration"] != receipt["selected_iteration"] or not receipt["changed_actor_tensors"]):
        raise ValueError("LA checkpoint selection disagrees with validation record")
    rows = list(_rollouts(network, frozen_training, read_json(output / "protocol.json"),
                          read_json(output / "base_scenario.json"), read_json(OFFER_PROTOCOL),
                          RLlibMAPPO.load(output / "policy.pt"), RLlibMAPPO.load(frozen_path / "policy.pt")))
    with (output / "heldout.jsonl").open(encoding="utf-8") as stream:
        if list(map(json.loads, stream)) != rows:
            raise ValueError("LA policy test outcomes differ from frozen replay")
    summary = _statistics(rows)
    if read_json(output / "summary.json") != summary:
        raise ValueError("LA training summary differs from verified outcomes")
    return summary


def run_m1(network_path, output):
    network = read_json(network_path)
    base = read_json(DEFAULT_SCENARIO)
    protocol = {**read_json(m1_benchmark.DEFAULT_PROTOCOL), "id": "m1-la-osmnx-post-outage-v1"}
    output.mkdir(parents=True, exist_ok=False)
    written = output / "protocol.json"
    write_json(written, protocol)
    results = {}
    for profile in PROFILES:
        scenario_path = output / f"{profile}_input.json"
        write_json(scenario_path, _apply_profile(network, base, profile))
        profile_output = output / profile
        results[profile] = m1_benchmark.run(written, scenario_path, profile_output)
        if m1_benchmark.verify(profile_output) != results[profile]:
            raise ValueError("M1 LA benchmark could not replay its outputs")
    write_json(output / "manifest.json", {"road_network_sha256": digest(network_path),
                                          "protocol_sha256": digest(written), "source_sha256": _source_hashes(),
                                          "by_modeled_profile": {profile: digest(output / profile / "manifest.json")
                                                                 for profile in PROFILES}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify", "m1"):
        item = subs.add_parser(name)
        item.add_argument("network", type=Path)
        item.add_argument("output", type=Path)
        if name != "m1":
            item.add_argument("frozen", type=Path)
    args = parser.parse_args()
    if args.command == "m1":
        result = run_m1(args.network, args.output)
    else:
        job = run if args.command == "run" else verify
        result = job(args.network, args.frozen, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
