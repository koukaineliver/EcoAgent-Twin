"""Freeze independently seeded SUMO traffic splits for the Month 3 comparisons."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .m1_benchmark import check_world_protocol, iter_snapshots
from .m2_sumo_experiment import measure, verify_measurement
from .m2_sumo_network import CONDITIONS
from .model import Scenario
from .replay import DEFAULT_SCENARIO


DEFAULT_PROTOCOL = DEFAULT_SCENARIO.with_name("m3_mvp.json")
SPLITS = ("training", "validation", "evaluation")


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def check_protocol(config: dict, base: dict) -> None:
    check_world_protocol(config, base)
    if config["traffic_delay_max_min"] or config["delay_energy_wh_per_min"]:
        raise ValueError("measured SUMO matrices must be used without traffic/energy perturbation")
    for field in ("traffic_seeds", "case_seeds"):
        if set(config[field]) != set(SPLITS):
            raise ValueError(f"{field} must have three declared splits")
        seen = set()
        for split in SPLITS:
            values = config[field][split]
            if (not values or any(type(value) is not int or value <= 0 for value in values)
                    or len(set(values)) != len(values) or seen.intersection(values)):
                raise ValueError(f"{field} must contain disjoint positive seeds in each split")
            seen.update(values)


def make_worlds(config: dict, output: Path) -> list[dict]:
    worlds = []
    for split in SPLITS:
        for seed in config["traffic_seeds"][split]:
            for traffic in CONDITIONS:
                base = read_json(output / "sumo" / str(seed) / traffic / "scenario.json")
                partitions = [(split, [seed], config["case_seeds"][split])]
                for key, raw in iter_snapshots(config, base, partitions):
                    raw["id"] = f"{raw['id']}-{traffic}"
                    Scenario.from_dict(raw)
                    worlds.append({"split": split, "traffic": traffic, "traffic_seed": seed, "case": key, "snapshot": raw})
    return worlds


def prepare(config_path: Path, base_path: Path, output: Path) -> dict:
    config, base = read_json(config_path), read_json(base_path)
    check_protocol(config, base)
    output.mkdir(parents=True, exist_ok=False)
    for name, source in (("protocol.json", config_path), ("base_scenario.json", base_path)):
        (output / name).write_bytes(source.read_bytes())
    seeds = [seed for split in SPLITS for seed in config["traffic_seeds"][split]]

    def capture(seed: int) -> tuple[str, str]:
        destination = output / "sumo" / str(seed)
        measure(output / "base_scenario.json", destination, seed)
        verify_measurement(destination)
        return str(seed), digest(destination / "manifest.json")

    with ThreadPoolExecutor(max_workers=3) as pool:
        measurements = dict(pool.map(capture, seeds))
    worlds = make_worlds(config, output)
    with (output / "worlds.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for world in worlds:
            stream.write(json.dumps(world, allow_nan=False) + "\n")
    manifest = {
        "schema": 1, "protocol_id": config["id"], "data_basis": "synthetic SUMO traffic and outage scenarios",
        "summed_ev_probe_trips": len(seeds) * 2 * 42,
        "world_counts": {split: sum(world["split"] == split for world in worlds) for split in SPLITS},
        "measurements_sha256": measurements,
        "sha256": {name: digest(output / name) for name in ("protocol.json", "base_scenario.json", "worlds.jsonl")},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_worlds(output: Path) -> tuple[dict, dict, list[dict]]:
    manifest = read_json(output / "manifest.json")
    for name in ("protocol.json", "base_scenario.json", "worlds.jsonl"):
        if digest(output / name) != manifest["sha256"][name]:
            raise ValueError(f"corpus fingerprint mismatch: {name}")
    config, base = read_json(output / "protocol.json"), read_json(output / "base_scenario.json")
    check_protocol(config, base)
    if manifest["schema"] != 1 or manifest["protocol_id"] != config["id"]:
        raise ValueError("corpus manifest does not identify the saved protocol")
    with (output / "worlds.jsonl").open(encoding="utf-8") as stream:
        worlds = [json.loads(line) for line in stream]
    return config, manifest, worlds


def verify(output: Path) -> dict:
    config, manifest, worlds = load_worlds(output)
    for split in SPLITS:
        for seed in config["traffic_seeds"][split]:
            path = output / "sumo" / str(seed)
            if digest(path / "manifest.json") != manifest["measurements_sha256"][str(seed)]:
                raise ValueError(f"SUMO manifest mismatch for seed {seed}")
            verify_measurement(path)
    if worlds != make_worlds(config, output):
        raise ValueError("scenario corpus differs from the verified SUMO measurements")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    build.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    build.add_argument("--output", type=Path)
    commands.add_parser("verify").add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or Path("runs") / f"m3_corpus_{stamp}"
            result = prepare(args.protocol, args.scenario, output)
        else:
            output, result = args.directory, verify(args.directory)
        print(output.resolve())
        print(json.dumps(result["world_counts"]))
    except (OSError, ImportError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
