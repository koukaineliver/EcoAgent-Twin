"""Prepare fresh SUMO traffic inputs for the frozen Month 4 chaos protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .m2_sumo_experiment import measure, verify_measurement
from .m3_cases import digest, read_json, write_json
from .replay import DEFAULT_SCENARIO


DEFAULT_PROTOCOL = DEFAULT_SCENARIO.with_name("m4_chaos_pilot.json")


def _validate(protocol: dict) -> None:
    if protocol["schema"] != 1 or not protocol["id"]:
        raise ValueError("expected a version 1 Month 4 protocol")
    seeds = protocol["traffic_seeds"]
    if (not seeds or any(type(seed) is not int or seed <= 0 for seed in seeds)
            or len(seeds) != len(set(seeds)) or len(seeds) % 2):
        raise ValueError("counterbalanced traffic requires an even number of distinct positive seeds")
    if set(protocol["case_seeds"]) != {"moderate", "severe"} or len(set(protocol["case_seeds"].values())) != 2:
        raise ValueError("moderate and severe cases require separate case seeds")
    if protocol["traffic_delay_max_min"] or protocol["delay_energy_wh_per_min"]:
        raise ValueError("Month 4 cases use fresh SUMO matrices before declared disruption overlays")


def prepare(config_path: Path, output: Path) -> None:
    config = read_json(config_path)
    _validate(config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(config_path.read_bytes())
    (output / "base_scenario.json").write_bytes(DEFAULT_SCENARIO.read_bytes())

    def capture(seed: int) -> tuple[str, str]:
        folder = output / "sumo" / str(seed)
        measure(output / "base_scenario.json", folder, seed)
        verify_measurement(folder)
        return str(seed), digest(folder / "manifest.json")

    with ThreadPoolExecutor(max_workers=3) as pool:
        measurements = dict(pool.map(capture, config["traffic_seeds"]))
    write_json(output / "manifest.json", {
        "schema": 1, "kind": "month_4_sumo_corpus", "protocol_id": config["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "completed_ev_probes": 84 * len(measurements),
        "measurements_sha256": measurements,
        "sha256": {name: digest(output / name) for name in ("protocol.json", "base_scenario.json")},
    })


def verify(output: Path) -> dict:
    manifest = read_json(output / "manifest.json")
    config = read_json(output / "protocol.json")
    _validate(config)
    if (manifest["schema"] != 1 or manifest["kind"] != "month_4_sumo_corpus"
            or config["id"] != manifest["protocol_id"]):
        raise ValueError("traffic corpus does not identify the saved Month 4 protocol")
    for name in ("protocol.json", "base_scenario.json"):
        if digest(output / name) != manifest["sha256"][name]:
            raise ValueError(f"corpus fingerprint differs for {name}")
    for seed in config["traffic_seeds"]:
        folder = output / "sumo" / str(seed)
        if digest(folder / "manifest.json") != manifest["measurements_sha256"][str(seed)]:
            raise ValueError(f"SUMO measurement fingerprint differs for seed {seed}")
        verify_measurement(folder)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    create.add_argument("--output", type=Path)
    commands.add_parser("verify").add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or Path("runs") / f"m4_corpus_{stamp}"
            prepare(args.protocol, output)
        else:
            output = args.output
            verify(output)
        print(output.resolve())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
