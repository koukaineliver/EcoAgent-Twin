"""Run a locker-outage comparison or verify a saved comparison by replay."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from . import __version__
from .model import Scenario
from .replay import DEFAULT_SCENARIO, METHODS, compare, replay_saved


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(scenario_path: Path, output_dir: Path) -> None:
    source = scenario_path.read_bytes()
    scenario = Scenario.from_dict(json.loads(source))
    records = {}
    rows = []
    for method, (record, latency_ms) in compare(scenario).items():
        records[method] = record
        rows.append({"method": method, **record["metrics"], "decision_latency_ms": latency_ms})

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "scenario.json").write_bytes(source)
    _write_json(output_dir / "replay.json", records)
    with (output_dir / "comparison.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = ["__main__.py", "model.py", "controllers.py", "replay.py"]
    _write_json(output_dir / "manifest.json", {
        "schema": 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "scenario_id": scenario.id,
        "data_basis": "synthetic",
        "runtime_python": platform.python_version(),
        "prototype_version": __version__,
        "controller_randomness": "none",
        "sha256": {name: _sha256(output_dir / name) for name in ("scenario.json", "replay.json", "comparison.csv")},
        "source_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in sources},
    })
    print(f"Saved run to {output_dir.resolve()}")
    for row in rows:
        print(f"{row['method']}: {row['served_parcels']}/{row['total_parcels']} served, "
              f"{row['travel_min']} min travel, {row['travel_wh']} Wh")


def _verify(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "manifest.json")
    for name in ("scenario.json", "replay.json", "comparison.csv"):
        if _sha256(run_dir / name) != manifest["sha256"][name]:
            raise ValueError(f"artifact fingerprint mismatch: {name}")
    scenario = Scenario.from_dict(_read_json(run_dir / "scenario.json"))
    records = _read_json(run_dir / "replay.json")
    with (run_dir / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if list(records) != list(METHODS) or [row["method"] for row in rows] != list(METHODS):
        raise ValueError("missing, repeated, or unexpected methods in saved comparison")
    for method, row in zip(METHODS, rows, strict=True):
        record = records[method]
        replay = replay_saved(scenario, record)
        if replay != record:
            raise ValueError(f"recorded routes do not reproduce the trace: {method}")
        for name, value in replay["metrics"].items():
            if row[name] != ("" if value is None else str(value)):
                raise ValueError(f"comparison does not reproduce {method}.{name}")
    print(f"Verified fingerprints, replay, and service metrics for {len(records)} controllers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run all controllers on the same outage snapshot")
    run.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    run.add_argument("--output", type=Path)
    verify = commands.add_parser("verify", help="check saved files and replay all recorded routes")
    verify.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            _run(args.scenario, args.output if args.output else Path("runs") / stamp)
        else:
            _verify(args.run_dir)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
