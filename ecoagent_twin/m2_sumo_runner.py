"""Run SUMO road probes and convert measured trip data into routing inputs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from math import ceil, isfinite
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from xml.etree import ElementTree


END_SECONDS = 7200
PROCESS_TIMEOUT_S = 120
INPUT_FILES = ("roads.nod.xml", "roads.edg.xml", "traffic.rou.xml")
OUTPUT_FILES = (
    "network.net.xml", "tripinfo.xml", "netconvert.version.log",
    "sumo.version.log", "netconvert.log", "sumo.log",
)


def _binaries() -> tuple[Path, Path, Path | None]:
    try:
        import sumo
    except ImportError:
        package_home = None
    else:
        package_home = getattr(sumo, "SUMO_HOME", None)

    suffix = ".exe" if os.name == "nt" else ""
    for value in (package_home, os.environ.get("SUMO_HOME")):
        if not value:
            continue
        home = Path(value).resolve()
        netconvert = home / "bin" / f"netconvert{suffix}"
        engine = home / "bin" / f"sumo{suffix}"
        if netconvert.is_file() and engine.is_file():
            return netconvert, engine, home

    netconvert_path = shutil.which("netconvert")
    engine_path = shutil.which("sumo")
    if netconvert_path and engine_path:
        return Path(netconvert_path).resolve(), Path(engine_path).resolve(), None
    raise FileNotFoundError("SUMO and netconvert were not found in eclipse-sumo or PATH")


def _as_text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _invoke(
    directory: Path, command: list[str], logfile: str, environment: dict[str, str]
) -> str:
    log_path = directory / logfile
    failure: BaseException | None = None
    return_code = None
    try:
        completed = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROCESS_TIMEOUT_S,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr = _as_text(error.stdout), _as_text(error.stderr)
        failure = error
    except OSError as error:
        stdout, stderr = "", str(error)
        failure = error

    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"argv: {command!r}\nreturncode: {return_code}\n")
        if failure is not None:
            log.write(f"process error: {failure}\n")
        log.write(f"\n[stdout]\n{stdout}\n[stderr]\n{stderr}\n")
    if failure is not None:
        raise RuntimeError(f"{Path(command[0]).name} failed to run. See {log_path}") from failure

    output = stdout + "\n" + stderr
    explicit_errors = [
        line for line in output.splitlines() if line.strip().lower().startswith("error:")
    ]
    if return_code != 0 or explicit_errors:
        detail = next(iter(explicit_errors), f"exit code {return_code}")
        raise RuntimeError(f"{Path(command[0]).name}: {detail}. See {log_path}")
    return output


def _version(output: str) -> str:
    for line in output.splitlines():
        candidate = line.strip()
        if candidate.startswith("Eclipse SUMO "):
            return candidate
    raise RuntimeError("SUMO version command returned no version string")


def run_sumo(directory: Path, seed: int) -> dict[str, Any]:
    """Run netconvert and SUMO in a directory containing the three input XML files."""
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("SUMO seed must be a nonnegative 32-bit signed integer")
    directory = directory.resolve(strict=True)
    for name in INPUT_FILES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty SUMO input: {path}")
    for name in OUTPUT_FILES:
        if (directory / name).exists():
            raise FileExistsError(f"SUMO output already exists: {directory / name}")

    netconvert, engine, home = _binaries()
    environment = os.environ.copy()
    if home is not None:
        environment["SUMO_HOME"] = str(home)
    commands = {
        "netconvert_version": [str(netconvert), "--version"],
        "sumo_version": [str(engine), "--version"],
        "netconvert": [
            str(netconvert),
            "--node-files", "roads.nod.xml",
            "--edge-files", "roads.edg.xml",
            "--output-file", "network.net.xml",
        ],
        "sumo": [
            str(engine),
            "--net-file", "network.net.xml",
            "--route-files", "traffic.rou.xml",
            "--tripinfo-output", "tripinfo.xml",
            "--tripinfo-output.write-unfinished", "true",
            "--tripinfo-output.write-undeparted", "true",
            "--time-to-teleport", "-1",
            "--max-num-teleports", "0",
            "--ignore-route-errors", "false",
            "--seed", str(seed),
            "--end", str(END_SECONDS),
            "--no-step-log", "true",
        ],
    }
    netconvert_version = _version(
        _invoke(directory, commands["netconvert_version"], "netconvert.version.log", environment)
    )
    sumo_version = _version(
        _invoke(directory, commands["sumo_version"], "sumo.version.log", environment)
    )
    _invoke(directory, commands["netconvert"], "netconvert.log", environment)
    _invoke(directory, commands["sumo"], "sumo.log", environment)
    for name in ("network.net.xml", "tripinfo.xml"):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"SUMO did not produce {path}")
    return {
        "version": sumo_version,
        "netconvert_version": netconvert_version,
        "seed": seed,
        "directory": str(directory),
        "commands": commands,
        "outputs": {name: str(directory / name) for name in OUTPUT_FILES},
    }


def _number(element: ElementTree.Element, attribute: str, vehicle_id: str) -> Decimal:
    try:
        number = Decimal(element.attrib[attribute])
    except (KeyError, InvalidOperation) as error:
        raise ValueError(f"Invalid or missing {attribute} for SUMO vehicle {vehicle_id}") from error
    if not number.is_finite() or not isfinite(number):
        raise ValueError(f"Nonfinite {attribute} for SUMO vehicle {vehicle_id}")
    return number


def read_trips(path: Path, probes: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    """Return complete EV probe measurements with conservative minute/Wh rounding."""
    if not probes or any(not probe for probe in probes):
        raise ValueError("Expected a nonempty mapping of SUMO probe IDs")
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as error:
        raise ValueError(f"Malformed SUMO tripinfo XML: {path}") from error
    if root.tag != "tripinfos":
        raise ValueError(f"Unexpected SUMO tripinfo XML root: {root.tag}")

    seen: set[str] = set()
    rows: dict[str, dict[str, Any]] = {}
    for trip in root.findall("tripinfo"):
        vehicle = trip.get("id", "")
        if not vehicle or vehicle in seen:
            raise ValueError(f"Missing or duplicate SUMO vehicle ID: {vehicle!r}")
        seen.add(vehicle)
        depart = _number(trip, "depart", vehicle)
        arrival = _number(trip, "arrival", vehicle)
        vaporized = trip.get("vaporized", "").lower() not in ("", "false", "0")
        if depart < 0 or arrival < depart or vaporized:
            raise ValueError(f"Unfinished or vaporized SUMO vehicle: {vehicle}")

        batteries = trip.findall("battery")
        if len(batteries) > 1:
            raise ValueError(f"Duplicate SUMO battery records for {vehicle}")
        battery = batteries[0] if batteries else None
        if battery is not None and _number(battery, "depleted", vehicle) != 0:
            raise ValueError(f"SUMO battery depleted for {vehicle}")
        if vehicle not in probes:
            continue
        if battery is None:
            raise ValueError(f"Missing SUMO battery output for probe {vehicle}")

        duration = _number(trip, "duration", vehicle)
        delay = _number(trip, "departDelay", vehicle)
        distance = _number(trip, "routeLength", vehicle)
        consumed = _number(battery, "totalEnergyConsumed", vehicle)
        regenerated = _number(battery, "totalEnergyRegenerated", vehicle)
        if min(duration, distance, consumed) <= 0 or min(delay, regenerated) < 0:
            raise ValueError(f"Invalid time, distance, or energy for SUMO probe {vehicle}")
        if abs(arrival - depart - duration) > Decimal("0.02"):
            raise ValueError(f"Inconsistent arrival, departure, and duration for SUMO probe {vehicle}")
        origin, destination = probes[vehicle]
        rows[vehicle] = {
            "probe_id": vehicle,
            "origin": origin,
            "destination": destination,
            "duration_s": float(duration),
            "depart_delay_s": float(delay),
            "distance_m": float(distance),
            "consumed_wh": float(consumed),
            "regenerated_wh": float(regenerated),
            "travel_min": ceil((duration + delay) / 60),
            "travel_wh": ceil(consumed),
        }
    missing = probes.keys() - rows.keys()
    if missing:
        raise ValueError(f"Missing completed SUMO probes: {', '.join(sorted(missing))}")
    return [rows[probe_id] for probe_id in sorted(rows)]
