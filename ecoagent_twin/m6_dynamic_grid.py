"""Apply verified hourly EIA-930 load intensity to saved Amazon route energy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from math import isclose, isfinite
from pathlib import Path, PurePosixPath
from statistics import fmean
from urllib.request import urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .empirical_amazon import file_sha256
from .empirical_energy import CAMX_GRID_LOSS


SOURCE_URL = "https://www.eia.gov/electricity/gridmonitor/knownissues/xls/CISO.xlsx"
METHOD_URL = "https://www.eia.gov/electricity/930-content/about.html"
JUNE_REPORT = "https://www.caiso.com/documents/greenhousegasemissions-trackingreport-june2018.pdf"
HOUR = timedelta(hours=1)
EXCEL_EPOCH = datetime(1899, 12, 30, tzinfo=timezone.utc)
LB_TO_KG = 0.45359237
POST_HORIZON_OFFSET_MIN = 15
ARTIFACTS = ("hourly_rates.json", "comparison.jsonl", "summary.json")


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime
    wh: float
    kind: str


def _hour_energy(interval: Interval):
    """Split a constant-power interval at UTC hour endings."""
    start = interval.start
    duration = (interval.end - start).total_seconds()
    if duration <= 0 or interval.wh < 0:
        raise ValueError("a battery-energy interval must have positive duration and nonnegative energy")
    while start < interval.end:
        hour_end = start.replace(minute=0, second=0, microsecond=0) + HOUR
        stop = min(hour_end, interval.end)
        yield hour_end, interval.wh * (stop - start).total_seconds() / duration
        start = stop


def _energy_intervals(world: dict, events: list[dict], row: dict) -> list[Interval]:
    alert = datetime.fromisoformat(world["metadata"]["projected_alert_utc"]).astimezone(timezone.utc)
    charging = []
    pending = {}
    for event in events:
        if event["kind"] == "charging_started":
            vehicle = event["vehicle_id"]
            if vehicle in pending:
                raise ValueError("duplicate charge start")
            pending[vehicle] = event["time_min"]
        elif event["kind"] == "charging_completed":
            charge_start = pending.pop(event["vehicle_id"])
            charging.append((charge_start, event["time_min"], event["charged_wh"]))
    if pending or sum(amount for _, _, amount in charging) != row["onboard_recharge_wh"]:
        raise ValueError("saved charge events do not reconcile with the energy receipt")
    consumed = row["travel_wh"] + row["carrier_fallback_traction_wh"]
    if row["initial_battery_wh"] + row["onboard_recharge_wh"] - row["travel_wh"] != row["terminal_battery_wh"]:
        raise ValueError("saved battery inventory does not reconcile")
    in_route = min(consumed, row["onboard_recharge_wh"])
    # Charge beyond this comparison's retired battery energy remains inventory
    # for other work. Prorate within saved charging intervals without a negative
    # meter reading or a credit at an unrelated later carbon intensity.
    intervals = [
        Interval(alert + timedelta(minutes=start), alert + timedelta(minutes=end),
                 amount * in_route / row["onboard_recharge_wh"], "in_route")
        for start, end, amount in charging if in_route
    ]
    remaining = consumed - in_route
    if remaining:
        begin = alert + timedelta(minutes=world["snapshot"]["horizon_min"] + POST_HORIZON_OFFSET_MIN)
        duration = timedelta(minutes=remaining / world["charger"]["charger_rate_wh_per_min"])
        intervals.append(Interval(begin, begin + duration, remaining, "post_horizon"))
    return intervals


def _workbook_sheet(archive: ZipFile) -> tuple[str, list[str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(
        child for child in workbook.findall(".//{*}sheet") if child.get("name") == "Published Hourly Data"
    )
    link = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = next(rel.attrib["Target"] for rel in relationships if rel.attrib["Id"] == link)
    path = PurePosixPath("xl") / target
    if ".." in path.parts or path.is_absolute():
        raise ValueError("unexpected worksheet location")
    strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return path.as_posix(), ["".join(cell.itertext()) for cell in strings]


def _cells(row, strings: list[str]) -> dict[str, str]:
    result = {}
    for cell in row:
        value = cell.find("{*}v")
        if value is not None and value.text is not None:
            col = cell.attrib["r"].rstrip("0123456789")
            result[col] = strings[int(value.text)] if cell.get("t") == "s" else value.text
    return result


def _rates(workbook: Path, wanted: set[datetime]) -> tuple[dict[datetime, dict], datetime | None]:
    """Match UTC hour endings and independently check the workbook's rate units."""
    if not wanted:
        return {}, None
    earliest = None
    matched = {}
    stop = max(wanted)
    with ZipFile(workbook) as archive:
        path, strings = _workbook_sheet(archive)
        columns = {}
        with archive.open(path) as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if not row.tag.endswith("}row"):
                    continue
                cells = _cells(row, strings)
                if not columns:
                    titles = {value: col for col, value in cells.items()}
                    columns = {alias: titles[title] for alias, title in {
                        "ba": "BA", "utc": "UTC time", "co2": "CO2 Emissions Consumed",
                        "mwh": "Consumed Electricity", "rate": "CO2 Emissions Intensity for Consumed Electricity",
                    }.items()}
                else:
                    hour_end = EXCEL_EPOCH + int(round(float(cells[columns["utc"]]) * 24)) * HOUR
                    if hour_end > stop:
                        break
                    if columns["rate"] in cells:
                        if earliest is None:
                            earliest = hour_end
                        if hour_end in wanted:
                            value = float(cells[columns["rate"]])
                            tons = float(cells[columns["co2"]])
                            mwh = float(cells[columns["mwh"]])
                            if (
                                cells[columns["ba"]] != "CISO" or hour_end in matched
                                or not all(map(isfinite, (value, tons, mwh)))
                                or value < 0 or mwh <= 0
                                or not isclose(value * LB_TO_KG, tons / mwh, rel_tol=2e-6, abs_tol=1e-8)
                            ):
                                raise ValueError(f"invalid EIA CISO intensity at {hour_end.isoformat()}")
                            matched[hour_end] = {
                                "xlsx_row": int(row.attrib["r"]), "consumed_mwh": mwh,
                                "consumed_metric_tons_co2": tons, "published_lb_co2_per_kwh": value,
                                "kg_co2_per_kwh": tons / mwh,
                            }
                row.clear()
    return matched, earliest


def _source(input_dir: Path):
    manifest = json.loads((input_dir / "manifest.json").read_bytes())
    for name in ("worlds.jsonl", "observations.jsonl"):
        if file_sha256(input_dir / name) != manifest["sha256"][name]:
            raise ValueError(f"empirical run fingerprint differs: {name}")
    with (input_dir / "worlds.jsonl").open(encoding="utf-8") as worlds, (
        input_dir / "observations.jsonl"
    ).open(encoding="utf-8") as observations:
        for raw, observed in zip(worlds, observations, strict=True):
            world, trace = json.loads(raw), json.loads(observed)
            group = []
            for row in trace["rows"]:
                if world["snapshot"]["id"] != row["case_id"]:
                    raise ValueError("empirical case and trace identifiers differ")
                action = row["selected_intervention"]
                observation = trace["preserve"] if action == "preserve_observed_order" else trace["candidates"][action]
                intervals = _energy_intervals(world, observation["result"]["events"], row)
                group.append((row, intervals))
            yield world, group


def _recompute(input_dir: Path, workbook: Path):
    cases = list(_source(input_dir))
    wanted = {hour for _, group in cases for _, intervals in group for interval in intervals
              for hour, _ in _hour_energy(interval)}
    rates, earliest = _rates(workbook, wanted)
    rows = []
    missing_days = Counter()
    methods = defaultdict(list)
    for world, group in cases:
        meta = world["metadata"]
        required = {hour for _, intervals in group for interval in intervals for hour, _ in _hour_energy(interval)}
        matched = required.issubset(rates)
        if not matched:
            missing_days[meta["observed_route_date"]] += 1
        for row, intervals in group:
            records = []
            kg_by_hour = None
            if matched:
                efficiency = meta["ev_profile"]["charging_efficiency"] * (1 - CAMX_GRID_LOSS)
                for interval in intervals:
                    for hour, battery_wh in _hour_energy(interval):
                        grid_kwh = battery_wh / 1000 / efficiency
                        records.append({
                            "hour_ending_utc": hour.isoformat(), "kind": interval.kind,
                            "allocated_grid_kwh": grid_kwh,
                            "allocated_kg_co2": grid_kwh * rates[hour]["kg_co2_per_kwh"],
                        })
                if abs(sum(item["allocated_grid_kwh"] for item in records) - row["total_grid_generation_kwh"]) > 1e-8:
                    raise ValueError("hourly energy failed to reproduce annual-accounting total")
                kg_by_hour = sum(item["allocated_kg_co2"] for item in records)
            customer = row["customer_pickup_kg_co2_at_assumed_share"]
            result = {
                "case_id": row["case_id"], "route_id": row["route_id"], "method": row["method"],
                "observed_route_date": meta["observed_route_date"],
                "coverage": "complete_matched_case" if matched else "missing_official_hours",
                "missing_utc_hours": sorted(hour.isoformat() for hour in required - rates.keys()),
                "carrier_hourly_kg_co2": kg_by_hour,
                "total_hourly_kg_co2_at_assumed_pickup_share": None if kg_by_hour is None else kg_by_hour + customer,
                "annual_egrid_carrier_kg_co2e": row["carrier_operational_kg_co2e"],
                "annual_egrid_total_kg_co2e_at_assumed_pickup_share": row["operational_kg_co2e_at_assumed_share"],
                "hours": records,
            }
            rows.append(result)
            if matched:
                methods[row["method"]].append(result)
    report = {
        "source": {
            "author": "U.S. Energy Information Administration, Form EIA-930",
            "balancing_authority": "California ISO (CISO); regional proxy for anonymized depots",
            "workbook_url": SOURCE_URL, "methodology_url": METHOD_URL, "workbook_sha256": file_sha256(workbook),
            "column": "CO2 Emissions Intensity for Consumed Electricity",
            "first_nonmissing_utc_hour_ending": None if earliest is None else earliest.isoformat(),
            "early_period_gap": (
                "EIA publishes hourly CO2 beginning July 2018. The California ISO June PDF reports month sums by "
                "hour of day, without dated load intensity values."
            ),
            "june_aggregation_source": JUNE_REPORT,
        },
        "schedule": {
            "alert_time": "estimated UTC from Amazon departure time, directed travel and planned service",
            "onboard": "simulate uniform battery input over saved charge intervals, capped at the energy retired for this case",
            "post_horizon": (
                "assume remaining battery replenishment to cover in-window and fallback route energy starts at "
                f"the matched case horizon plus {POST_HORIZON_OFFSET_MIN} minutes, at the modeled charger rate"
            ),
            "grid_conversion": "original charging efficiency and CAMX loss, EIA load kg CO2/kWh including imports",
            "boundary": (
                "Fixed saved routes, average load intensity, modeled timing and customer pickup. "
                "Published annual eGRID uses generation CO2e, so level comparisons also change estimation basis."
            ),
        },
        "coverage": {
            "source_cases": len(cases),
            "matched_cases": len(cases) - sum(missing_days.values()),
            "matched_routes": len({r["route_id"] for r in rows if r["coverage"] == "complete_matched_case"}),
            "source_arm_rows": len(rows),
            "matched_arm_rows": sum(map(len, methods.values())),
            "uncovered_case_counts_by_route_date": dict(sorted(missing_days.items())),
            "required_unique_hours": len(wanted), "matched_unique_hours": len(rates),
        },
        "by_method_matched_only": {
            method: {"cases": len(group), **{
                f"mean_{key}": fmean(row[key] for row in group)
                for key in ("carrier_hourly_kg_co2", "total_hourly_kg_co2_at_assumed_pickup_share",
                            "annual_egrid_carrier_kg_co2e", "annual_egrid_total_kg_co2e_at_assumed_pickup_share")
            }}
            for method, group in methods.items()
        },
    }
    return {key.isoformat(): rates[key] for key in sorted(rates)}, rows, report


def _json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(input_dir: Path, workbook: Path, output: Path) -> dict:
    hours, rows, report = _recompute(input_dir, workbook)
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "hourly_rates.json", hours)
    _json(output / "summary.json", report)
    with (output / "comparison.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    _json(output / "manifest.json", {
        "schema": 1, "empirical_manifest_sha256": file_sha256(input_dir / "manifest.json"),
        "workbook_sha256": file_sha256(workbook),
        "source_code_sha256": file_sha256(Path(__file__)),
        "sha256": {name: file_sha256(output / name) for name in ARTIFACTS},
    })
    return report


def verify(input_dir: Path, workbook: Path, output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_bytes())
    if (
        manifest["empirical_manifest_sha256"] != file_sha256(input_dir / "manifest.json")
        or manifest["workbook_sha256"] != file_sha256(workbook)
        or any(manifest["sha256"][name] != file_sha256(output / name) for name in ARTIFACTS)
    ):
        raise ValueError("hourly comparison fingerprint mismatch")
    hours, rows, summary = _recompute(input_dir, workbook)
    with (output / "comparison.jsonl").open(encoding="utf-8") as stream:
        if (
            [json.loads(line) for line in stream] != rows
            or json.loads((output / "hourly_rates.json").read_bytes()) != hours
            or json.loads((output / "summary.json").read_bytes()) != summary
        ):
            raise ValueError("hourly comparison or source rows do not reproduce")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        step = commands.add_parser(name)
        step.add_argument("empirical", type=Path)
        step.add_argument("workbook", type=Path)
        step.add_argument("output", type=Path)
    download = commands.add_parser("download")
    download.add_argument("workbook", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "download":
            with urlopen(SOURCE_URL, timeout=60) as response, args.workbook.open("xb") as stream:
                while data := response.read(1024 * 1024):
                    stream.write(data)
            print(args.workbook.resolve(), file_sha256(args.workbook))
            return
        operation = run if args.command == "run" else verify
        receipt = operation(args.empirical, args.workbook, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(args.output.resolve(), json.dumps(receipt["coverage"]))


if __name__ == "__main__":
    main()
