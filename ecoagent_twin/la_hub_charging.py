"""Replay the Month 6 charging comparison with pinned Los Angeles road costs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .empirical_amazon import file_sha256
from .m3_cases import read_json, write_json
from .m5_experiment import _state
from .m6_empirical_benchmark import _offer_state
from .m6_hub_charging_experiment import (
    METHODS, M5_ALERT_UTC, TRANSITION_LIMIT, _coverage, _lines, _measure, _metadata, _project, _run_source, _summary,
)


def _source_cases(amazon: Path, distinct: Path) -> list:
    if _run_source(amazon, ("worlds.jsonl", "observations.jsonl", "offer_protocol.json", "osmnx_distances.json"))["schema"] != 2:
        raise ValueError("Select the OSMnx Amazon replay")
    manifest = _run_source(distinct, ("traces.jsonl", "offer_protocol.json"))
    if manifest["kind"] != "la_osmnx_frozen_controller_transfer":
        raise ValueError("Select the OSMnx controller replay")
    cases = []
    protocol = read_json(amazon / "offer_protocol.json")
    for world, trace in zip(_lines(amazon / "worlds.jsonl"), _lines(amazon / "observations.jsonl"), strict=True):
        scenario, settings, offers, fallback, alert = _offer_state(world, protocol)
        if (offers, fallback, alert) != (trace["offers"], trace["fallbacks"], trace["alert"]):
            raise ValueError("Amazon offer state did not replay")
        meta = world["metadata"]
        utc = datetime.fromisoformat(meta["projected_alert_utc"]).astimezone(timezone.utc)
        cases.append(_project(scenario, settings, offers, utc - timedelta(minutes=scenario.outage_min),
                              study="amazon", efficiency=meta["ev_profile"]["charging_efficiency"],
                              nominal_rate=settings["charger_rate_wh_per_min"], panel={"route_id": meta["route_id"],
                              "date": meta["observed_route_date"], "EV_profile": meta["ev_profile"]["id"],
                              "SOC": meta["battery_context"]}))
    protocol = read_json(distinct / "offer_protocol.json")
    for trace in _lines(distinct / "traces.jsonl"):
        world = trace["world"]
        if (world["fleet_size"], world["traffic_seed"], world["severity"]) != (2, 4011, "severe"):
            continue
        scenario, settings, offers, fallback, alert = _state(world, protocol)
        if (offers, fallback, alert) != (trace["offers"], trace["fallbacks"], trace["alert"]):
            raise ValueError("University Park offer state did not replay")
        cases.append(_project(scenario, settings, offers, M5_ALERT_UTC - timedelta(minutes=scenario.outage_min),
                              study="m5_stress", efficiency=.90, nominal_rate=240, stress=True,
                              panel={key: trace[key] for key in ("traffic_seed", "traffic", "severity", "chaos_kind")}))
    if Counter(case.study for case in cases)["m5_stress"] != 12:
        raise ValueError("The registered University Park stress panel is incomplete")
    return cases


def experiment(amazon: Path, distinct: Path, workbook: Path, output: Path, *, verify: bool = False) -> dict:
    sources = {"amazon_manifest_sha256": file_sha256(amazon / "manifest.json"),
               "la_manifest_sha256": file_sha256(distinct / "manifest.json"), "eia_workbook_sha256": file_sha256(workbook),
               "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
                   "la_hub_charging.py", "m6_hub_charging_experiment.py", "m6_hub_charging.py", "m6_dynamic_grid.py")}}
    cases, rates, first_hour = _coverage(_source_cases(amazon, distinct), workbook)
    specification = {"selection": "First eight fully covered Amazon routes and seed 4011 University Park severe cases",
                     "forecast": "One-week hourly CO2 persistence with realized CISO CO2 at charge time",
                     "boundary": "Modeled EV use, charging loss and terminal replenishment",
                     "earliest_hour": first_hour.isoformat(), "transition_limit": TRANSITION_LIMIT}
    archived = read_json(output) if verify else None
    if archived is not None and (archived["sources"], archived["specification"], archived["hourly_rates"]) != (
        sources, specification, rates
    ):
        raise ValueError("LA charging inputs differ from the archived run")
    rows = []
    prior_rows = archived["rows"] if archived is not None else [None] * len(cases)
    for index, (case, prior) in enumerate(zip(cases, prior_rows, strict=True), 1):
        rows.append({**_metadata(case), "methods": {name: _measure(case, rates, name, TRANSITION_LIMIT,
                    None if prior is None else prior["methods"][name]) for name in METHODS}})
        if not verify and index % 8 == 0:
            print(f"{index} matched charging cases complete", flush=True)
    summary = _summary(rows)
    if archived is not None:
        if (archived["rows"], archived["summary"]) != (rows, summary):
            raise ValueError("LA charging schedules or summary do not replay")
    else:
        write_json(output, {"schema": "la_hub_charging_v1", "sources": sources, "specification": specification,
                            "hourly_rates": rates, "summary": summary, "rows": rows})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    for field in ("amazon", "distinct", "workbook", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    summary = experiment(args.amazon, args.distinct, args.workbook, args.output, verify=args.command == "verify")
    print({key: value["cases_per_arm"] for key, value in summary.items()})


if __name__ == "__main__":
    main()
