"""Matched multi-hub and grid-aware planning on frozen Month 5 and Amazon cases."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import partial
import json
from math import ceil
from pathlib import Path
from statistics import fmean
from time import perf_counter

from .empirical_amazon import file_sha256
from .empirical_energy import CAMX_GRID_LOSS
from .m3_cases import read_json, write_json
from .m5_experiment import _state
from .m6_dynamic_grid import HOUR, METHOD_URL, SOURCE_URL, _rates
from .m6_empirical_benchmark import _offer_state
from .m6_hub_charging import Hub, HubOutageEnv, carbon, energy_ledger, replay, search
from .model import Scenario


METHODS = ("depot_energy", "hub_energy", "hub_carbon")
LAG = timedelta(days=7)
M5_ALERT_UTC = datetime(2018, 7, 16, 18, 45, tzinfo=timezone.utc)
AMAZON_ROUTE_LIMIT = 8
TRANSITION_LIMIT = 12_000


@dataclass(frozen=True)
class Case:
    study: str
    scenario: Scenario
    settings: dict
    origin: datetime
    depot_rate: int
    registered_rate: int
    battery_cap: int
    efficiency: float
    offer_spend: int
    hub: str
    panel: dict

    @property
    def terminal_target(self) -> int:
        return max(self.battery_cap, *(vehicle.battery_wh for vehicle in self.scenario.vehicles))


def _lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _run_source(path: Path, filenames: tuple[str, ...]) -> dict:
    manifest = read_json(path / "manifest.json")
    for name in filenames:
        if file_sha256(path / name) != manifest["sha256"][name]:
            raise ValueError(f"archived input fingerprint mismatch: {name}")
    for name, expected in manifest["source_sha256"].items():
        if file_sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"saved run source fingerprint mismatch: {name}")
    return manifest


def _hub(scenario: Scenario) -> str:
    return min((node for node, site in scenario.lockers.items() if site.available and site.capacity > 0),
               key=lambda node: (sum(scenario.travel_energy[vehicle.node, node] for vehicle in scenario.vehicles), node))


def _project(scenario, settings, offers, origin, *, study, efficiency, nominal_rate, panel, stress=False):
    hub = _hub(scenario)
    original = {vehicle.id: vehicle.battery_wh for vehicle in scenario.vehicles}
    if stress:
        vehicles = tuple(replace(vehicle, battery_wh=min(vehicle.battery_wh, vehicle.reserve_wh + max(
            scenario.travel_energy[vehicle.node, scenario.depot], scenario.travel_energy[vehicle.node, hub]
        ) + 10)) for vehicle in scenario.vehicles)
        scenario = replace(scenario, vehicles=vehicles)
    return Case(study, scenario,
                {key: settings[key] for key in ("preassigned_slots", "dispatch_after_min")},
                origin, settings["charger_rate_wh_per_min"], nominal_rate, settings["charger_capacity_wh"],
                efficiency, offers["metrics"]["spent_cents"], hub,
                {**panel, "archived_initial_soc_wh": original,
                 "replay_initial_soc_wh": {v.id: v.battery_wh for v in scenario.vehicles}})


def _source_cases(amazon: Path, m5: Path):
    paths = {"amazon": amazon, "m5": m5}
    formats = {"amazon": ("worlds.jsonl", "observations.jsonl", "offer_protocol.json"),
               "m5": ("traces.jsonl", "offer_protocol.json")}
    for study, filenames in formats.items():
        _run_source(paths[study], filenames)
    empirical = read_json(amazon / "offer_protocol.json")
    independent = read_json(m5 / "offer_protocol.json")
    cases = []
    for world, trace in zip(_lines(amazon / "worlds.jsonl"), _lines(amazon / "observations.jsonl"), strict=True):
        scenario, settings, offers, fallback, alert = _offer_state(world, empirical)
        if ((offers, fallback, alert) != (trace["offers"], trace["fallbacks"], trace["alert"])
                or {row["case_id"] for row in trace["rows"]} != {scenario.id}):
            raise ValueError("Amazon charging projection differs from archived offer state")
        metadata = world["metadata"]
        utc = datetime.fromisoformat(metadata["projected_alert_utc"]).astimezone(timezone.utc)
        cases.append(_project(scenario, settings, offers, utc - timedelta(minutes=scenario.outage_min),
                              study="amazon", efficiency=metadata["ev_profile"]["charging_efficiency"],
                              nominal_rate=settings["charger_rate_wh_per_min"],
                              panel={"route_id": metadata["route_id"], "date": metadata["observed_route_date"],
                                     "EV_profile": metadata["ev_profile"]["id"], "SOC": metadata["battery_context"]}))
    for trace in _lines(m5 / "traces.jsonl"):
        world = trace["world"]
        if (world["fleet_size"], world["traffic_seed"], world["severity"]) != (2, 301, "severe"):
            continue
        scenario, settings, offers, fallback, alert = _state(world, independent)
        if (offers, fallback, alert) != (trace["offers"], trace["fallbacks"], trace["alert"]):
            raise ValueError("M5 charging stress differs from archived offer state")
        cases.append(_project(scenario, settings, offers, M5_ALERT_UTC - timedelta(minutes=scenario.outage_min),
                              study="m5_stress", efficiency=0.90, nominal_rate=240, stress=True,
                              panel={k: trace[k] for k in ("traffic_seed", "traffic", "severity", "chaos_kind")}))
    if Counter(c.study for c in cases)["m5_stress"] != 12:
        raise ValueError("the fixed two-EV stress panel is incomplete")
    return cases


def _hours(case: Case) -> set[datetime]:
    extra = 15 + len(case.scenario.vehicles) * ceil(case.terminal_target / case.registered_rate)
    start = case.origin + timedelta(minutes=case.scenario.outage_min)
    end = case.origin + timedelta(minutes=case.scenario.horizon_min + extra)
    first = start.replace(minute=0, second=0, microsecond=0) + HOUR
    return {first + i * HOUR for i in range(ceil((end - first) / HOUR) + 1)}


def _coverage(cases, workbook):
    needed = set().union(*(_hours(case) for case in cases))
    found, earliest = _rates(workbook, needed | {hour - LAG for hour in needed})
    eligible = [case for case in cases if all(hour in found and hour - LAG in found for hour in _hours(case))]
    counts = Counter(case.panel["route_id"] for case in eligible if case.study == "amazon")
    routes = dict.fromkeys(case.panel["route_id"] for case in sorted(eligible, key=lambda item: (
        item.panel.get("date", ""), item.panel.get("route_id", "")))
                           if case.study == "amazon" and counts[case.panel["route_id"]] == 4)
    selected = set(list(routes)[:AMAZON_ROUTE_LIMIT])
    included = [case for case in eligible if case.study != "amazon" or case.panel["route_id"] in selected]
    if len(selected) != AMAZON_ROUTE_LIMIT or sum(case.study == "m5_stress" for case in included) != 12:
        raise ValueError("insufficient complete lagged-and-realized EIA hours")
    used = set().union(*(_hours(case) for case in included))
    matched = {hour.isoformat(): found[hour] for hour in sorted(used | {hour - LAG for hour in used})}
    return included, matched, earliest


def _settings(case, method):
    hubs = (Hub(case.scenario.depot, case.depot_rate),) if case.depot_rate > 0 else ()
    if method != "depot_energy":
        hubs += (Hub(case.hub, case.registered_rate),)
    return {**case.settings, "max_battery_wh": case.battery_cap, "hubs": hubs}


def _measure(case, rates, method, transition_limit, archived=None):
    factory = partial(HubOutageEnv, case.scenario, **_settings(case, method))
    ledger = partial(energy_ledger, scenario=case.scenario, terminal_target_wh=case.terminal_target,
                     terminal_rate_wh_per_min=case.registered_rate)
    realized = {hour: rates[hour.isoformat()]["kg_co2_per_kwh"] for hour in _hours(case)}
    forecast = {hour: rates[(hour - LAG).isoformat()]["kg_co2_per_kwh"] for hour in _hours(case)}
    factor = case.efficiency * (1 - CAMX_GRID_LOSS)

    def objective(result):
        m = result["metrics"]
        ranking = (m["travel_wh"], m["travel_min"])
        if method == "hub_carbon":
            ranking = (round(1_000_000 * carbon(ledger(result), case.origin, forecast, factor)), *ranking)
        return -m["served_parcels"], *ranking

    if archived is None:
        started = perf_counter()
        plan = search(factory, objective, transition_limit=transition_limit)
        elapsed_ms = (perf_counter() - started) * 1000
    else:
        plan, elapsed_ms = archived["plan"], archived["planning_ms"]
        if (replay(factory, plan["actions"]) != plan["result"]
                or list(objective(plan["result"])) != plan["objective"]):
            raise ValueError("route or planning objective cannot be replayed")
        rebuilt = search(factory, objective, transition_limit=transition_limit)
        if plan != rebuilt:
            raise ValueError("archived deterministic search does not reproduce")
    intervals = ledger(plan["result"])
    events = plan["result"]["events"]
    return {
        "plan": plan, "planning_ms": elapsed_ms, "charging_intervals": intervals,
        "predicted_kg_co2": carbon(intervals, case.origin, forecast, factor),
        "realized_hourly_kg_co2": carbon(intervals, case.origin, realized, factor),
        "queue_wait_min": sum(event["queue_wait_min"] for event in events if event["kind"] == "charge_reserved"),
        "hub_sessions": sum(row["hub_id"] != case.scenario.depot for row in intervals),
        "hubs": [asdict(hub) for hub in _settings(case, method)["hubs"]],
    }


def _metadata(case):
    return {"case_id": case.scenario.id, "study": case.study, "panel": case.panel,
            "alert_utc": (case.origin + timedelta(minutes=case.scenario.outage_min)).isoformat(),
            "origin_utc": case.origin.isoformat(), "horizon_min": case.scenario.horizon_min,
            "same_preassigned_slots": case.settings["preassigned_slots"], "credit_spend_cents": case.offer_spend,
            "terminal_target_per_vehicle_wh": case.terminal_target, "terminal_depot_rate_wh_per_min": case.registered_rate,
            "modeled_available_locker_hub": case.hub}


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["study"]].append(row)
    result = {}
    for study, group in groups.items():
        method_stats = {}
        for method in METHODS:
            arms = [row["methods"][method] for row in group]
            metrics = [arm["plan"]["result"]["metrics"] for arm in arms]
            method_stats[method] = {
                "mean_service": fmean(m["service_rate"] for m in metrics),
                "parcels_served": sum(m["served_parcels"] for m in metrics),
                "mean_traction_wh": fmean(m["travel_wh"] for m in metrics),
                "mean_predicted_kg_co2": fmean(a["predicted_kg_co2"] for a in arms),
                "mean_realized_rate_kg_co2": fmean(a["realized_hourly_kg_co2"] for a in arms),
                "cases_with_hub_charging": sum(a["hub_sessions"] > 0 for a in arms),
                "cases_with_queue": sum(a["queue_wait_min"] > 0 for a in arms),
                "total_queue_wait_min": sum(a["queue_wait_min"] for a in arms),
                "exact_action_tree_cases": sum(a["plan"]["exact_in_simulator"] for a in arms),
                "mean_planning_ms": fmean(a["planning_ms"] for a in arms),
            }
        result[study] = {"cases_per_arm": len(group), "by_method": method_stats,
                         "different_hub_vs_depot_schedules": sum(row["methods"]["hub_energy"]["plan"]["actions"] !=
                                                                 row["methods"]["depot_energy"]["plan"]["actions"] for row in group),
                         "different_carbon_vs_energy_schedules": sum(row["methods"]["hub_energy"]["plan"]["actions"] !=
                                                                     row["methods"]["hub_carbon"]["plan"]["actions"] for row in group)}
    return result


def experiment(amazon, m5, workbook, output, *, verify=False):
    sources = {"amazon_manifest_sha256": file_sha256(amazon / "manifest.json"),
               "m5_manifest_sha256": file_sha256(m5 / "manifest.json"), "eia_workbook_sha256": file_sha256(workbook),
               "code_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
                   "m6_hub_charging.py", "m6_hub_charging_experiment.py", "m6_dynamic_grid.py", "m2_simulator.py")}}
    cases, rates, first_hour = _coverage(_source_cases(amazon, m5), workbook)
    specification = {"forecast_source": "same UTC hour seven days earlier (retrospective persistence proxy)",
                     "realized_source": SOURCE_URL, "intensity_method": METHOD_URL,
                     "earliest_official_hour_end_utc": first_hour.isoformat(),
                     "selection_rule": "first eight chronologically covered whole Amazon routes; M5 seed 301 severe, two EVs, all chaos and traffic classes",
                     "charger_scope": "one modeled plug at the eligible locker closest by summed directed Wh from initial vehicle nodes, plus archived depot status",
                     "forecast_boundary": "maximize on-time service, then minimize predicted electricity CO2 in hub_carbon, then traction Wh and travel minute; compare under measured held-out regional rates",
                     "battery_boundary": "onboard traction and fixed terminal replenishment, charging and grid losses; excludes downstream fallback delivery, pickup and stationary auxiliary loads",
                     "terminal_scope": "serial restored depot plug from horizon + 15 minutes in every arm; equalized battery target",
                     "transition_limit": TRANSITION_LIMIT}
    previous = read_json(output) if verify else None
    if verify and (previous["sources"], previous["specification"], previous["hourly_rates"]) != (sources, specification, rates):
        raise ValueError("charging source, design or hourly fingerprints differ")
    rows = []
    old = previous["rows"] if verify else [None] * len(cases)
    for prior, case in zip(old, cases, strict=True):
        rows.append({**_metadata(case), "methods": {
            method: _measure(case, rates, method, TRANSITION_LIMIT, None if prior is None else prior["methods"][method])
            for method in METHODS
        }})
    summary = _summary(rows)
    if verify:
        if previous["rows"] != rows or previous["summary"] != summary:
            raise ValueError("saved charging receipts or aggregates cannot be verified")
    else:
        write_json(output, {"schema": "m6_hub_charging_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                            "sources": sources, "specification": specification, "hourly_rates": rates,
                            "summary": summary, "rows": rows})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    for name in ("amazon", "m5", "workbook", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    summary = experiment(args.amazon, args.m5, args.workbook, args.output, verify=args.command == "verify")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
