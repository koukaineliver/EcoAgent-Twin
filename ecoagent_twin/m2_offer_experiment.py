"""Run and replay a matched, synthetic opt-in locker-reassignment pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean

from .m1_benchmark import check_world_protocol, iter_snapshots
from .m2_customer_mesa import respond
from .m2_experiment import _rollout
from .m2_offer_allocator import CandidateOffer, allocate_ortools, allocate_rule
from .m2_offers import OfferLedger, OfferOption, UtilityProfile
from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .model import Parcel, Scenario
from .replay import DEFAULT_SCENARIO


DEFAULT_PROTOCOL = DEFAULT_SCENARIO.with_name("m2_offers.json")
ALLOCATORS = {"ortools": allocate_ortools, "rule": allocate_rule}
CONTRASTS = ("no_offer", "zero_credit", "fixed_credit")
TABLE_COLUMNS = (
    "case_id", "condition", "day", "seed", "profile", "arm", "demand", "served",
    "affected_demand", "affected_served", "offers", "acceptances", "reassignments",
    "declines", "expiries", "credit_spent_cents", "recovery_minutes", "traction_wh", "charged_wh",
)
ARTIFACTS = ("protocol.json", "base_scenario.json", "traces.jsonl", "comparison.csv", "summary.json")


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _validate(protocol: dict, base: dict) -> dict[str, UtilityProfile]:
    check_world_protocol(protocol, base)
    for field in ("days", "seeds"):
        values = protocol[field]
        if (not values or any(type(value) is not int or value <= 0 for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{field} must contain unique positive integers")
    for field in ("response_after_min", "offer_expiry_after_min", "fixed_credit_cents", "budget_cents",
                  "charger_rate_wh_per_min", "charger_capacity_wh"):
        if type(protocol[field]) is not int or protocol[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    cutoff = base["horizon_min"] - base["outage"]["time_min"]
    if not protocol["response_after_min"] < protocol["offer_expiry_after_min"] < cutoff:
        raise ValueError("response must precede expiry and expiry must precede the episode horizon")
    if protocol["fixed_credit_cents"] > protocol["budget_cents"]:
        raise ValueError("one fixed credit must fit within the budget")
    fallback = protocol["fallback"]
    if (fallback["mode"] != "deferred_home_delivery"
            or type(fallback["availability_after_alert_min"]) is not int
            or fallback["availability_after_alert_min"] <= cutoff):
        raise ValueError("declare a deferred home-delivery fallback after the common horizon")
    targets = {locker for parcel in base["parcels_in_original_plan_order"] for locker in parcel["alternatives"]}
    if not targets <= protocol["locker_pickup"].keys():
        raise ValueError("alternate lockers are missing synthetic pickup attributes")
    for pickup in [fallback, *protocol["locker_pickup"].values()]:
        for key in ("pickup_min", "pickup_cost_cents"):
            if type(pickup[key]) is not int or pickup[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
    profiles = {name: UtilityProfile(**values) for name, values in protocol["profiles"].items()}
    if protocol["reference_profile"] not in profiles:
        raise ValueError("reference profile is missing")
    return profiles


def _projected_deposit(scenario: Scenario, parcel: Parcel, locker: str, dispatch: int, protocol: dict) -> int | None:
    vehicle = next(vehicle for vehicle in scenario.vehicles if vehicle.id == parcel.vehicle_id)
    depot = scenario.depot
    deadline = min(parcel.deadline_min, scenario.horizon_min - scenario.travel_time[locker, depot])
    required_return = scenario.travel_energy[locker, depot] + vehicle.reserve_wh
    direct_time = dispatch + scenario.travel_time[vehicle.node, locker] + scenario.service_min
    if (direct_time <= deadline
            and vehicle.battery_wh - scenario.travel_energy[vehicle.node, locker] >= required_return):
        return direct_time
    rate = protocol["charger_rate_wh_per_min"]
    if rate <= 0:
        return None
    arrival_energy = vehicle.battery_wh - scenario.travel_energy[vehicle.node, depot]
    cap = protocol["charger_capacity_wh"]
    if cap <= arrival_energy or arrival_energy < vehicle.reserve_wh:
        return None
    charge_minutes = (cap - arrival_energy + rate - 1) // rate
    via_depot = (dispatch + scenario.travel_time[vehicle.node, depot] + charge_minutes
                 + scenario.travel_time[depot, locker] + scenario.service_min)
    if via_depot <= deadline and cap - scenario.travel_energy[depot, locker] >= required_return:
        return via_depot
    return None


def _schedule(scenario: Scenario, protocol: dict, profile: UtilityProfile, allocator: str) -> tuple[OfferOption, ...]:
    dispatch = scenario.outage_min + protocol["offer_expiry_after_min"]
    fallback = protocol["fallback"]
    capacities = {locker_id: locker.capacity for locker_id, locker in scenario.lockers.items() if locker.available}
    options = {}
    candidates = []
    for parcel in scenario.parcels:
        if scenario.lockers[parcel.original_locker].available:
            continue
        for locker_id in parcel.alternatives:
            if not capacities.get(locker_id):
                continue
            ready = _projected_deposit(scenario, parcel, locker_id, dispatch, protocol)
            if ready is None:
                continue
            pickup = protocol["locker_pickup"][locker_id]
            option = OfferOption(
                parcel.id, locker_id,
                pickup["pickup_min"] - fallback["pickup_min"],
                ready - scenario.outage_min - fallback["availability_after_alert_min"],
                pickup["pickup_cost_cents"] - fallback["pickup_cost_cents"],
            )
            options[parcel.id, locker_id] = option
            score = round(1_000_000 * profile.accept_probability(option, protocol["fixed_credit_cents"]))
            candidates.append(CandidateOffer(parcel.id, locker_id, score))
    selected = ALLOCATORS[allocator](
        candidates, capacities, protocol["fixed_credit_cents"], protocol["budget_cents"],
    )
    return tuple(options[candidate.parcel_id, candidate.locker_id] for candidate in selected)


def prepare_offer_episode(scenario: Scenario, protocol: dict, key: dict, profile: UtilityProfile,
                          schedule: tuple[OfferOption, ...], arm: str) -> tuple[Scenario, int, dict, list[dict]]:
    """Resolve customer offers once before evaluating a fleet controller."""
    expiry = scenario.outage_min + protocol["offer_expiry_after_min"]
    ledger = OfferLedger(scenario, budget_cents=protocol["budget_cents"])
    credit_cents = protocol["fixed_credit_cents"] if arm == "fixed_credit" else 0
    if arm != "no_offer":
        for option in schedule:
            if not ledger.issue(option, credit_cents, now_min=scenario.outage_min, expires_min=expiry):
                raise ValueError("offer allocator returned an infeasible offer schedule")
        respond(
            ledger, ((option, credit_cents) for option in schedule), profile,
            scenario=key["scenario"], day=key["day"], seed=key["seed"],
            response_min=scenario.outage_min + protocol["response_after_min"],
        )
    ledger.expire(expiry)
    offers = ledger.result()
    slots = offers["assignments"]
    fallback = [
        {"time_min": expiry, "kind": "fallback_scheduled", "parcel_id": parcel.id,
         "mode": protocol["fallback"]["mode"],
         "available_min": scenario.outage_min + protocol["fallback"]["availability_after_alert_min"]}
        for parcel in scenario.parcels
        if parcel.id not in slots and not scenario.lockers[parcel.original_locker].available
    ]
    routed = replace(scenario, parcels=tuple(
        replace(parcel, original_locker=slots.get(parcel.id, parcel.original_locker), alternatives=())
        for parcel in scenario.parcels
    ))
    return routed, expiry, offers, fallback


def _evaluate(scenario: Scenario, protocol: dict, key: dict, profile: UtilityProfile,
              schedule: tuple[OfferOption, ...], arm: str) -> dict:
    routed, expiry, offers, fallback = prepare_offer_episode(scenario, protocol, key, profile, schedule, arm)
    factory = partial(
        OutageEnv, routed, charger_rate_wh_per_min=protocol["charger_rate_wh_per_min"],
        charger_capacity_wh=protocol["charger_capacity_wh"],
        preassigned_slots=offers["assignments"], dispatch_after_min=expiry,
    )
    replay = _rollout(factory, RuleBased(routed).choose, measure=False)
    return {"offers": offers, "fallbacks": fallback, "fleet_decisions": replay["actions"], "fleet": replay["result"]}


def _table_row(scenario: Scenario, key: dict, name: str, arm: str, outcome: dict) -> dict:
    affected = {parcel.id for parcel in scenario.parcels if not scenario.lockers[parcel.original_locker].available}
    fleet = outcome["fleet"]["metrics"]
    offer = outcome["offers"]["metrics"]
    return {
        "case_id": scenario.id, "condition": key["scenario"], "day": key["day"], "seed": key["seed"],
        "profile": name, "arm": arm,
        "demand": fleet["total_parcels"], "served": fleet["served_parcels"],
        "affected_demand": len(affected),
        "affected_served": len(affected.difference(outcome["fleet"]["unserved_parcel_ids"])),
        "offers": offer["issued"], "acceptances": offer["accepted"], "reassignments": offer["reassignments"],
        "declines": offer["declined"], "expiries": offer["expired"], "credit_spent_cents": offer["spent_cents"],
        "recovery_minutes": fleet["recovery_95_min"], "traction_wh": fleet["travel_wh"],
        "charged_wh": fleet["charged_wh"],
    }


def _cases(protocol: dict, base: dict, allocator: str):
    profiles = _validate(protocol, base)
    for key, raw in iter_snapshots(protocol, base, [("pilot", protocol["days"], protocol["seeds"])]):
        scenario = Scenario.from_dict(raw)
        schedule = _schedule(scenario, protocol, profiles[protocol["reference_profile"]], allocator)
        rows = []
        recorded = {}
        for name, profile in profiles.items():
            outcomes = {}
            for arm in CONTRASTS:
                outcome = _evaluate(scenario, protocol, key, profile, schedule, arm)
                outcomes[arm] = outcome
                rows.append(_table_row(scenario, key, name, arm, outcome))
            recorded[name] = outcomes
        yield {"case": key, "snapshot": raw, "profiles": recorded}, rows


def _summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"], row["arm"]].append(row)
    profiles = {}
    for (profile, arm), group in groups.items():
        offers = sum(row["offers"] for row in group)
        accepted = sum(row["acceptances"] for row in group)
        profiles.setdefault(profile, {})[arm] = {
            "episodes": len(group),
            "mean_service_rate": fmean(row["served"] / row["demand"] for row in group),
            "affected_served": sum(row["affected_served"] for row in group),
            "offers": offers,
            "acceptances": accepted,
            "acceptance_rate": accepted / offers if offers else None,
            "reassignments": sum(row["reassignments"] for row in group),
            "credit_spent_cents": sum(row["credit_spent_cents"] for row in group),
        }
    for group in profiles.values():
        zero, fixed = group["zero_credit"], group["fixed_credit"]
        if fixed["offers"] != zero["offers"]:
            raise ValueError("matched offer schedule changed across the credit arms")
        group["fixed_minus_zero"] = {
            "extra_acceptances": fixed["acceptances"] - zero["acceptances"],
            "extra_affected_deposits": fixed["affected_served"] - zero["affected_served"],
            "service_change_percentage_points": 100 * (fixed["mean_service_rate"] - zero["mean_service_rate"]),
        }
    return {
        "data_basis": "synthetic recipients and post-outage scenarios",
        "matched_snapshots": len({row["case_id"] for row in rows}),
        "evaluation_rows": len(rows),
        "profiles": profiles,
    }


def run(config: Path, base_path: Path, output: Path, allocator: str) -> dict:
    inputs = {"protocol.json": config.read_bytes(), "base_scenario.json": base_path.read_bytes()}
    protocol, base = map(json.loads, inputs.values())
    _validate(protocol, base)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in inputs.items():
        (output / name).write_bytes(data)
    rows = []
    with (output / "traces.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for trace, case_rows in _cases(protocol, base, allocator):
            stream.write(json.dumps(trace, allow_nan=False) + "\n")
            rows.extend(case_rows)
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    _write_json(output / "summary.json", summary)
    sources = ("m2_offer_experiment.py", "m2_offers.py", "m2_customer_mesa.py", "m2_offer_allocator.py",
               "m2_simulator.py", "m2_rule.py", "m2_experiment.py", "model.py", "m1_benchmark.py")
    _write_json(output / "manifest.json", {
        "schema": 1, "protocol_id": protocol["id"], "allocator": allocator,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "mesa_version": version("mesa"),
        "ortools_version": version("ortools") if allocator == "ortools" else None,
        "artifacts_sha256": {name: sha256((output / name).read_bytes()).hexdigest() for name in ARTIFACTS},
        "sources_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
    })
    return summary


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_bytes())
    for name in ARTIFACTS:
        if sha256((output / name).read_bytes()).hexdigest() != manifest["artifacts_sha256"][name]:
            raise ValueError(f"artifact fingerprint mismatch: {name}")
    protocol, base = (json.loads((output / name).read_bytes()) for name in ARTIFACTS[:2])
    if manifest["schema"] != 1 or manifest["protocol_id"] != protocol["id"]:
        raise ValueError("manifest does not identify the saved offer protocol")
    rows = []
    with (output / "traces.jsonl").open(encoding="utf-8") as stream:
        for line, (replay, case_rows) in zip(stream, _cases(protocol, base, manifest["allocator"]), strict=True):
            if json.loads(line) != replay:
                raise ValueError(f"offer or fleet trace differs for {replay['case']}")
            rows.extend(case_rows)
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(TABLE_COLUMNS):
            raise ValueError("unexpected offer comparison columns")
        for saved, replay in zip(reader, rows, strict=True):
            if saved != {name: "" if value is None else str(value) for name, value in replay.items()}:
                raise ValueError("offer comparison row differs from replay")
    summary = _summarize(rows)
    if summary != json.loads((output / "summary.json").read_bytes()):
        raise ValueError("offer summary differs from replay")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    runner.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    runner.add_argument("--allocator", choices=ALLOCATORS, default="ortools")
    runner.add_argument("--output", type=Path)
    commands.add_parser("verify").add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or Path("runs") / f"m2_offers_{stamp}"
            summary = run(args.protocol, args.scenario, output, args.allocator)
            print(f"Saved offer pilot to {output.resolve()}")
        else:
            summary = verify(args.output)
        print(f"{summary['matched_snapshots']} matched snapshots, {summary['evaluation_rows']} evaluation rows")
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
