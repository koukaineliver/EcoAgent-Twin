"""Shared controller comparison and delivery-event scoring."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

from . import controllers
from .model import Plan, RouteOutcome, Scenario, Stop, evaluate_plan


DEFAULT_SCENARIO = Path(__file__).with_name("scenarios") / "locker_outage.json"
METHODS = {
    "keep_plan": controllers.keep_plan,
    "rule_based": controllers.rule_based,
    "exact_replan": controllers.exact_replan,
}


def simulate(scenario: Scenario, plan: Plan, outcomes: dict[str, RouteOutcome]) -> dict:
    events = [{
        "time_min": scenario.outage_min,
        "kind": "locker_outage",
        "locker_id": scenario.outage_locker,
    }]
    for vehicle in scenario.vehicles:
        outcome = outcomes[vehicle.id]
        for stop, completion in zip(outcome.stops, outcome.deposits_min, strict=True):
            events.append({
                "time_min": completion,
                "kind": "parcel_deposited",
                "vehicle_id": vehicle.id,
                **asdict(stop),
            })
        events.append({
            "time_min": outcome.return_min,
            "kind": "vehicle_returned",
            "vehicle_id": vehicle.id,
            "remaining_wh": outcome.remaining_wh,
        })
    events.sort(key=lambda event: event["time_min"])
    deposits = [event for event in events if event["kind"] == "parcel_deposited"]
    total = len(scenario.parcels)
    served = {event["parcel_id"] for event in deposits}
    threshold = (95 * total + 99) // 100
    return {
        "routes": {vehicle_id: [asdict(stop) for stop in stops] for vehicle_id, stops in plan.items()},
        "events": events,
        "unserved_parcel_ids": sorted({parcel.id for parcel in scenario.parcels} - served),
        "metrics": {
            "total_parcels": total,
            "served_parcels": len(served),
            "unserved_parcels": total - len(served),
            "service_rate": len(served) / total,
            "recovery_95_min": deposits[threshold - 1]["time_min"] - scenario.outage_min
            if len(deposits) >= threshold else None,
            "travel_min": sum(outcome.travel_min for outcome in outcomes.values()),
            "travel_wh": sum(outcome.travel_wh for outcome in outcomes.values()),
        },
    }


def compare(scenario: Scenario) -> dict[str, tuple[dict, float]]:
    decisions = {}
    for method, choose in METHODS.items():
        started = perf_counter_ns()
        plan = choose(scenario)
        outcomes = evaluate_plan(scenario, plan)
        latency_ms = (perf_counter_ns() - started) / 1_000_000
        decisions[method] = simulate(scenario, plan, outcomes), latency_ms
    return decisions


def replay_saved(scenario: Scenario, record: dict) -> dict:
    plan = {
        vehicle_id: tuple(Stop(**stop) for stop in stops)
        for vehicle_id, stops in record["routes"].items()
    }
    return simulate(scenario, plan, evaluate_plan(scenario, plan))
