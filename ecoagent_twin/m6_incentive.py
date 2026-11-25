"""Allocate personalized credits and one bounded modeled follow-up after a decline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import isfinite

from .m2_customer_mesa import respond
from .m2_offer_experiment import _projected_deposit
from .m2_offers import OfferLedger, OfferOption, UtilityProfile
from .model import Scenario


ARMS = ("no_offer", "fixed", "personalized", "adaptive")
NEGOTIATION_WINDOW_MIN = 3


@dataclass(frozen=True)
class Quote:
    option: OfferOption
    credit_cents: int
    p_accept: float
    p_at_selection: float


def _options(
    scenario: Scenario, protocol: dict, affected: list[str], dispatch_min: int,
) -> list[OfferOption]:
    fallback = protocol["fallback"]
    result = []
    for parcel in sorted(scenario.parcels, key=lambda item: item.id):
        if parcel.id not in affected:
            continue
        for locker_id in sorted(parcel.alternatives):
            locker = scenario.lockers[locker_id]
            if not locker.available or not locker.capacity:
                continue
            projected = _projected_deposit(scenario, parcel, locker_id, dispatch_min, protocol)
            if projected is None:
                continue
            pickup = protocol["locker_pickup"][locker_id]
            result.append(OfferOption(
                parcel.id, locker_id,
                pickup["pickup_min"] - fallback["pickup_min"],
                projected - scenario.outage_min - fallback["availability_after_alert_min"],
                pickup["pickup_cost_cents"] - fallback["pickup_cost_cents"],
            ))
    return result


def _allocate(candidates: list[Quote], scenario: Scenario, budget: int) -> list[Quote]:
    """Maximize rounded expected accepts under a worst-case committed-credit budget."""
    from ortools.sat.python import cp_model

    quotes = sorted(
        (quote for quote in candidates if quote.credit_cents <= budget and quote.p_at_selection > 0),
        key=lambda quote: (quote.option.parcel_id, quote.option.locker_id, quote.credit_cents),
    )
    if not quotes:
        return []
    if max(budget, sum(quote.credit_cents for quote in quotes)) >= 2**62:
        raise ValueError("Credit coefficients exceed the solver's supported integer range")
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"quote_{index}") for index in range(len(quotes))]
    pairs = list(zip(quotes, variables, strict=True))
    by_parcel, by_locker = defaultdict(list), defaultdict(list)
    for quote, choice in pairs:
        by_parcel[quote.option.parcel_id].append(choice)
        by_locker[quote.option.locker_id].append(choice)
    for group in by_parcel.values():
        model.add_at_most_one(group)
    for locker_id, group in by_locker.items():
        model.add(sum(group) <= scenario.lockers[locker_id].capacity)
    spend = sum(quote.credit_cents * choice for quote, choice in pairs)
    model.add(spend <= budget)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0

    def maximize(expression) -> None:
        model.maximize(expression)
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(f"Incentive allocation failed: {solver.status_name(status)}")
        model.add(expression == solver.value(expression))

    maximize(sum(round(1_000_000 * quote.p_at_selection) * choice for quote, choice in pairs))
    maximize(sum(variables))
    maximize(-spend)
    for start in range(0, len(variables), 30):
        block = variables[start:start + 30]
        maximize(sum(choice * (1 << bit) for choice, bit in zip(
            block, reversed(range(len(block))), strict=True
        )))
    return [quote for quote, choice in pairs if solver.value(choice)]


def _quotes(
    options: list[OfferOption],
    profiles: dict[str, UtilityProfile],
    credits: tuple[int, ...],
    declines: dict[str, Quote] | None = None,
) -> list[Quote]:
    quotes = []
    for option in options:
        for credit in credits:
            unconditional = profiles[option.parcel_id].accept_probability(option, credit)
            if declines is None:
                probability = unconditional
            else:
                previous = declines[option.parcel_id].p_accept
                probability = (unconditional - previous) / (1 - previous) if previous < 1 else 0
            if probability > 0:
                quotes.append(Quote(option, credit, unconditional, probability))
    return quotes


def _resolve_round(
    scenario: Scenario,
    key: dict,
    profiles: dict[str, UtilityProfile],
    quotes: list[Quote],
    budget: int,
    issue_min: int,
    round_index: int,
) -> tuple[dict, dict[str, Quote]]:
    ledger = OfferLedger(scenario, budget_cents=budget)
    for quote in quotes:
        if not ledger.issue(quote.option, quote.credit_cents, issue_min, issue_min + 2):
            raise RuntimeError("Incentive allocation did not pass the common offer ledger")
    for quote in quotes:
        respond(
            ledger, [(quote.option, quote.credit_cents)], profiles[quote.option.parcel_id],
            scenario=key["scenario"], day=key["day"], seed=key["seed"], response_min=issue_min + 1,
        )
    result = ledger.result()
    declined = {event["parcel_id"] for event in result["events"] if event["kind"] == "offer_declined"}
    receipt = {
        **result,
        "events": [{**event, "negotiation_round": round_index} for event in result["events"]],
        "quote_plans": [{
            "parcel_id": quote.option.parcel_id, "locker_id": quote.option.locker_id,
            "credit_cents": quote.credit_cents, "modeled_unconditional_acceptance": quote.p_accept,
            "modeled_acceptance_at_quote_selection": quote.p_at_selection,
        } for quote in quotes],
    }
    return receipt, {quote.option.parcel_id: quote for quote in quotes if quote.option.parcel_id in declined}


def negotiate(
    scenario: Scenario,
    protocol: dict,
    key: dict,
    profiles: dict[str, UtilityProfile],
    *,
    arm: str = "adaptive",
) -> tuple[Scenario, int, dict, list[dict]]:
    """Return common-clock dispatch commitments from modeled recipient responses."""
    if arm not in ARMS:
        raise ValueError(f"Unknown incentive arm: {arm}")
    budget, fixed = protocol["budget_cents"], protocol["fixed_credit_cents"]
    if any(type(value) is not int or value < 0 for value in (budget, fixed)):
        raise ValueError("Credit amounts must be nonnegative integer cents")
    dispatch = scenario.outage_min + NEGOTIATION_WINDOW_MIN
    fallback_protocol = protocol["fallback"]
    if (dispatch >= scenario.horizon_min or fallback_protocol["mode"] != "deferred_home_delivery"
            or scenario.outage_min + fallback_protocol["availability_after_alert_min"] <= scenario.horizon_min):
        raise ValueError("Common dispatch must precede the horizon and deferred fallback must follow it")
    affected = sorted(parcel.id for parcel in scenario.parcels
                      if not scenario.lockers[parcel.original_locker].available)
    if not all(isinstance(profiles.get(parcel), UtilityProfile) for parcel in affected):
        raise ValueError("Every affected parcel needs a declared simulated preference profile")

    base_capacity = {locker_id: locker.capacity for locker_id, locker in scenario.lockers.items()}
    credits = (fixed,) if arm == "fixed" else tuple(sorted({fixed * level // 4 for level in range(4)}))
    options = _options(scenario, protocol, affected, dispatch) if arm != "no_offer" else []
    first_quotes = _allocate(_quotes(options, profiles, credits), scenario, budget)
    first, declines = _resolve_round(scenario, key, profiles, first_quotes, budget, scenario.outage_min, 1)
    rounds = [first]

    if arm == "adaptive" and declines:
        used = Counter(first["assignments"].values())
        residual = replace(scenario, lockers={
            locker_id: replace(locker, capacity=base_capacity[locker_id] - used[locker_id])
            for locker_id, locker in scenario.lockers.items()
        })
        remainder = budget - first["metrics"]["spent_cents"]
        follow_options = _options(residual, protocol, sorted(declines), dispatch)
        follow_credits = tuple(sorted({*credits, fixed}))
        follow_quotes = _allocate(_quotes(follow_options, profiles, follow_credits, declines), residual, remainder)
        follow, _ = _resolve_round(
            residual, key, profiles, follow_quotes, remainder, scenario.outage_min + 1, 2,
        )
        rounds.append(follow)

    assignments = {parcel: locker for record in rounds for parcel, locker in record["assignments"].items()}
    metrics = {
        name: sum(record["metrics"][name] for record in rounds)
        for name in ("spent_cents", "outstanding_cents", "issued", "accepted", "declined", "expired")
    }
    if (metrics["spent_cents"] > budget or metrics["outstanding_cents"]
            or sum(len(record["assignments"]) for record in rounds) != len(assignments)
            or any(count > base_capacity[locker] for locker, count in Counter(assignments.values()).items())):
        raise RuntimeError("Negotiation commitments failed final budget or capacity certification")
    metrics.update({
        "budget_cents": budget,
        "reassignments": len(assignments),
        "followup_issued": rounds[1]["metrics"]["issued"] if len(rounds) == 2 else 0,
        "followup_accepted": rounds[1]["metrics"]["accepted"] if len(rounds) == 2 else 0,
    })
    offers = {
        "arm": arm, "response_basis": "simulated Mesa logit with shared persistent parcel draw",
        "events": [entry for record in rounds for entry in record["events"]],
        "assignments": assignments, "metrics": metrics, "rounds": rounds,
    }
    fallback = [{
        "time_min": dispatch, "kind": "fallback_scheduled", "parcel_id": parcel,
        "mode": fallback_protocol["mode"],
        "available_min": scenario.outage_min + fallback_protocol["availability_after_alert_min"],
    } for parcel in affected if parcel not in assignments]
    routed = replace(scenario, parcels=tuple(
        replace(parcel, original_locker=assignments.get(parcel.id, parcel.original_locker), alternatives=())
        for parcel in scenario.parcels
    ))
    return routed, dispatch, offers, fallback
