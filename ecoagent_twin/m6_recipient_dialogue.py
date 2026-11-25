"""Collect explicit recipient replies to personalized locker offers on the event clock."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

from .m2_experiment import _rollout
from .m2_offer_experiment import DEFAULT_PROTOCOL
from .m2_offers import OfferLedger, UtilityProfile
from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .m6_incentive import NEGOTIATION_WINDOW_MIN, Quote, _allocate, _options, _quotes
from .model import Scenario
from .replay import DEFAULT_SCENARIO


@dataclass(frozen=True)
class Reply:
    parcel_id: str
    accepted: bool
    time_min: int


Respond = Callable[[tuple[dict, ...]], Iterable[Reply]]


def _round(scenario: Scenario, quotes: list[Quote], budget: int, start: int, end: int,
           respond: Respond) -> dict:
    ledger = OfferLedger(scenario, budget_cents=budget)
    cards = []
    for quote in quotes:
        if not ledger.issue(quote.option, quote.credit_cents, start, end):
            raise RuntimeError("Projected offer failed the budget, slot, or eligibility check")
        cards.append({**asdict(quote.option), "credit_cents": quote.credit_cents,
                      "issued_min": start, "expires_min": end})

    replies = list(respond(tuple(card.copy() for card in cards))) if cards else []
    previous = start
    for reply in replies:
        if (not isinstance(reply, Reply) or type(reply.accepted) is not bool
                or type(reply.time_min) is not int or not previous <= reply.time_min <= end):
            raise ValueError("Reply timestamps must be ordered within their offer round")
        ledger.resolve(reply.parcel_id, now_min=reply.time_min, accepted=reply.accepted)
        previous = reply.time_min
    ledger.expire(end)
    return {**ledger.result(), "cards": cards, "replies": [asdict(reply) for reply in replies]}


def negotiate_with_recipients(
    scenario: Scenario, protocol: dict, profiles: dict[str, UtilityProfile], respond: Respond,
) -> tuple[Scenario, dict]:
    """Apply explicit replies and offer one bounded revision after a timely decline.

    The host supplies arrival minutes on the trusted episode clock. This function
    is a synchronous integration point for a local UI or an event-driven host.
    Offer issuance reserves slots and credits before any reply enters the ledger.
    """
    budget, fixed = protocol["budget_cents"], protocol["fixed_credit_cents"]
    if any(type(amount) is not int or amount < 0 for amount in (budget, fixed)):
        raise ValueError("Offer budget and credits must be nonnegative integer cents")
    start = scenario.outage_min
    dispatch = start + NEGOTIATION_WINDOW_MIN
    revise = dispatch - 1
    fallback = protocol["fallback"]
    deferred = start + fallback["availability_after_alert_min"]
    if not dispatch < scenario.horizon_min < deferred or fallback["mode"] != "deferred_home_delivery":
        raise ValueError("Departure and deferred delivery must straddle the scenario horizon")
    affected = sorted(parcel.id for parcel in scenario.parcels
                      if not scenario.lockers[parcel.original_locker].available)
    if any(not isinstance(profiles.get(parcel_id), UtilityProfile) for parcel_id in affected):
        raise ValueError("An affected parcel is missing its declared incentive preferences")

    credits = tuple(sorted({fixed * level // 4 for level in range(4)}))
    first_quotes = _allocate(_quotes(_options(scenario, protocol, affected, dispatch), profiles, credits),
                             scenario, budget)
    first = _round(scenario, first_quotes, budget, start, revise, respond)
    rounds = [first]
    declined = {event["parcel_id"] for event in first["events"] if event["kind"] == "offer_declined"}
    if declined:
        accepted = Counter(first["assignments"].values())
        residual = replace(scenario, lockers={
            key: replace(locker, capacity=locker.capacity - accepted[key])
            for key, locker in scenario.lockers.items()
        })
        remainder = budget - first["metrics"]["spent_cents"]
        follow = _options(residual, protocol, sorted(declined), dispatch)
        previous = {quote.option.parcel_id: quote for quote in first_quotes if quote.option.parcel_id in declined}
        quotes = _allocate(_quotes(follow, profiles, (*credits, fixed), previous), residual, remainder)
        rounds.append(_round(residual, quotes, remainder, revise, dispatch, respond))

    assignments = {parcel: locker for record in rounds for parcel, locker in record["assignments"].items()}
    spent = sum(record["metrics"]["spent_cents"] for record in rounds)
    routed = replace(scenario, parcels=tuple(
        replace(parcel, original_locker=assignments.get(parcel.id, parcel.original_locker), alternatives=())
        for parcel in scenario.parcels
    ))
    return routed, {
        "dispatch_min": dispatch, "budget_cents": budget, "spent_cents": spent,
        "assignments": assignments, "rounds": rounds,
        "fallback": [{"parcel_id": parcel, "mode": fallback["mode"], "available_min": deferred}
                     for parcel in affected if parcel not in assignments],
    }


def _keyboard(cards: tuple[dict, ...]) -> Iterable[Reply]:
    for card in cards:
        parcel = card["parcel_id"]
        print(f"{parcel}: locker {card['locker_id']} with a {card['credit_cents']}-cent credit. "
              f"Compared with deferred delivery, pickup time changes by {card['extra_pickup_min']:+g} min "
              f"and estimated availability changes by {card['extra_delay_min']:+g} min.")
        answer = input("Accept the alternative? [y/n/Enter to skip] ").strip().lower()
        while answer not in {"y", "n", ""}:
            answer = input("Enter y, n, or return to skip: ").strip().lower()
        if answer:
            yield Reply(parcel, answer == "y", card["expires_min"] - 1)


def _demonstration(scenario_path: Path, protocol_path: Path, respond: Respond) -> dict:
    raw, config = scenario_path.read_bytes(), protocol_path.read_bytes()
    scenario = Scenario.from_dict(json.loads(raw))
    protocol = json.loads(config)
    profile = UtilityProfile(**protocol["profiles"][protocol["reference_profile"]])
    routed, receipt = negotiate_with_recipients(scenario, protocol, {p.id: profile for p in scenario.parcels}, respond)
    environment = partial(OutageEnv, routed, charger_rate_wh_per_min=protocol["charger_rate_wh_per_min"],
                          charger_capacity_wh=protocol["charger_capacity_wh"], preassigned_slots=receipt["assignments"],
                          dispatch_after_min=receipt["dispatch_min"])
    result = _rollout(environment, RuleBased(routed).choose, measure=False)
    return {"scenario_sha256": sha256(raw).hexdigest(), "protocol_sha256": sha256(config).hexdigest(),
            "scope": "operator replies to fictional offers on a fixed simulator clock", **receipt,
            "route": {"actions": result["actions"], "physical_result": result["result"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "verify"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            if args.receipt.exists():
                raise FileExistsError(args.receipt)
            print("Demonstration with fictional parcels. Each reply uses the offer round's simulated clock.")
            payload = _demonstration(args.scenario, args.protocol, _keyboard)
            with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            archived = json.loads(args.receipt.read_bytes())
            nonempty_rounds = iter(record for record in archived["rounds"] if record["cards"])
            payload = _demonstration(args.scenario, args.protocol, lambda cards: (
                Reply(**reply) for reply in next(nonempty_rounds)["replies"]
            ))
            if payload != archived:
                raise ValueError("Saved offers or route differ from replay")
        print(f"{args.receipt.resolve()} · {len(payload['assignments'])} commitments · {payload['spent_cents']} simulated cents")
    except (EOFError, OSError, RuntimeError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
