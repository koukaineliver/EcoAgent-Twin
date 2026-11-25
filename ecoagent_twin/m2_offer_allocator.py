"""Allocate a fixed set of eligible offers under slot and credit limits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateOffer:
    parcel_id: str
    locker_id: str
    score: int


def _offer_key(offer: CandidateOffer) -> tuple[str, str]:
    return offer.parcel_id, offer.locker_id


def _eligible_offers(
    candidates: Iterable[CandidateOffer],
    capacities: dict[str, int],
    reference_credit_cents: int,
    budget_cents: int,
) -> tuple[tuple[CandidateOffer, ...], int]:
    credits = (reference_credit_cents, budget_cents)
    if any(type(value) is not int or value < 0 for value in credits):
        raise ValueError("Reference credit and budget must be nonnegative integer cents")
    if any(type(capacity) is not int or capacity < 0 for capacity in capacities.values()):
        raise ValueError("Locker capacities must be nonnegative integers")

    by_pair: dict[tuple[str, str], CandidateOffer] = {}
    for offer in candidates:
        identifiers = _offer_key(offer)
        if any(not isinstance(name, str) or not name for name in identifiers):
            raise ValueError("Offers require nonempty parcel and locker IDs")
        if type(offer.score) is not int:
            raise ValueError("Offer scores must be integers")
        if capacities.get(offer.locker_id, 0) == 0:
            continue
        previous = by_pair.get(identifiers)
        if previous is None or offer.score > previous.score:
            by_pair[identifiers] = offer

    eligible = tuple(sorted(by_pair.values(), key=_offer_key))
    offer_limit = len(eligible)
    if reference_credit_cents:
        offer_limit = min(offer_limit, budget_cents // reference_credit_cents)
    return eligible, offer_limit


def allocate_rule(
    candidates: Iterable[CandidateOffer],
    capacities: dict[str, int],
    reference_credit_cents: int,
    budget_cents: int,
) -> tuple[CandidateOffer, ...]:
    """Greedily pick high scores, using parcel and locker IDs to break ties."""
    eligible, limit = _eligible_offers(candidates, capacities, reference_credit_cents, budget_cents)
    remaining = capacities.copy()
    chosen: list[CandidateOffer] = []
    parcels: set[str] = set()
    for offer in sorted(eligible, key=lambda candidate: (-candidate.score, *_offer_key(candidate))):
        if len(chosen) == limit:
            break
        if offer.parcel_id in parcels or remaining[offer.locker_id] == 0:
            continue
        chosen.append(offer)
        parcels.add(offer.parcel_id)
        remaining[offer.locker_id] -= 1
    return tuple(sorted(chosen, key=_offer_key))


def allocate_ortools(
    candidates: Iterable[CandidateOffer],
    capacities: dict[str, int],
    reference_credit_cents: int,
    budget_cents: int,
) -> tuple[CandidateOffer, ...]:
    """Maximize offer count, then total score, then prefer earlier stable IDs."""
    eligible, limit = _eligible_offers(candidates, capacities, reference_credit_cents, budget_cents)
    if limit == 0:
        return ()

    from ortools.sat.python import cp_model

    if sum(abs(offer.score) for offer in eligible) >= 2**62:
        raise ValueError("The combined magnitude of offer scores exceeds the supported range")
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"offer_{index}") for index in range(len(eligible))]
    by_parcel = defaultdict(list)
    by_locker = defaultdict(list)
    for offer, variable in zip(eligible, variables, strict=True):
        by_parcel[offer.parcel_id].append(variable)
        by_locker[offer.locker_id].append(variable)
    for parcel_offers in by_parcel.values():
        model.add_at_most_one(parcel_offers)
    for locker, locker_offers in by_locker.items():
        model.add(sum(locker_offers) <= min(capacities[locker], limit))
    offer_count = sum(variables)
    model.add(offer_count <= limit)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0

    def maximize_and_fix(expression) -> None:
        model.maximize(expression)
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            detail = solver.status_name(status)
            raise RuntimeError(f"Offer allocation did not reach an optimum: {detail}")
        model.add(expression == solver.value(expression))

    maximize_and_fix(offer_count)
    maximize_and_fix(
        sum(offer.score * variable for offer, variable in zip(eligible, variables, strict=True))
    )
    # Fix one binary block at a time to retain exact ordering without overflowing integer scores.
    block_size = 30
    for start in range(0, len(variables), block_size):
        block = variables[start : start + block_size]
        weights = (1 << shift for shift in reversed(range(len(block))))
        maximize_and_fix(
            sum(weight * variable for weight, variable in zip(weights, block, strict=True))
        )

    return tuple(
        offer
        for offer, variable in zip(eligible, variables, strict=True)
        if solver.value(variable)
    )
