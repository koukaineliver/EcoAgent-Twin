"""Synthetic incremental utility and a budgeted alternate-locker offer ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .model import Scenario


@dataclass(frozen=True)
class OfferOption:
    """Signed alternate-minus-deferred-fallback attributes for one parcel."""

    parcel_id: str
    locker_id: str
    extra_pickup_min: float
    extra_delay_min: float
    extra_cost_cents: int

    def __post_init__(self) -> None:
        if not self.parcel_id or not self.locker_id:
            raise ValueError("parcel and locker IDs must be nonempty")
        if not all(math.isfinite(value) for value in (self.extra_pickup_min, self.extra_delay_min)):
            raise ValueError("pickup and delay differences must be finite")
        if type(self.extra_cost_cents) is not int:
            raise ValueError("cost difference must be integer cents")


@dataclass(frozen=True)
class UtilityProfile:
    """Assumed logit coefficients per currency unit and per minute."""

    intercept: float
    money_sensitivity: float
    pickup_minute_sensitivity: float
    delay_minute_sensitivity: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.intercept):
            raise ValueError("utility intercept must be finite")
        sensitivities = (
            self.money_sensitivity, self.pickup_minute_sensitivity, self.delay_minute_sensitivity,
        )
        if not all(math.isfinite(value) and value >= 0 for value in sensitivities):
            raise ValueError("utility sensitivities must be finite and nonnegative")

    def accept_probability(self, option: OfferOption, credit_cents: int) -> float:
        """Compute acceptance for the offered option relative to its fallback."""
        _require_cents(credit_cents)
        net_money = (credit_cents - option.extra_cost_cents) / 100
        utility = (
            self.intercept
            + self.money_sensitivity * net_money
            - self.pickup_minute_sensitivity * option.extra_pickup_min
            - self.delay_minute_sensitivity * option.extra_delay_min
        )
        if math.isnan(utility):
            raise ValueError("utility terms exceed the supported numeric range")
        if utility >= 0:
            return 1 / (1 + math.exp(-utility))
        odds = math.exp(utility)
        return odds / (1 + odds)


def paired_draw(day: int | str, seed: int, parcel_id: str, *, scenario: str = "") -> float:
    """Use the same latent response draw for both arms of a matched parcel."""
    key = json.dumps([scenario, str(day), seed, parcel_id], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], byteorder="big") >> 11) / 2**53


def _require_cents(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("money amounts must be nonnegative integer cents")


@dataclass
class _Offer:
    option: OfferOption
    credit_cents: int
    expires_min: int
    status: str = "pending"
    response_accepted: bool = False


class OfferLedger:
    """Reserve credit and one slot per offer until a terminal response.

    A valid acceptance pays the credit and commits the slot. All other terminal
    outcomes release both holds. The ledger certifies proposed locker
    assignments against its scenario and its own reservations.
    """

    def __init__(self, scenario: Scenario, *, budget_cents: int) -> None:
        _require_cents(budget_cents)
        self._scenario = scenario
        self._budget_cents = budget_cents
        self._now = scenario.outage_min
        self._affected = {
            parcel.id: parcel for parcel in scenario.parcels
            if not scenario.lockers[parcel.original_locker].available
        }
        self._offers: dict[str, _Offer] = {}
        self._events: list[dict] = []

    def _log(self, kind: str, parcel_id: str, **fields) -> None:
        self._events.append({"time_min": self._now, "kind": kind, "parcel_id": parcel_id, **fields})

    def _locker_denial(self, option: OfferOption, *, exclude_own_hold: bool = False) -> str | None:
        parcel = self._affected.get(option.parcel_id)
        if parcel is None:
            return "original_locker_was_available_or_parcel_unknown"
        if option.locker_id not in parcel.alternatives:
            return "locker_ineligible"
        locker = self._scenario.lockers.get(option.locker_id)
        if locker is None or not locker.available:
            return "locker_unavailable"
        used = sum(
            offer.option.locker_id == option.locker_id
            and offer.status in ("pending", "accepted")
            and not (exclude_own_hold and parcel_id == option.parcel_id)
            for parcel_id, offer in self._offers.items()
        )
        return "locker_full" if used >= locker.capacity else None

    def _credit_total(self, *statuses: str) -> int:
        return sum(offer.credit_cents for offer in self._offers.values() if offer.status in statuses)

    def _issue_denial(self, option: OfferOption, credit_cents: int) -> str | None:
        if option.parcel_id in self._offers:
            return "offer_already_issued"
        reason = self._locker_denial(option)
        if reason is not None:
            return reason
        if self._credit_total("pending", "accepted") + credit_cents > self._budget_cents:
            return "budget_exceeded"
        return None

    def can_issue(self, option: OfferOption, credit_cents: int) -> bool:
        """Check eligibility and resources at the ledger's current time."""
        _require_cents(credit_cents)
        return self._issue_denial(option, credit_cents) is None

    def issue(self, option: OfferOption, credit_cents: int, now_min: int, expires_min: int) -> bool:
        """Hold budget and slot, allowing responses in [now_min, expires_min)."""
        _require_cents(credit_cents)
        if type(expires_min) is not int:
            raise ValueError("expiry must use integer minutes")
        self.expire(now_min)
        reason = (
            "expiry_not_after_issue" if expires_min <= self._now
            else self._issue_denial(option, credit_cents)
        )
        if reason is not None:
            self._log("offer_denied", option.parcel_id, locker_id=option.locker_id, reason=reason)
            return False
        self._offers[option.parcel_id] = _Offer(option, credit_cents, expires_min)
        fields = asdict(option)
        fields.pop("parcel_id")
        self._log("offer_issued", option.parcel_id, **fields, credit_cents=credit_cents,
                  expires_min=expires_min)
        return True

    def expire(self, now_min: int) -> int:
        """Release each unanswered offer whose exclusive expiry has arrived."""
        if type(now_min) is not int or now_min < self._now:
            raise ValueError("offer times must be nondecreasing integer minutes")
        overdue = sorted(
            (offer.expires_min, parcel_id)
            for parcel_id, offer in self._offers.items()
            if offer.status == "pending" and offer.expires_min <= now_min
        )
        for expires_min, parcel_id in overdue:
            self._now = expires_min
            offer = self._offers[parcel_id]
            offer.status = "expired"
            self._log("offer_expired", parcel_id, locker_id=offer.option.locker_id,
                      expires_min=expires_min, released_cents=offer.credit_cents)
        self._now = now_min
        return len(overdue)

    def resolve(self, parcel_id: str, *, now_min: int, accepted: bool) -> str:
        """Resolve a response, rechecking the locker before paying and committing."""
        if type(accepted) is not bool:
            raise ValueError("accepted must be boolean")
        self.expire(now_min)
        offer = self._offers.get(parcel_id)
        if offer is None:
            self._log("offer_response_denied", parcel_id, reason="offer_unknown")
            return "denied"
        if offer.status != "pending":
            late = offer.status == "expired"
            self._log("offer_response_denied", parcel_id, locker_id=offer.option.locker_id,
                      reason="after_expiry" if late else "offer_already_resolved",
                      response_accepted=accepted)
            return "expired" if late else "denied"

        if not accepted:
            offer.status = "declined"
            self._log("offer_declined", parcel_id, locker_id=offer.option.locker_id,
                      released_cents=offer.credit_cents)
        else:
            offer.response_accepted = True
            reason = self._locker_denial(offer.option, exclude_own_hold=True)
            if reason is None:
                offer.status = "accepted"
                self._log("offer_accepted", parcel_id, locker_id=offer.option.locker_id,
                          paid_cents=offer.credit_cents)
            else:
                offer.status = "denied"
                self._log("offer_response_denied", parcel_id, locker_id=offer.option.locker_id,
                          reason=reason, response_accepted=True, released_cents=offer.credit_cents)
        return offer.status

    def result(self) -> dict:
        assignments = {
            parcel_id: offer.option.locker_id for parcel_id, offer in self._offers.items()
            if offer.status == "accepted"
        }
        return {
            "events": [event.copy() for event in self._events],
            "assignments": assignments,
            "metrics": {
                "budget_cents": self._budget_cents,
                "spent_cents": self._credit_total("accepted"),
                "outstanding_cents": self._credit_total("pending"),
                "issued": len(self._offers),
                "accepted": sum(offer.response_accepted for offer in self._offers.values()),
                "declined": sum(offer.status == "declined" for offer in self._offers.values()),
                "expired": sum(offer.status == "expired" for offer in self._offers.values()),
                "reassignments": len(assignments),
            },
        }
