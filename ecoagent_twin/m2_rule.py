"""Deterministic baseline for sequential post-outage routing."""

from .m2_simulator import Decision
from .model import Scenario


class RuleBased:
    """Choose only from the same certified actions offered to the learner.

    The environment offers charging only when it makes a remaining deposit
    feasible. The rule uses it if no deposit is immediately feasible.
    """

    def __init__(self, scenario: Scenario):
        self._deadlines = {parcel.id: parcel.deadline_min for parcel in scenario.parcels}

    def choose(self, decision: Decision) -> int:
        deposits = []
        charges = []
        returns = []
        for index, action in enumerate(decision.actions):
            if action.kind == "deposit":
                deposits.append((
                    self._deadlines[action.parcel_id],
                    action.travel_min,
                    action.completion_min,
                    action.parcel_id,
                    action.locker_id,
                    index,
                ))
            elif action.kind == "charge":
                charges.append((action.completion_min, action.travel_min, index))
            elif action.kind == "return":
                returns.append((action.completion_min, action.travel_min, index))

        if deposits:
            return min(deposits)[-1]
        if charges:
            return min(charges)[-1]
        if returns:
            return min(returns)[-1]
        raise ValueError("decision has no supported feasible action")


def choose(decision: Decision, scenario: Scenario) -> int:
    """Select one action for callers that do not retain a controller instance."""
    return RuleBased(scenario).choose(decision)
