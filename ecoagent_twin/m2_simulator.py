"""Shared event-driven simulator for the synthetic post-outage decision task."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq

from .model import Parcel, Scenario


ACTOR_FEATURE_DIM = 10
CRITIC_FEATURE_DIM = 16
ACTION_FEATURE_DIM = 10


@dataclass(frozen=True)
class Action:
    kind: str
    parcel_id: str | None
    locker_id: str | None
    travel_min: int
    completion_min: int
    energy_wh: int


@dataclass(frozen=True)
class Decision:
    vehicle_id: str
    time_min: int
    actions: tuple[Action, ...]
    actor_features: tuple[float, ...]
    critic_features: tuple[float, ...]
    action_features: tuple[tuple[float, ...], ...]


@dataclass
class _VehicleState:
    node: str
    battery_wh: int
    reserve_wh: int
    returned: bool = False


@dataclass(frozen=True)
class _Commitment:
    action: Action
    charge_wh: int = 0


class OutageEnv:
    """Advance one certified dispatch decision at a time on a common clock.

    A locker slot is reserved at commitment and occupied at deposit. Vehicles
    act in ID order when ready on the same tick. Pending events on that tick
    settle before a new observation, with completions before plug starts.
    The depot plug takes serial reservations in commitment order. Charging
    replenishes to its cap. An accepted offer holds a slot for its parcel
    before routing begins. The recovery clock still starts at the outage.
    """

    def __init__(
        self, scenario: Scenario, *, charger_rate_wh_per_min: int, charger_capacity_wh: int,
        preassigned_slots: dict[str, str] | None = None,
        dispatch_after_min: int | None = None,
    ) -> None:
        if (
            type(charger_rate_wh_per_min) is not int
            or type(charger_capacity_wh) is not int
            or charger_rate_wh_per_min < 0
            or charger_capacity_wh < 0
            or (charger_rate_wh_per_min > 0 and charger_capacity_wh == 0)
        ):
            raise ValueError("charger rate and capacity must be compatible nonnegative integers")
        dispatch_min = scenario.outage_min
        if dispatch_after_min is not None:
            if (
                type(dispatch_after_min) is not int
                or not scenario.outage_min <= dispatch_after_min <= scenario.horizon_min
            ):
                raise ValueError("dispatch clock must be an integer between the outage and horizon")
            dispatch_min = dispatch_after_min
            for vehicle in scenario.vehicles:
                return_min = dispatch_min + scenario.travel_time[vehicle.node, scenario.depot]
                if return_min > scenario.horizon_min:
                    raise ValueError(f"dispatch clock leaves no time for {vehicle.id} to return")

        self.scenario = scenario
        self._charger_rate = charger_rate_wh_per_min
        self._charger_capacity = charger_capacity_wh
        self._now = scenario.outage_min
        self._charger_free_min = dispatch_min
        self._vehicles = {
            v.id: _VehicleState(v.node, v.battery_wh, v.reserve_wh)
            for v in sorted(scenario.vehicles, key=lambda v: v.id)
        }
        self._parcels = {p.id: p for p in scenario.parcels}
        ordered_parcels = sorted(scenario.parcels, key=lambda p: p.id)
        self._custody = {
            vehicle_id: tuple(parcel for parcel in ordered_parcels if parcel.vehicle_id == vehicle_id)
            for vehicle_id in self._vehicles
        }
        self._open = set(self._parcels)
        self._deposit_times: list[int] = []
        self._preassigned_slots = self._validate_preassigned_slots(preassigned_slots)
        self._reserved: Counter[str] = Counter(self._preassigned_slots.values())
        self._occupied: Counter[str] = Counter()
        self._pending: list[tuple[int, int, str, str]] = []
        self._commitments: dict[str, _Commitment] = {}
        self._events: list[dict] = []
        self._travel_min = self._travel_wh = self._charged_wh = 0
        self._time_scale = scenario.horizon_min - scenario.outage_min
        self._energy_scale = max(1, charger_capacity_wh, *(v.battery_wh for v in scenario.vehicles))
        self._slot_scale = max(1, sum(l.capacity for l in scenario.lockers.values() if l.available))
        self._log("locker_outage", locker_id=scenario.outage_locker)
        self._now = dispatch_min
        if preassigned_slots is not None or dispatch_after_min is not None:
            for parcel_id, locker_id in sorted(self._preassigned_slots.items()):
                self._log("offer_slot_preassigned", parcel_id=parcel_id, locker_id=locker_id,
                          vehicle_id=self._parcels[parcel_id].vehicle_id)
            self._log("routing_started")
        self._decision = self._advance()

    def _validate_preassigned_slots(self, slots: dict[str, str] | None) -> dict[str, str]:
        if slots is None:
            return {}
        if not isinstance(slots, dict):
            raise ValueError("preassigned slots must map parcel IDs to locker IDs")
        reserved: Counter[str] = Counter()
        bindings = dict(slots)
        for parcel_id, locker_id in bindings.items():
            if not isinstance(parcel_id, str) or not isinstance(locker_id, str):
                raise ValueError("preassigned parcel and locker IDs must be strings")
            parcel = self._parcels.get(parcel_id)
            locker = self.scenario.lockers.get(locker_id)
            if parcel is None or locker is None or not locker.available:
                raise ValueError(f"invalid preassigned slot: {parcel_id} -> {locker_id}")
            if locker_id not in (parcel.original_locker, *parcel.alternatives):
                raise ValueError(f"preassigned locker is ineligible for {parcel_id}: {locker_id}")
            reserved[locker_id] += 1
            if reserved[locker_id] > locker.capacity:
                raise ValueError(f"preassigned slots exceed capacity at {locker_id}")
        return bindings

    def observation(self) -> Decision | None:
        """Return the live decision, or None after all vehicles have returned."""
        return self._decision

    def _log(self, kind: str, **fields) -> None:
        self._events.append({"time_min": self._now, "kind": kind, **fields})

    def _free_slots(self, locker_id: str, parcel_id: str | None = None) -> int:
        locker = self.scenario.lockers[locker_id]
        owns_slot = parcel_id is not None and self._preassigned_slots.get(parcel_id) == locker_id
        return (
            locker.capacity - self._reserved[locker_id] - self._occupied[locker_id] + owns_slot
            if locker.available else 0
        )

    def _deposit_actions(
        self, parcels: tuple[Parcel, ...], node: str, now: int, battery: int, reserve: int,
    ) -> list[Action]:
        scenario = self.scenario
        actions = []
        for parcel in parcels:
            assigned = self._preassigned_slots.get(parcel.id)
            lockers = (assigned,) if assigned is not None else sorted(
                (parcel.original_locker, *parcel.alternatives)
            )
            for locker_id in lockers:
                if self._free_slots(locker_id, parcel.id) <= 0:
                    continue
                arc = node, locker_id
                back = locker_id, scenario.depot
                minutes = scenario.travel_time[arc]
                energy = scenario.travel_energy[arc]
                completion = now + minutes + scenario.service_min
                if (
                    completion <= parcel.deadline_min
                    and completion + scenario.travel_time[back] <= scenario.horizon_min
                    and battery - energy - scenario.travel_energy[back] >= reserve
                ):
                    actions.append(Action("deposit", parcel.id, locker_id, minutes, completion, energy))
        return actions

    def _charge_action(
        self, vehicle: _VehicleState, parcels: tuple[Parcel, ...], direct: Action,
    ) -> Action | None:
        if not self._charger_rate or not parcels:
            return None
        amount = self._charger_capacity - (vehicle.battery_wh - direct.energy_wh)
        if amount <= 0:
            return None
        start = max(direct.completion_min, self._charger_free_min)
        duration = (amount + self._charger_rate - 1) // self._charger_rate
        completion = start + duration
        if completion > self.scenario.horizon_min or not self._deposit_actions(
            parcels, self.scenario.depot, completion, self._charger_capacity, vehicle.reserve_wh,
        ):
            return None
        return Action("charge", None, None, direct.travel_min, completion, direct.energy_wh)

    def _make_decision(self, vehicle_id: str) -> Decision:
        scenario = self.scenario
        vehicle = self._vehicles[vehicle_id]
        owned = self._custody[vehicle_id]
        parcels = tuple(p for p in owned if p.id in self._open)
        deposits = self._deposit_actions(
            parcels, vehicle.node, self._now, vehicle.battery_wh, vehicle.reserve_wh,
        )
        back = vehicle.node, scenario.depot
        travel = scenario.travel_time[back]
        direct = Action("return", None, None, travel, self._now + travel, scenario.travel_energy[back])
        if (
            direct.completion_min > scenario.horizon_min
            or vehicle.battery_wh - direct.energy_wh < vehicle.reserve_wh
        ):
            raise ValueError(f"vehicle {vehicle_id} has no feasible depot return")
        actions = deposits
        charge = self._charge_action(vehicle, parcels, direct)
        if charge is not None:
            actions.append(charge)
        actions.append(direct)
        count = len(self._vehicles)
        supply = sum(self._free_slots(locker_id) for locker_id in scenario.lockers)
        actor = (
            (self._now - scenario.outage_min) / self._time_scale,
            vehicle.battery_wh / self._energy_scale,
            vehicle.reserve_wh / self._energy_scale,
            len(parcels) / max(1, len(owned)),
            (len(owned) - len(parcels)) / max(1, len(owned)),
            float(vehicle.node == scenario.depot),
            travel / self._time_scale,
            direct.energy_wh / self._energy_scale,
            float(bool(self._charger_rate)),
            max(0, self._charger_free_min - self._now) / self._time_scale,
        )
        critic = actor + (
            sum(not v.returned and key not in self._commitments for key, v in self._vehicles.items()) / count,
            len(self._commitments) / count,
            sum(v.returned for v in self._vehicles.values()) / count,
            len(self._deposit_times) / len(self._parcels),
            supply / self._slot_scale,
            sum(self._reserved.values()) / self._slot_scale,
        )
        return Decision(
            vehicle_id, self._now, tuple(actions), actor, critic,
            tuple(self._action_features(vehicle, action) for action in actions),
        )

    def _action_features(self, vehicle: _VehicleState, action: Action) -> tuple[float, ...]:
        deadline = self.scenario.horizon_min
        remaining = vehicle.battery_wh - action.energy_wh
        resource = 0.0
        if action.kind == "deposit":
            parcel = self._parcels[action.parcel_id]
            deadline = parcel.deadline_min
            locker = self.scenario.lockers[action.locker_id]
            resource = self._free_slots(action.locker_id, action.parcel_id) / locker.capacity
        elif action.kind == "charge":
            resource = (self._charger_capacity - remaining) / self._energy_scale
            remaining = self._charger_capacity
        return (
            float(action.kind == "deposit"),
            float(action.kind == "charge"),
            float(action.kind == "return"),
            action.travel_min / self._time_scale,
            (action.completion_min - self._now) / self._time_scale,
            action.energy_wh / self._energy_scale,
            (deadline - self._now) / self._time_scale,
            (deadline - action.completion_min) / self._time_scale,
            resource,
            (remaining - vehicle.reserve_wh) / self._energy_scale,
        )

    def _settle(self, vehicle_id: str, kind: str) -> None:
        commitment = self._commitments[vehicle_id]
        action = commitment.action
        vehicle = self._vehicles[vehicle_id]
        if kind == "charging_started":
            vehicle.node = self.scenario.depot
            self._log(kind, vehicle_id=vehicle_id, completion_min=action.completion_min)
            return
        if kind == "parcel_deposited":
            self._reserved[action.locker_id] -= 1
            self._occupied[action.locker_id] += 1
            self._open.remove(action.parcel_id)
            self._deposit_times.append(self._now)
            vehicle.node = action.locker_id
            self._log(kind, vehicle_id=vehicle_id, parcel_id=action.parcel_id, locker_id=action.locker_id)
        elif kind == "charging_completed":
            vehicle.battery_wh += commitment.charge_wh
            self._charged_wh += commitment.charge_wh
            self._log(kind, vehicle_id=vehicle_id, charged_wh=commitment.charge_wh,
                      remaining_wh=vehicle.battery_wh)
        else:
            vehicle.node = self.scenario.depot
            vehicle.returned = True
            self._log(kind, vehicle_id=vehicle_id, remaining_wh=vehicle.battery_wh)
        del self._commitments[vehicle_id]

    def _advance(self) -> Decision | None:
        while True:
            while self._pending and self._pending[0][0] == self._now:
                _, _, vehicle_id, kind = heapq.heappop(self._pending)
                self._settle(vehicle_id, kind)
            for vehicle_id, vehicle in self._vehicles.items():
                if not vehicle.returned and vehicle_id not in self._commitments:
                    return self._make_decision(vehicle_id)
            if not self._pending:
                return None
            self._now = self._pending[0][0]

    def step(self, action_index: int) -> float:
        """Commit a masked action, advance the clock, and return its team reward."""
        decision = self._decision
        if decision is None:
            raise ValueError("episode is already terminal")
        if type(action_index) is not int or not 0 <= action_index < len(decision.actions):
            raise ValueError("action index is outside the feasible action mask")
        vehicle_id = decision.vehicle_id
        vehicle = self._vehicles[vehicle_id]
        action = decision.actions[action_index]
        before = len(self._deposit_times)
        self._travel_min += action.travel_min
        self._travel_wh += action.energy_wh
        vehicle.battery_wh -= action.energy_wh
        charge_wh = 0
        if action.kind == "deposit":
            if action.parcel_id in self._preassigned_slots:
                # The slot stays held through this commitment and becomes occupied at deposit.
                del self._preassigned_slots[action.parcel_id]
            else:
                self._reserved[action.locker_id] += 1
            self._log("locker_reserved", vehicle_id=vehicle_id, parcel_id=action.parcel_id,
                      locker_id=action.locker_id)
            completion_kind = "parcel_deposited"
        elif action.kind == "charge":
            charge_wh = self._charger_capacity - vehicle.battery_wh
            arrival = self._now + action.travel_min
            start = max(arrival, self._charger_free_min)
            self._charger_free_min = action.completion_min
            self._log("charge_reserved", vehicle_id=vehicle_id, arrival_min=arrival,
                      start_min=start, completion_min=action.completion_min)
            heapq.heappush(self._pending, (start, 1, vehicle_id, "charging_started"))
            completion_kind = "charging_completed"
        else:
            completion_kind = "vehicle_returned"
        self._commitments[vehicle_id] = _Commitment(action, charge_wh)
        heapq.heappush(self._pending, (action.completion_min, 0, vehicle_id, completion_kind))
        self._decision = self._advance()
        return (
            (len(self._deposit_times) - before) / len(self._parcels)
            - 0.01 * action.energy_wh / (len(self._vehicles) * self._energy_scale)
        )

    def result(self) -> dict:
        if self._decision is not None:
            raise ValueError("result is only available after the episode terminates")
        total = len(self._parcels)
        served = len(self._deposit_times)
        threshold = (95 * total + 99) // 100
        return {
            "events": [event.copy() for event in self._events],
            "metrics": {
                "total_parcels": total,
                "served_parcels": served,
                "unserved_parcels": len(self._open),
                "service_rate": served / total,
                "recovery_95_min": (
                    self._deposit_times[threshold - 1] - self.scenario.outage_min
                    if served >= threshold else None
                ),
                "travel_min": self._travel_min,
                "travel_wh": self._travel_wh,
                "charged_wh": self._charged_wh,
            },
            "unserved_parcel_ids": sorted(self._open),
        }
