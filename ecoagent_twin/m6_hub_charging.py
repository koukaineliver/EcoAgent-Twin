"""Versioned multi-site charging and search under the existing outage constraints."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
import heapq
from math import ceil, isfinite

from .m2_simulator import Action, OutageEnv, _Commitment
from .m6_dynamic_grid import Interval, _hour_energy


@dataclass(frozen=True)
class Hub:
    node: str
    rate_wh_per_min: int
    plugs: int = 1


class HubOutageEnv(OutageEnv):
    """Let cleanly certified routes charge at a depot or an eligible locker hub."""

    def __init__(self, scenario, *, hubs: tuple[Hub, ...], max_battery_wh: int,
                 preassigned_slots=None, dispatch_after_min=None):
        if type(max_battery_wh) is not int or max_battery_wh <= 0:
            raise ValueError("charge cap must be a positive integer")
        self.hubs = {hub.node: hub for hub in hubs}
        if len(self.hubs) != len(hubs):
            raise ValueError("charger sites must be distinct")
        for hub in hubs:
            if (hub.node != scenario.depot and
                    (hub.node not in scenario.lockers or not scenario.lockers[hub.node].available)):
                raise ValueError("locker chargers require an available locker")
            if any(type(value) is not int or value <= 0 for value in (hub.plugs, hub.rate_wh_per_min)):
                raise ValueError("charger plug count and rate must be positive integers")
        dispatch = scenario.outage_min if dispatch_after_min is None else dispatch_after_min
        self._plugs = {hub.node: [dispatch] * hub.plugs for hub in hubs}
        super().__init__(scenario, charger_rate_wh_per_min=0, charger_capacity_wh=max_battery_wh,
                         preassigned_slots=preassigned_slots, dispatch_after_min=dispatch_after_min)

    def _charging(self, vehicle, hub):
        scenario = self.scenario
        arc = vehicle.node, hub.node
        energy, travel = scenario.travel_energy[arc], scenario.travel_time[arc]
        after_arrival = vehicle.battery_wh - energy
        amount = self._charger_capacity - after_arrival
        if after_arrival < vehicle.reserve_wh or amount <= 0:
            return None
        arrival = self._now + travel
        free = self._plugs[hub.node]
        plug = min(range(len(free)), key=lambda index: (free[index], index))
        start = max(arrival, free[plug])
        finish = start + ceil(amount / hub.rate_wh_per_min)
        back = hub.node, scenario.depot
        if (finish + scenario.travel_time[back] > scenario.horizon_min
                or self._charger_capacity - scenario.travel_energy[back] < vehicle.reserve_wh):
            return None
        return Action("charge", None, hub.node, travel, finish, energy), amount, arrival, start, plug

    def _make_decision(self, vehicle_id):
        decision = super()._make_decision(vehicle_id)
        vehicle = self._vehicles[vehicle_id]
        parcels = tuple(parcel for parcel in self._custody[vehicle_id] if parcel.id in self._open)
        charges = []
        for hub in self.hubs.values():
            booking = self._charging(vehicle, hub)
            if booking is None:
                continue
            action, _, _, _, _ = booking
            if self._deposit_actions(parcels, hub.node, action.completion_min, self._charger_capacity,
                                     vehicle.reserve_wh):
                charges.append(action)
        actions = (*decision.actions[:-1], *charges, decision.actions[-1])
        return replace(decision, actions=actions,
                       action_features=tuple(self._action_features(vehicle, item) for item in actions))

    def step(self, action_index):
        decision = self.observation()
        if decision is None or type(action_index) is not int or not 0 <= action_index < len(decision.actions):
            raise ValueError("action index is outside the feasible action mask")
        action = decision.actions[action_index]
        if action.kind != "charge":
            return super().step(action_index)
        vehicle_id = decision.vehicle_id
        vehicle = self._vehicles[vehicle_id]
        _, amount, arrival, start, plug = self._charging(vehicle, self.hubs[action.locker_id])
        self._plugs[action.locker_id][plug] = action.completion_min
        before = len(self._deposit_times)
        self._travel_min += action.travel_min
        self._travel_wh += action.energy_wh
        vehicle.battery_wh -= action.energy_wh
        self._log("charge_reserved", vehicle_id=vehicle_id, hub_id=action.locker_id, plug_id=plug,
                  arrival_min=arrival, start_min=start, completion_min=action.completion_min,
                  queue_wait_min=start - arrival, arrival_soc_wh=vehicle.battery_wh)
        self._commitments[vehicle_id] = _Commitment(action, amount)
        heapq.heappush(self._pending, (start, 1, vehicle_id, "charging_started"))
        heapq.heappush(self._pending, (action.completion_min, 0, vehicle_id, "charging_completed"))
        self._decision = self._advance()
        return ((len(self._deposit_times) - before) / len(self._parcels)
                - 0.01 * action.energy_wh / (len(self._vehicles) * self._energy_scale))

    def _settle(self, vehicle_id, kind):
        if kind == "charging_started":
            action = self._commitments[vehicle_id].action
            self._vehicles[vehicle_id].node = action.locker_id
            self._log(kind, vehicle_id=vehicle_id, hub_id=action.locker_id, completion_min=action.completion_min)
        else:
            super()._settle(vehicle_id, kind)


def replay(factory, actions: list[dict]) -> dict:
    environment = factory()
    for event in actions:
        view = environment.observation()
        if view is None or (view.vehicle_id, view.time_min) != (event["vehicle_id"], event["time_min"]):
            raise ValueError("archived action differs from live decision epoch")
        environment.step(view.actions.index(Action(**event["action"])))
    return environment.result()


def _dispatch(view, action):
    return {"vehicle_id": view.vehicle_id, "time_min": view.time_min, "action": asdict(action)}


def search(factory, objective, *, transition_limit: int) -> dict:
    """Bounded full fleet search, pruning only impossible service improvements."""
    if type(transition_limit) is not int or transition_limit < 1:
        raise ValueError("transition limit must be a positive integer")
    root = factory()
    scenario = root.scenario
    ownership = Counter(parcel.vehicle_id for parcel in scenario.parcels)
    safety = factory()
    incumbent = []
    while (view := safety.observation()) is not None:
        action = view.actions[-1]
        incumbent.append(_dispatch(view, action))
        safety.step(len(view.actions) - 1)
    best_result = safety.result()
    best_rank = objective(best_result)

    def branches(environment, view, path, available, served):
        order = sorted(range(len(view.actions)), key=lambda i: (
            {"deposit": 0, "charge": 1, "return": 2}[view.actions[i].kind],
            view.actions[i].completion_min, view.actions[i].energy_wh, i,
        ), reverse=True)
        return [(environment, path, available, served, index) for index in order]

    pending = branches(root, root.observation(), (), len(scenario.parcels), Counter())
    transitions = pruned = 0
    exact = True
    while pending:
        saved, path, available, served, index = pending.pop()
        view = saved.observation()
        action = view.actions[index]
        if action.kind == "return":
            available -= ownership[view.vehicle_id] - served[view.vehicle_id]
        if -available > best_rank[0]:
            pruned += 1
            continue
        if transitions == transition_limit:
            exact = False
            break
        if action.kind == "deposit":
            served = served.copy()
            served[view.vehicle_id] += 1
        child = deepcopy(saved, memo={id(scenario): scenario})
        child.step(index)
        transitions += 1
        path = (*path, _dispatch(view, action))
        if (next_view := child.observation()) is not None:
            pending.extend(branches(child, next_view, path, available, served))
        else:
            result = child.result()
            if (rank := objective(result)) < best_rank:
                incumbent, best_result, best_rank = list(path), result, rank
    if replay(factory, incumbent) != best_result:
        raise ValueError("selected route disagrees with fresh simulator replay")
    return {"actions": incumbent, "result": best_result, "transitions": transitions, "pruned": pruned,
            "exact_in_simulator": exact, "objective": list(best_rank)}


def energy_ledger(result: dict, scenario, *, terminal_target_wh: int, terminal_rate_wh_per_min: int,
                  terminal_offset_min: int = 15) -> list[dict]:
    """Account for real in-simulator charge and a standardized post-horizon plug."""
    if type(terminal_target_wh) is not int or type(terminal_rate_wh_per_min) is not int or (
        terminal_target_wh < max(vehicle.battery_wh for vehicle in scenario.vehicles)
        or terminal_rate_wh_per_min <= 0 or terminal_offset_min < 0
    ):
        raise ValueError("terminal SOC and replenishment schedule are invalid")
    initial = {vehicle.id: vehicle.battery_wh for vehicle in scenario.vehicles}
    charges, terminal, open_sessions = [], {}, {}
    for event in result["events"]:
        kind = event["kind"]
        if kind == "charge_reserved":
            open_sessions[event["vehicle_id"]] = event
        elif kind == "charging_completed":
            charge = open_sessions.pop(event["vehicle_id"])
            if charge["completion_min"] != event["time_min"]:
                raise ValueError("charge receipt differs from completion time")
            charges.append({"kind": "in_window", "vehicle_id": event["vehicle_id"], "hub_id": charge["hub_id"],
                            "start_min": charge["start_min"], "end_min": event["time_min"],
                            "battery_wh": event["charged_wh"]})
        elif kind == "vehicle_returned":
            terminal[event["vehicle_id"]] = event["remaining_wh"]
    if open_sessions or set(terminal) != set(initial):
        raise ValueError("missing charge completion or terminal SOC")
    metric = result["metrics"]
    onboard = sum(charge["battery_wh"] for charge in charges)
    if onboard != metric["charged_wh"] or sum(initial.values()) + onboard - metric["travel_wh"] != sum(terminal.values()):
        raise ValueError("in-window battery energy does not reconcile")
    clock = scenario.horizon_min + terminal_offset_min
    for vehicle_id, soc in sorted(terminal.items()):
        amount = terminal_target_wh - soc
        if amount < 0:
            raise ValueError("terminal stock exceeds common target")
        if amount:
            end = clock + ceil(amount / terminal_rate_wh_per_min)
            charges.append({"kind": "post_horizon", "vehicle_id": vehicle_id, "hub_id": scenario.depot,
                            "start_min": clock, "end_min": end, "battery_wh": amount})
            clock = end
    required = metric["travel_wh"] + sum(terminal_target_wh - soc for soc in initial.values())
    if sum(part["battery_wh"] for part in charges) != required:
        raise ValueError("equalized electricity differs from traction and fixed stock increment")
    return charges


def carbon(rows: list[dict], start: datetime, rates: dict[datetime, float], delivery_factor: float) -> float:
    """Return kg CO2 from frozen regional UTC hour-ending rates and plug intervals."""
    if start.utcoffset() != timedelta(0) or not 0 < delivery_factor <= 1:
        raise ValueError("a UTC origin and valid delivery factor are required")
    kilograms = 0.0
    for row in rows:
        interval = Interval(start + timedelta(minutes=row["start_min"]), start + timedelta(minutes=row["end_min"]),
                            row["battery_wh"], row["kind"])
        for hour, wh in _hour_energy(interval):
            rate = rates[hour]
            if not isfinite(rate) or rate < 0:
                raise ValueError("grid intensity must be finite and nonnegative")
            kilograms += rate * wh / 1_000 / delivery_factor
    return kilograms
