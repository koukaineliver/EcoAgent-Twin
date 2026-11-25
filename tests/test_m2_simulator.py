import json
from pathlib import Path
import unittest

from ecoagent_twin.m2_rule import RuleBased, choose
from ecoagent_twin.m2_simulator import OutageEnv
from ecoagent_twin.model import Scenario


def small_scenario(*, capacity=1, batteries=(200, 200), horizon=12, deadline=11):
    nodes = ("depot", "start", "closed", "available")
    vehicles = [
        {"id": f"ev-{index}", "node": "start", "battery_wh": battery, "reserve_wh": 40}
        for index, battery in enumerate(batteries, 1)
    ]
    return Scenario.from_dict({
        "schema": 1,
        "data_basis": "synthetic",
        "id": "m2-invariant-fixture",
        "depot": "depot",
        "outage": {"locker_id": "closed", "time_min": 0},
        "horizon_min": horizon,
        "service_min_per_parcel": 1,
        "locker_free_capacity": {"closed": 2, "available": capacity},
        "vehicles": vehicles,
        "parcels_in_original_plan_order": [
            {
                "id": f"P{index}", "vehicle_id": vehicle["id"],
                "original_locker": "closed", "alternatives": ["available"],
                "deadline_min": deadline,
            }
            for index, vehicle in enumerate(vehicles, 1)
        ],
        "travel_minutes": {a: {b: int(a != b) for b in nodes} for a in nodes},
        "travel_wh": {a: {b: 25 * int(a != b) for b in nodes} for a in nodes},
    })


class SimulatorTest(unittest.TestCase):
    def env(self, scenario, *, charge_cap=120):
        return OutageEnv(
            scenario, charger_rate_wh_per_min=20, charger_capacity_wh=charge_cap
        )

    def finish(self, env, rule):
        action_sequence = []
        for _ in range(30):
            decision = env.observation()
            if decision is None:
                return action_sequence
            index = rule.choose(decision)
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, len(decision.actions))
            reward = env.step(index)
            action_sequence.append((decision, index, reward))
        self.fail("sequential rollout exceeded the fixture action bound")

    def test_rule_rollout_replays_and_prefers_earlier_deadline(self):
        fixture = Path(__file__).resolve().parents[1] / "ecoagent_twin/scenarios/locker_outage.json"
        scenario = Scenario.from_dict(json.loads(fixture.read_text(encoding="utf-8")))
        first, second = self.env(scenario, charge_cap=2500), self.env(scenario, charge_cap=2500)
        decision = first.observation()
        action = decision.actions[choose(decision, scenario)]
        self.assertEqual((action.kind, action.parcel_id, action.locker_id), ("deposit", "P101", "east"))

        rule = RuleBased(scenario)
        self.assertEqual(self.finish(first, rule), self.finish(second, rule))
        self.assertEqual(first.result(), second.result())
        result = first.result()
        metrics = result["metrics"]
        self.assertEqual(metrics["total_parcels"], 6)
        self.assertEqual(metrics["served_parcels"] + metrics["unserved_parcels"], 6)
        self.assertEqual(len(result["unserved_parcel_ids"]), metrics["unserved_parcels"])

    def test_same_time_decision_cannot_double_reserve_last_slot(self):
        scenario = small_scenario()
        env = self.env(scenario, charge_cap=200)
        rule = RuleBased(scenario)
        first = env.observation()
        first_idx = rule.choose(first)
        self.assertEqual(first.actions[first_idx].kind, "deposit")
        env.step(first_idx)

        second = env.observation()
        self.assertEqual((first.vehicle_id, second.vehicle_id), ("ev-1", "ev-2"))
        self.assertEqual(first.time_min, second.time_min)
        self.assertEqual([action.kind for action in second.actions], ["return"])
        self.finish(env, rule)
        result = env.result()
        self.assertEqual(result["metrics"]["served_parcels"], 1)
        self.assertEqual(tuple(result["unserved_parcel_ids"]), ("P2",))
        self.assertIsNone(result["metrics"]["recovery_95_min"])
        self.assertEqual(
            sum(event["kind"] == "locker_reserved" for event in result["events"]), 1
        )

    def test_deadline_horizon_and_reserve_mask_infeasible_deposit(self):
        for scenario in (
            small_scenario(batteries=(80,)),
            small_scenario(batteries=(200,), deadline=1),
            small_scenario(batteries=(200,), horizon=2, deadline=2),
        ):
            with self.subTest(scenario=scenario):
                env = self.env(scenario, charge_cap=scenario.vehicles[0].battery_wh)
                self.assertEqual([a.kind for a in env.observation().actions], ["return"])
                self.finish(env, RuleBased(scenario))
                result = env.result()
                self.assertEqual(result["metrics"]["served_parcels"], 0)
                self.assertEqual(result["metrics"]["charged_wh"], 0)
                returns = [
                    event for event in result["events"] if event["kind"] == "vehicle_returned"
                ]
                self.assertEqual(len(returns), 1)
                self.assertLessEqual(returns[0]["time_min"], scenario.horizon_min)
                self.assertGreaterEqual(returns[0]["remaining_wh"], scenario.vehicles[0].reserve_wh)

    def test_charging_is_serial_and_limited_to_battery_capacity(self):
        scenario = small_scenario(capacity=2, batteries=(80, 80))
        env = self.env(scenario)
        rule = RuleBased(scenario)
        decision = env.observation()
        self.assertNotIn("deposit", [action.kind for action in decision.actions])
        self.assertEqual(decision.actions[rule.choose(decision)].kind, "charge")

        self.finish(env, rule)
        result = env.result()
        starts = [
            event["time_min"] for event in result["events"] if event["kind"] == "charging_started"
        ]
        ends = [
            event["time_min"] for event in result["events"] if event["kind"] == "charging_completed"
        ]
        self.assertEqual(starts, [1, 5])
        self.assertEqual(ends, [5, 9])
        metrics = result["metrics"]
        self.assertEqual(metrics["charged_wh"], 130)
        self.assertEqual(metrics["served_parcels"], 2)
        self.assertEqual(metrics["recovery_95_min"], 11)

    def test_invalid_action_preserves_decision_and_event_log(self):
        scenario = small_scenario()
        env = self.env(scenario, charge_cap=200)
        decision = env.observation()
        for index in (-1, len(decision.actions)):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    env.step(index)
                self.assertEqual(env.observation(), decision)
        reference = self.env(scenario, charge_cap=200)
        rule = RuleBased(scenario)
        self.assertEqual(self.finish(env, rule), self.finish(reference, rule))
        self.assertEqual(env.result()["events"], reference.result()["events"])


if __name__ == "__main__":
    unittest.main()
