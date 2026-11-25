from dataclasses import replace
import json
from pathlib import Path
import unittest

from ecoagent_twin.m2_rule import RuleBased
from ecoagent_twin.m2_simulator import OutageEnv
from ecoagent_twin.model import Scenario


FIXTURE = Path(__file__).resolve().parents[1] / "ecoagent_twin" / "scenarios" / "locker_outage.json"


class OfferSlotsTest(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))

    def env(self, scenario=None, **options):
        return OutageEnv(
            self.scenario if scenario is None else scenario,
            **{"charger_rate_wh_per_min": 0, "charger_capacity_wh": 0, **options},
        )

    def step(self, env, kind, parcel_id=None, locker_id=None):
        decision = env.observation()
        index = next(
            index for index, action in enumerate(decision.actions)
            if (action.kind, action.parcel_id, action.locker_id) == (kind, parcel_id, locker_id)
        )
        env.step(index)

    def finish(self, env):
        rule = RuleBased(env.scenario)
        for _ in range(20):
            decision = env.observation()
            if decision is None:
                return env.result()
            env.step(rule.choose(decision))
        self.fail("rollout exceeded the fixture action bound")

    def test_slot_at_capacity_is_visible_only_to_its_owner(self):
        env = self.env(preassigned_slots={"P201": "east"}, dispatch_after_min=20)
        first = env.observation()
        self.assertEqual(first.vehicle_id, "ev-1")
        self.assertNotIn("east", [action.locker_id for action in first.actions])
        self.step(env, "deposit", "P101", "west")
        second = env.observation()
        self.assertEqual((second.vehicle_id, second.time_min), ("ev-2", first.time_min))
        self.step(env, "deposit", "P201", "east")
        result = self.finish(env)
        self.assertIn(
            {"time_min": 20, "kind": "offer_slot_preassigned", "vehicle_id": "ev-2",
             "parcel_id": "P201", "locker_id": "east"},
            result["events"],
        )
        self.assertNotIn("P201", result["unserved_parcel_ids"])

    def test_commitment_converts_the_reservation_without_an_extra_slot(self):
        lockers = dict(self.scenario.lockers)
        lockers["east"] = replace(lockers["east"], capacity=2)
        scenario = replace(self.scenario, lockers=lockers)
        bindings = {"P101": "east"}
        env = self.env(scenario, preassigned_slots=bindings)
        bindings.clear()
        first = env.observation()
        self.assertEqual(
            [action.locker_id for action in first.actions if action.parcel_id == "P101"], ["east"]
        )
        self.step(env, "deposit", "P101", "east")
        self.step(env, "deposit", "P201", "east")
        result = self.finish(env)
        east_deposits = [
            event["parcel_id"] for event in result["events"]
            if event["kind"] == "parcel_deposited" and event["locker_id"] == "east"
        ]
        self.assertEqual(east_deposits, ["P101", "P201"])

    def test_unserved_offer_keeps_slot_reserved(self):
        env = self.env(preassigned_slots={"P101": "east"})
        self.step(env, "return")
        second = env.observation()
        self.assertEqual(second.vehicle_id, "ev-2")
        self.assertNotIn("P201", [action.parcel_id for action in second.actions])
        result = self.finish(env)
        self.assertTrue({"P101", "P201"} <= set(result["unserved_parcel_ids"]))

    def test_dispatch_delay_counts_in_recovery_from_the_true_alert(self):
        scenario = replace(
            self.scenario, parcels=self.scenario.parcels[:1], vehicles=self.scenario.vehicles[:1]
        )
        env = self.env(scenario, preassigned_slots={"P101": "east"}, dispatch_after_min=22)
        self.assertEqual(env.observation().time_min, 22)
        result = self.finish(env)
        self.assertEqual(
            [(event["kind"], event["time_min"]) for event in result["events"][:3]],
            [("locker_outage", 20), ("offer_slot_preassigned", 22), ("routing_started", 22)],
        )
        self.assertEqual(result["metrics"]["recovery_95_min"], 4)

    def test_charge_can_unlock_a_parcels_own_reserved_slot(self):
        scenario = replace(
            self.scenario,
            parcels=(replace(self.scenario.parcels[0], deadline_min=34),),
            vehicles=(replace(self.scenario.vehicles[0], battery_wh=550, reserve_wh=400),),
        )
        env = self.env(
            scenario, preassigned_slots={"P101": "east"},
            charger_rate_wh_per_min=700, charger_capacity_wh=900,
        )
        decision = env.observation()
        self.assertEqual([action.kind for action in decision.actions], ["charge", "return"])
        result = self.finish(env)
        self.assertEqual(result["metrics"]["charged_wh"], 470)
        self.assertEqual(result["metrics"]["served_parcels"], 1)

    def test_invalid_bindings_and_clock_are_rejected(self):
        for slots in (
            {"missing": "east"},
            {"P101": "missing"},
            {"P101": "central"},
            {"P201": "west"},
            {"P101": "east", "P201": "east"},
        ):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                self.env(preassigned_slots=slots)
        for dispatch in (19, 43, 22.5, True, 42):
            with self.subTest(dispatch=dispatch), self.assertRaisesRegex(ValueError, "dispatch clock"):
                self.env(dispatch_after_min=dispatch)

    def test_opt_in_at_alert_with_no_bindings_keeps_default_outcome(self):
        baseline = self.finish(self.env())
        opted_in = self.finish(self.env(dispatch_after_min=self.scenario.outage_min))
        events = opted_in["events"]
        self.assertEqual(events.pop(1), {"kind": "routing_started", "time_min": 20})
        self.assertEqual(opted_in, baseline)


if __name__ == "__main__":
    unittest.main()
