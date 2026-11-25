import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import unittest

from ecoagent_twin.m2_simulator import OutageEnv
from ecoagent_twin.m3_route_repair import repair
from ecoagent_twin.model import Scenario


BASE = Path(__file__).resolve().parents[1] / "ecoagent_twin/scenarios/locker_outage.json"
NO_CHARGE = {"charger_rate_wh_per_min": 0, "charger_capacity_wh": 0}


@unittest.skipUnless(importlib.util.find_spec("ortools"), "OR-Tools is required for M3 repair tests")
class RouteRepairTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(BASE.read_text(encoding="utf-8"))
        self.scenario = Scenario.from_dict(self.raw)

    def replay(self, plan, scenario, **options):
        env = OutageEnv(scenario, **options)
        while (decision := env.observation()) is not None:
            env.step(plan.choose(decision))
        result = env.result()
        self.assertEqual(result["metrics"], plan.receipt["certified_metrics"])
        self.assertEqual(result["unserved_parcel_ids"], plan.receipt["certified_unserved_parcel_ids"])
        return result

    def test_uses_cp_sat_and_optimizes_shared_slots_with_deterministic_ties(self):
        first = repair(self.scenario, **NO_CHARGE, time_limit_s=2)
        second = repair(self.scenario, **NO_CHARGE, time_limit_s=2)
        self.assertEqual(first.receipt["strategy"], "cp_sat")
        self.assertTrue(first.receipt["all_objectives_optimal_within_columns"])
        self.assertEqual(first.receipt["certified_dispatches"], second.receipt["certified_dispatches"])
        result = self.replay(first, self.scenario, **NO_CHARGE)
        self.assertEqual(result["metrics"]["served_parcels"], 6)
        self.assertEqual(result["metrics"]["travel_min"], 23)
        east = [event for event in result["events"] if event.get("locker_id") == "east"
                and event["kind"] == "parcel_deposited"]
        self.assertEqual([event["parcel_id"] for event in east], ["P201"])

    def test_preserves_offer_slots_and_delayed_dispatch(self):
        options = {**NO_CHARGE, "preassigned_slots": {"P101": "west", "P201": "east"},
                   "dispatch_after_min": 21}
        plan = repair(self.scenario, **options, time_limit_s=2)
        self.assertEqual(plan.receipt["strategy"], "cp_sat")
        result = self.replay(plan, self.scenario, **options)
        self.assertEqual(result["metrics"]["served_parcels"], 6)
        self.assertEqual(result["metrics"]["recovery_95_min"], 15)
        for parcel, locker in options["preassigned_slots"].items():
            self.assertIn((parcel, locker), [
                (event["parcel_id"], event["locker_id"])
                for event in result["events"] if event["kind"] == "parcel_deposited"
            ])

    def test_cp_sat_schedules_two_queued_charges_without_using_the_fallback(self):
        raw = self.raw
        raw["outage"]["time_min"], raw["horizon_min"] = 0, 12
        raw["locker_free_capacity"]["east"] = 2
        for vehicle in raw["vehicles"]:
            vehicle["battery_wh"], vehicle["reserve_wh"] = 80, 40
        raw["parcels_in_original_plan_order"] = [
            {**parcel, "alternatives": ["east"], "deadline_min": 11}
            for parcel in raw["parcels_in_original_plan_order"] if parcel["id"] in ("P101", "P201")
        ]
        nodes = raw["travel_minutes"]
        raw["travel_minutes"] = {a: {b: int(a != b) for b in nodes} for a in nodes}
        raw["travel_wh"] = {a: {b: 25 * int(a != b) for b in nodes} for a in nodes}
        scenario = Scenario.from_dict(raw)
        options = {"charger_rate_wh_per_min": 20, "charger_capacity_wh": 120}
        plan = repair(scenario, **options, time_limit_s=2)
        self.assertEqual(plan.receipt["strategy"], "cp_sat")
        self.assertEqual(plan.receipt["selected_charge_queue_wait_min"], {"ev-1": 0, "ev-2": 4})
        result = self.replay(plan, scenario, **options)
        self.assertEqual(result["metrics"]["served_parcels"], 2)
        self.assertEqual(result["metrics"]["charged_wh"], 130)
        self.assertEqual([e["time_min"] for e in result["events"] if e["kind"] == "charging_started"],
                         [1, 5])

    def test_reports_limited_search_and_certifies_fallback(self):
        capped = repair(self.scenario, **NO_CHARGE, columns_per_vehicle=1, time_limit_s=2)
        self.assertEqual(capped.receipt["column_capped_vehicles"], ["ev-1", "ev-2"])
        self.assertEqual(capped.receipt["certified_metrics"]["served_parcels"], 0)
        timed_out = repair(self.scenario, **NO_CHARGE, time_limit_s=1e-12)
        self.assertEqual(timed_out.receipt["strategy"], "rule_based_fallback")
        self.assertEqual(timed_out.receipt["fallback_reason"], "cp_sat_no_incumbent")
        self.assertEqual(self.replay(timed_out, self.scenario, **NO_CHARGE)["metrics"]["served_parcels"], 5)

    def test_mismatched_live_mask_raises_without_consuming_the_action(self):
        plan = repair(self.scenario, **NO_CHARGE, time_limit_s=2)
        decision = OutageEnv(self.scenario, **NO_CHARGE).observation()
        with self.assertRaisesRegex(ValueError, "masked"):
            plan.choose(replace(decision, actions=()))
        self.assertEqual(self.replay(plan, self.scenario, **NO_CHARGE)["metrics"]["served_parcels"], 6)


if __name__ == "__main__":
    unittest.main()
