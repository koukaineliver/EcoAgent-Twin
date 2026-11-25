from copy import deepcopy
import json
from pathlib import Path
import unittest

from ecoagent_twin.m2_simulator import OutageEnv
from ecoagent_twin.m4_chaos import COMPOUND_COMPONENTS, KINDS, SEVERITIES, inject
from ecoagent_twin.model import Scenario, evaluate_route


FIXTURE = Path(__file__).resolve().parents[1] / "ecoagent_twin/scenarios/locker_outage.json"
CHARGER = {"charger_rate_wh_per_min": 180, "charger_capacity_wh": 2_000}


class ChaosTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_all_variants_are_paired_reproducible_and_can_return(self):
        original = deepcopy(self.base)
        for kind in KINDS:
            for severity in SEVERITIES:
                with self.subTest(kind=kind, severity=severity):
                    first = inject(self.base, kind, severity, 17, **CHARGER)
                    self.assertEqual(first, inject(self.base, kind, severity, 17, **CHARGER))
                    raw, charger, events = first
                    self.assertTrue(all(row["time_min"] == self.base["outage"]["time_min"] for row in events))
                    self.assertEqual(events[0]["kind"], "locker_outage")
                    scenario = Scenario.from_dict(raw)
                    for vehicle in scenario.vehicles:
                        self.assertGreaterEqual(evaluate_route(scenario, vehicle, ()).remaining_wh,
                                                vehicle.reserve_wh)
                    environment = OutageEnv(scenario, **charger)
                    while (decision := environment.observation()) is not None:
                        direct = next(i for i, action in enumerate(decision.actions) if action.kind == "return")
                        environment.step(direct)
                    self.assertEqual(environment.result()["metrics"]["total_parcels"], len(scenario.parcels))
        self.assertEqual(self.base, original)

    def test_baseline_and_saturation_keep_the_closed_locker_unchanged(self):
        baseline, charger, events = inject(self.base, "locker_outage", "severe", 42, **CHARGER)
        self.assertEqual({**baseline, "id": self.base["id"]}, self.base)
        self.assertEqual(charger, CHARGER)
        self.assertEqual(len(events), 1)
        severe, charger, events = inject(self.base, "locker_saturation", "severe", 42, **CHARGER)
        closed = self.base["outage"]["locker_id"]
        before, after = self.base["locker_free_capacity"], severe["locker_free_capacity"]
        self.assertEqual(after[closed], before[closed])
        changes = events[1]["payload"]["capacity_changes"]
        self.assertEqual(len(changes), 2)
        for row in changes:
            self.assertGreater(row["free_slots_before"], row["free_slots_after"])
            self.assertEqual(after[row["locker_id"]], row["free_slots_after"])

    def test_charger_failure_includes_observed_low_battery_without_breaking_return(self):
        raw, charger, events = inject(self.base, "charger_outage", "severe", 9, **CHARGER)
        self.assertEqual(charger, {**CHARGER, "charger_rate_wh_per_min": 0})
        self.assertEqual(raw["travel_minutes"], self.base["travel_minutes"])
        self.assertEqual(raw["travel_wh"], self.base["travel_wh"])
        for row in events[1]["payload"]["battery_changes"]:
            self.assertLess(row["battery_after_wh"], row["battery_before_wh"])
            self.assertIn(row["battery_after_wh"] - row["direct_return_and_reserve_wh"], range(15, 76))

    def test_road_detours_are_directed_and_use_unblocked_measured_legs(self):
        raw, _, events = inject(self.base, "road_blockage", "severe", 33, **CHARGER)
        replacements = events[1]["payload"]["replacements"]
        self.assertEqual(len(replacements), 2)
        before, after = self.base["travel_minutes"], raw["travel_minutes"]
        changes = {(a, b) for a in before for b in before[a] if before[a][b] != after[a][b]}
        self.assertEqual(changes, {(row["origin"], row["destination"]) for row in replacements})
        for row in replacements:
            origin, destination, via = row["origin"], row["destination"], row["via"]
            self.assertGreater(after[origin][destination], before[origin][destination])
            self.assertNotEqual(destination, self.base["depot"])
            self.assertNotIn((origin, via), changes)
            self.assertNotIn((via, destination), changes)
            self.assertEqual(after[origin][destination], before[origin][via] + before[via][destination])
            energy = self.base["travel_wh"]
            self.assertEqual(raw["travel_wh"][origin][destination],
                             energy[origin][via] + energy[via][destination])

    def test_compound_freezes_original_and_disclosed_custody_in_denominator(self):
        self.base["parcels_in_original_plan_order"][0]["id"] = "surge-01"
        raw, charger, events = inject(self.base, "compound", "severe", 2, **CHARGER)
        self.assertEqual([event["kind"] for event in events],
                         ["locker_outage", "compound", *COMPOUND_COMPONENTS])
        self.assertEqual(charger["charger_rate_wh_per_min"], 0)
        existing = self.base["parcels_in_original_plan_order"]
        supplied = raw["parcels_in_original_plan_order"]
        self.assertEqual(supplied[:len(existing)], existing)
        self.assertEqual(len({parcel["id"] for parcel in supplied}), len(existing) + 2)
        added = events[-1]["payload"]["added_parcels"]
        self.assertEqual(added, supplied[len(existing):])
        self.assertEqual(len({row["vehicle_id"] for row in added}), 2)
        self.assertEqual(events[-1]["payload"]["onboard_after"], len(supplied))


if __name__ == "__main__":
    unittest.main()
