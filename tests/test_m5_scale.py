from copy import deepcopy
import json
from pathlib import Path
import unittest

from ecoagent_twin.m2_rule import RuleBased
from ecoagent_twin.m2_simulator import OutageEnv
from ecoagent_twin.m4_chaos import inject
from ecoagent_twin.m5_scale import FLEET_SIZES, scale_workload
from ecoagent_twin.model import Scenario


FIXTURE = Path(__file__).resolve().parents[1] / "ecoagent_twin/scenarios/locker_outage.json"


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixed_tiers_preserve_template_costs_and_scale_capacity(self):
        before = deepcopy(self.raw)
        for fleet_size in FLEET_SIZES:
            with self.subTest(fleet_size=fleet_size):
                raw, meta = scale_workload(self.raw, fleet_size)
                scenario = Scenario.from_dict(raw)
                count = fleet_size // 2
                self.assertEqual(len(scenario.vehicles), fleet_size)
                self.assertEqual(len(scenario.parcels), 6 * count)
                self.assertEqual(meta["open_locker_count"], 3 * count)
                self.assertEqual(meta["open_slots"], 6 * count)
                self.assertEqual(len(scenario.lockers), 1 + 3 * count)
                self.assertEqual([k for k, locker in scenario.lockers.items() if not locker.available],
                                 [scenario.outage_locker])
                self.assertEqual(meta["shared_infrastructure"]["depot"], self.raw["depot"])
                origins = meta["node_origins"]
                for field in ("travel_minutes", "travel_wh"):
                    self.assertTrue(all(
                        cost == self.raw[field][origins[a]][origins[b]]
                        for a, row in raw[field].items() for b, cost in row.items()
                    ))
                self.assertEqual((raw, meta), scale_workload(self.raw, fleet_size))
        self.assertEqual(self.raw, before)

    def test_retains_disclosed_custody_deadlines_and_battery_in_each_replica(self):
        disrupted, charger, _ = inject(
            self.raw, "compound", "severe", 5,
            charger_rate_wh_per_min=180, charger_capacity_wh=2000,
        )
        raw, meta = scale_workload(disrupted, 8)
        source = Scenario.from_dict(disrupted)
        scenario = Scenario.from_dict(raw)
        vehicles = {vehicle.id: vehicle for vehicle in scenario.vehicles}
        parcels = {parcel.id: parcel for parcel in scenario.parcels}
        self.assertEqual(len(parcels), 32)
        for replica in meta["replicas"]:
            for vehicle in source.vehicles:
                copied = vehicles[replica["vehicles"][vehicle.id]]
                self.assertEqual((copied.node, copied.battery_wh, copied.reserve_wh),
                                 (vehicle.node, vehicle.battery_wh, vehicle.reserve_wh))
            for parcel in source.parcels:
                copied = parcels[replica["parcels"][parcel.id]]
                self.assertEqual(copied.vehicle_id, replica["vehicles"][parcel.vehicle_id])
                self.assertEqual(copied.deadline_min, parcel.deadline_min)
                self.assertEqual(copied.original_locker, replica["lockers"][parcel.original_locker])
                self.assertEqual(copied.alternatives,
                                 tuple(replica["lockers"][locker] for locker in parcel.alternatives))
        env = OutageEnv(scenario, **charger)
        rule = RuleBased(scenario)
        while (decision := env.observation()) is not None:
            env.step(rule.choose(decision))
        self.assertEqual(env.result()["metrics"]["total_parcels"], 32)

    def test_pooled_layout_preserves_ids_and_increases_open_capacity(self):
        raw, meta = scale_workload(self.raw, 4, locker_layout="pooled")
        self.assertEqual(raw["travel_minutes"], self.raw["travel_minutes"])
        self.assertEqual(raw["travel_wh"], self.raw["travel_wh"])
        self.assertEqual(meta["open_locker_count"], 3)
        self.assertEqual(meta["open_slots"], 12)
        self.assertEqual(meta["replicas"][0]["lockers"], meta["replicas"][1]["lockers"])
        self.assertEqual(raw["locker_free_capacity"]["central"], self.raw["locker_free_capacity"]["central"])

    def test_rejects_invalid_scales_and_layout(self):
        for size in (0, 1, 3, 2.0, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "multiple"):
                scale_workload(self.raw, size)
        with self.assertRaisesRegex(ValueError, "layout"):
            scale_workload(self.raw, 4, locker_layout="unspecified")


if __name__ == "__main__":
    unittest.main()
