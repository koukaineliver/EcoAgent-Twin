import csv
from dataclasses import replace
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

from ecoagent_twin.__main__ import DEFAULT_SCENARIO, _read_json, _run, _verify
from ecoagent_twin.controllers import exact_replan, keep_plan, rule_based
from ecoagent_twin.model import Scenario, Stop, evaluate_plan, evaluate_route


class OutageReplayTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario.from_dict(_read_json(DEFAULT_SCENARIO))

    def test_controllers_share_feasibility_and_exact_reserves_scarce_slot(self):
        plans = [policy(self.scenario) for policy in (keep_plan, rule_based, exact_replan)]
        for plan in plans:
            evaluate_plan(self.scenario, plan)
        self.assertEqual([sum(map(len, plan.values())) for plan in plans], [3, 5, 6])
        self.assertIn(Stop("P101", "west"), plans[-1]["ev-1"])
        self.assertIn(Stop("P201", "east"), plans[-1]["ev-2"])

    def test_closed_locker_and_battery_reserve_are_enforced(self):
        vehicle = self.scenario.vehicles[0]
        with self.assertRaisesRegex(ValueError, "infeasible parcel or locker"):
            evaluate_route(self.scenario, vehicle, (Stop("P101", "central"),))
        low_battery = replace(vehicle, battery_wh=vehicle.reserve_wh + 200)
        with self.assertRaisesRegex(ValueError, "battery reserve"):
            evaluate_route(self.scenario, low_battery, (Stop("P101", "west"),))

    def test_shared_slot_capacity_is_enforced_across_vehicles(self):
        with self.assertRaisesRegex(ValueError, "shared locker capacity"):
            evaluate_plan(self.scenario, {
                "ev-1": (Stop("P101", "east"),),
                "ev-2": (Stop("P201", "east"),),
            })

    def test_replay_checks_metrics_and_detects_edited_artifact(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            _run(DEFAULT_SCENARIO, run)
            _verify(run)
            with (run / "comparison.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["recovery_95_min"] for row in rows], ["", "", "14"])

            with (run / "comparison.csv").open("a", encoding="utf-8") as handle:
                handle.write("modified\n")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                _verify(run)


if __name__ == "__main__":
    unittest.main()
