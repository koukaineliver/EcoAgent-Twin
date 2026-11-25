from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ecoagent_twin.m2_offer_allocator import CandidateOffer, allocate_ortools, allocate_rule
from ecoagent_twin.m2_offer_experiment import DEFAULT_PROTOCOL, DEFAULT_SCENARIO, run, verify


class OfferPilotTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("mesa"), "Mesa is required for customer-agent replay")
    def test_paired_offer_arms_and_replay(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = json.loads(DEFAULT_PROTOCOL.read_bytes())
            protocol.update(days=[181], seeds=[31])
            config = root / "pilot.json"
            config.write_text(json.dumps(protocol), encoding="utf-8")
            output = root / "result"
            result = run(config, DEFAULT_SCENARIO, output, "rule")
            self.assertEqual(result["evaluation_rows"], 27)
            self.assertEqual(verify(output), result)
            for line in (output / "traces.jsonl").read_text(encoding="utf-8").splitlines():
                trace = json.loads(line)
                for arms in trace["profiles"].values():
                    no_offer, zero, fixed = (arms[name] for name in ("no_offer", "zero_credit", "fixed_credit"))
                    schedules = [[(event["parcel_id"], event["locker_id"]) for event in arm["offers"]["events"]
                                  if event["kind"] == "offer_issued"] for arm in (zero, fixed)]
                    self.assertEqual(*schedules)
                    self.assertEqual(no_offer["offers"]["metrics"]["issued"], 0)
                    self.assertGreaterEqual(fixed["offers"]["metrics"]["accepted"], zero["offers"]["metrics"]["accepted"])
                    self.assertLessEqual(fixed["offers"]["metrics"]["spent_cents"], protocol["budget_cents"])
                    self.assertEqual(fixed["offers"]["metrics"]["outstanding_cents"], 0)

    @unittest.skipUnless(find_spec("ortools"), "OR-Tools is required for the CP-SAT allocator")
    def test_exact_allocator_reserves_scarce_slot_for_inflexible_parcel(self):
        candidates = (CandidateOffer("A", "east", 10), CandidateOffer("A", "west", 9), CandidateOffer("B", "east", 8))
        capacities = {"east": 1, "west": 1}
        self.assertEqual(len(allocate_rule(candidates, capacities, 200, 400)), 1)
        self.assertEqual(allocate_ortools(candidates, capacities, 200, 400), candidates[1:])


if __name__ == "__main__":
    unittest.main()
