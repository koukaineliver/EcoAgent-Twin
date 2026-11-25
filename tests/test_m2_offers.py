import json
import math
from pathlib import Path
import unittest

from ecoagent_twin.m2_offers import OfferLedger, OfferOption, UtilityProfile, paired_draw
from ecoagent_twin.model import Locker, Scenario


class UtilityTests(unittest.TestCase):
    def test_incremental_attributes_and_money_units(self):
        profile = UtilityProfile(-1.0, 2.0, 0.5, 0.25)
        option = OfferOption("P101", "east", 4, -8, 50)
        # Net credit is one currency unit. Extra pickup time and reduced delay cancel.
        self.assertAlmostEqual(profile.accept_probability(option, 150), 1 / (1 + math.exp(-1)))
        self.assertLess(profile.accept_probability(option, 50), profile.accept_probability(option, 150))
        self.assertEqual(UtilityProfile(1000, 0, 0, 0).accept_probability(option, 0), 1)
        self.assertEqual(UtilityProfile(-1000, 0, 0, 0).accept_probability(option, 0), 0)

    def test_shared_draw_is_stable_and_keyed_by_case_and_parcel(self):
        draw = paired_draw(1, 5, "P101")
        self.assertGreaterEqual(draw, 0.0)
        self.assertLess(draw, 1.0)
        self.assertEqual(draw, paired_draw("1", 5, "P101"))
        self.assertNotEqual(draw, paired_draw(2, 5, "P101"))
        self.assertNotEqual(draw, paired_draw(1, 6, "P101"))
        self.assertNotEqual(draw, paired_draw(1, 5, "P103"))
        self.assertNotEqual(draw, paired_draw(1, 5, "P101", scenario="capacity_pressure"))


class OfferLedgerTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "ecoagent_twin/scenarios/locker_outage.json"
        self.scenario = Scenario.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.ledger = OfferLedger(self.scenario, budget_cents=100)
        self.east1 = OfferOption("P101", "east", 0, 0, 0)
        self.east2 = OfferOption("P201", "east", 0, 0, 0)
        self.west = OfferOption("P103", "west", 0, 0, 0)

    def test_shared_holds_and_acceptance_accounting(self):
        self.assertTrue(self.ledger.issue(self.east1, 60, 20, 25))
        self.assertFalse(self.ledger.can_issue(self.east2, 0))
        self.assertFalse(self.ledger.issue(self.east2, 0, 20, 25))
        self.assertFalse(self.ledger.issue(self.west, 50, 20, 25))
        self.assertEqual(self.ledger.result()["metrics"]["outstanding_cents"], 60)
        self.assertEqual(self.ledger.resolve("P101", now_min=21, accepted=False), "declined")
        self.assertFalse(self.ledger.can_issue(self.east1, 0))
        self.assertTrue(self.ledger.issue(self.east2, 70, 21, 25))
        self.assertTrue(self.ledger.issue(self.west, 30, 21, 25))
        self.assertEqual(self.ledger.resolve("P201", now_min=21, accepted=True), "accepted")
        self.assertEqual(self.ledger.resolve("P103", now_min=24, accepted=True), "accepted")
        self.assertEqual(self.ledger.resolve("P103", now_min=24, accepted=True), "denied")
        result = self.ledger.result()
        self.assertEqual(result["assignments"], {"P201": "east", "P103": "west"})
        self.assertEqual(result["metrics"], {
            "budget_cents": 100, "spent_cents": 100, "outstanding_cents": 0,
            "issued": 3, "accepted": 2, "declined": 1, "expired": 0, "reassignments": 2,
        })
        self.assertEqual([event["reason"] for event in result["events"] if "reason" in event], [
            "locker_full", "budget_exceeded", "offer_already_resolved",
        ])

    def test_expiry_releases_slot_and_credit_and_denies_late_response(self):
        self.assertTrue(self.ledger.issue(self.east1, 100, 20, 22))
        self.assertEqual(self.ledger.expire(21), 0)
        self.assertEqual(self.ledger.resolve("P101", now_min=22, accepted=True), "expired")
        self.assertEqual(self.ledger.result()["events"][-1]["reason"], "after_expiry")
        self.assertTrue(self.ledger.issue(self.east2, 100, 22, 23))
        self.assertEqual(self.ledger.expire(26), 1)
        self.assertEqual(self.ledger.result()["events"][-1]["time_min"], 23)
        metrics = self.ledger.result()["metrics"]
        self.assertEqual((metrics["expired"], metrics["spent_cents"], metrics["outstanding_cents"]), (2, 0, 0))

    def test_revalidation_denies_closed_locker_without_payment(self):
        self.assertTrue(self.ledger.issue(self.east1, 100, 20, 25))
        self.scenario.lockers["east"] = Locker(capacity=1, available=False)
        self.assertEqual(self.ledger.resolve("P101", now_min=21, accepted=True), "denied")
        result = self.ledger.result()
        self.assertEqual(result["events"][-1]["reason"], "locker_unavailable")
        self.assertEqual(result["assignments"], {})
        self.assertEqual(result["metrics"]["spent_cents"], 0)
        self.assertEqual(result["metrics"]["outstanding_cents"], 0)
        self.assertEqual(result["metrics"]["accepted"], 1)
        self.assertEqual(result["metrics"]["reassignments"], 0)

    def test_eligibility_and_invalid_timestamp(self):
        unaffected = OfferOption("P102", "east", 0, 0, 0)
        wrong_locker = OfferOption("P201", "west", 0, 0, 0)
        self.assertFalse(self.ledger.can_issue(unaffected, 0))
        self.assertFalse(self.ledger.can_issue(wrong_locker, 0))
        self.assertFalse(self.ledger.issue(self.east1, 0, 20, 20))
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            self.ledger.issue(self.east1, 100, 19, 24)
        self.assertEqual(self.ledger.result()["metrics"]["issued"], 0)


if __name__ == "__main__":
    unittest.main()
