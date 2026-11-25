from dataclasses import FrozenInstanceError
from importlib.util import find_spec
import json
import unittest
from unittest.mock import patch

from ecoagent_twin.m3_supervisor import Candidate, _graph, supervise


@unittest.skipUnless(find_spec("langgraph"), "LangGraph is required for supervisor execution")
class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.alert = {"id": "high-demand", "facts": {"eligible_lockers": ["east"]}}
        self.keep = Candidate("keep", "Keep the MAPPO policy", 4, 800, 2, 22)
        self.repair = Candidate("route_repair", "Run approved repair", 5, 900, 1, 22)
        self.menu = (self.keep, self.repair)

    def test_graph_replays_deterministic_choice(self):
        self.assertTrue(
            {"propose", "validate", "accept", "fallback"} <= _graph().get_graph().nodes.keys()
        )
        with patch("ecoagent_twin.m3_supervisor.perf_counter_ns", return_value=0):
            first = supervise(self.alert, self.menu)
            second = supervise(self.alert, tuple(reversed(self.menu)))
        self.assertEqual(first, second)
        self.assertEqual((first["status"], first["decision_id"]), ("selected", "route_repair"))
        self.assertFalse(first["fallback_used"])
        self.assertEqual(first["backend"], "deterministic")

    def test_model_can_pick_any_approved_id_from_the_read_only_menu(self):
        def choose(view, *, timeout_s):
            self.assertGreater(timeout_s, 0)
            self.assertLessEqual(timeout_s, 1)
            self.assertEqual(view.candidates, self.menu)
            with self.assertRaises(TypeError):
                view.alert["facts"]["eligible_lockers"] = []
            with self.assertRaises(FrozenInstanceError):
                view.candidates[0].id = "outside"
            payload = view.to_dict()
            self.assertEqual(json.loads(json.dumps(payload)), payload)
            payload["alert"]["facts"]["eligible_lockers"].append("west")
            return {"candidate_id": "keep", "reason": "No repair requested."}

        result = supervise(self.alert, self.menu, model=choose)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["decision_id"], "keep")
        self.assertEqual(result["reason"], "No repair requested.")
        self.assertEqual(self.alert["facts"]["eligible_lockers"], ["east"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["backend"], "model")

    def test_unknown_and_malformed_model_responses_use_fallback(self):
        for value in ("outside", {}, None, {"candidate_id": "keep", "reason": 3}):
            def choose(view, *, timeout_s):
                return value
            with self.subTest(value=value):
                result = supervise(self.alert, self.menu, model=choose)
                self.assertEqual(result["status"], "invalid_proposal")
                self.assertEqual(result["decision_id"], "route_repair")
                self.assertTrue(result["fallback_used"])

    def test_returned_proposal_after_deadline_is_rejected_even_if_approved(self):
        time_ns = 0

        def choose(view, *, timeout_s):
            nonlocal time_ns
            self.assertEqual(timeout_s, 0.5)
            time_ns = 700_000_000
            return "keep"

        with patch("ecoagent_twin.m3_supervisor.perf_counter_ns", side_effect=lambda: time_ns):
            result = supervise(self.alert, self.menu, model=choose, budget_s=0.5)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["decision_id"], "route_repair")
        self.assertEqual(result["proposed_id"], "keep")
        self.assertTrue(result["deadline_exceeded"])
        self.assertEqual(result["elapsed_ms"], 700)
        self.assertTrue(result["fallback_used"])

    def test_adapter_timeout_and_error_have_distinct_fallback_statuses(self):
        for error, status in (
            (TimeoutError("private detail"), "timeout"),
            (RuntimeError("private detail"), "backend_error"),
        ):
            def choose(view, *, timeout_s):
                raise error
            with self.subTest(status=status):
                result = supervise(self.alert, self.menu, model=choose)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["decision_id"], "route_repair")
                self.assertNotIn("private detail", result["reason"])
                self.assertTrue(result["fallback_used"])

    def test_rejects_duplicate_candidates_mixed_cohorts_and_invalid_budgets(self):
        wrong_cohort = Candidate("wrong", "Different cohort", 6, 1000, 1, 22)
        for menu, message in (
            ((), "menu"), ((self.keep, self.keep), "unique"),
            ((self.keep, wrong_cohort), "same parcel cohort"),
        ):
            with self.subTest(menu=menu), self.assertRaisesRegex(ValueError, message):
                supervise(self.alert, menu)
        for seconds in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "budget"):
                supervise(self.alert, self.menu, budget_s=seconds)


if __name__ == "__main__":
    unittest.main()
