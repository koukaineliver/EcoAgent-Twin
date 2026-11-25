from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ecoagent_twin.m6_decision_xai import LOGITS, RULE, _case_receipt, _source_cases, _validated_prefix, verify


CONTINUE, REPAIR = "continue_mappo", "route_repair"
DETERMINISTIC, AGENTIC = "deterministic_supervisor", "agentic_supervisor"


def _trace():
    return {
        "case_id": "case-a",
        "alert": {"kind": "locker_outage"},
        "menu": [
            {"id": CONTINUE, "label": "Continue", "served": 3, "unserved": 1,
             "travel_wh": 100, "restricted_recovery_min": 22},
            {"id": REPAIR, "label": "Repair", "served": 4, "unserved": 0,
             "travel_wh": 110, "restricted_recovery_min": 22},
        ],
        "selections": {
            DETERMINISTIC: {"decision_id": REPAIR, "status": "selected", "fallback_used": False,
                            "reason": "Ranked feasible plans."},
            AGENTIC: {"decision_id": CONTINUE, "status": "selected", "fallback_used": False,
                      "reason": "Scored both orders."},
        },
        "model_scores": {CONTINUE: 3.0, REPAIR: 2.0},
    }


class _TestScorer:
    """An explicit linear test game with archived full scores of 3 and 2."""

    def __call__(self, view, *, timeout_s):
        self.last_scores = {view.candidates[0].id: float(view.candidates[0].served),
                            view.candidates[1].id: float(view.candidates[1].served) * 0.5}


class DecisionAttributionTests(unittest.TestCase):
    def test_rule_and_model_have_distinct_games_and_reconstruct_the_full_decision(self):
        decisions = _case_receipt("M5", ("case-a", {}, _trace()), _TestScorer())["decisions"]
        rule = decisions[DETERMINISTIC]["attribution"]
        model = decisions[AGENTIC]["attribution"]
        self.assertEqual((rule["unit"], model["unit"]), (RULE, LOGITS))
        self.assertEqual((rule["baseline_margin"], rule["full_margin"]), (-1, 1))
        self.assertEqual(rule["shapley_values"], {"service": 2, "energy": 0, "recovery": 0})
        self.assertEqual(model["shapley_values"], {"service": 1, "energy": 0, "recovery": 0})
        self.assertIsNone(rule["full_score_verified"])
        self.assertTrue(model["full_score_verified"])
        self.assertTrue(rule["efficiency_verified"] and model["efficiency_verified"])

    def test_timeout_uses_rule_and_knows_planning_fallback_is_a_different_path(self):
        trace = _trace()
        trace["selections"][AGENTIC].update(decision_id=REPAIR, status="timeout", fallback_used=True)
        decisions = _case_receipt("M4", ("case-a", {}, trace))["decisions"]
        self.assertEqual(decisions[AGENTIC]["attribution"], decisions[DETERMINISTIC]["attribution"])
        trace["selections"][AGENTIC].update(decision_id=CONTINUE, status="planning_timeout")
        decision = _case_receipt("M4", ("case-a", {}, trace))["decisions"][AGENTIC]
        self.assertIsNone(decision["attribution"])
        self.assertEqual(decision["decisive_factor"], "planning_deadline")

    def test_inconsistent_archived_model_margin_is_rejected(self):
        trace = _trace()
        trace["model_scores"][CONTINUE] += 0.01
        with self.assertRaisesRegex(ValueError, "full coalition disagrees"):
            _case_receipt("M5", ("case-a", {}, trace), _TestScorer())

    def test_verify_binds_original_trace_and_rejects_altered_attribution(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            trace = json.dumps(_trace()).encode()
            source_file = run / "traces.jsonl"
            source_file.write_bytes(trace + b"\n")
            manifest = {"sha256": {"traces.jsonl": sha256(source_file.read_bytes()).hexdigest()}}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            _, cases = _source_cases("M5", run.resolve(), None)
            saved = _case_receipt("M5", cases[0], _TestScorer())
            output = run / "receipt.jsonl"
            output.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(verify("M5", run, output), {"cases": 1, LOGITS: 1, RULE: 1})
            self.assertEqual(saved["source"]["row_sha256"], sha256(trace).hexdigest())

            incomplete = run / "interrupted.jsonl"
            incomplete.write_text(json.dumps(saved) + "\n{", encoding="utf-8")
            self.assertEqual(_validated_prefix("M5", cases, incomplete), [saved])

            changed = deepcopy(saved)
            changed["decisions"][AGENTIC]["attribution"]["shapley_values"]["service"] += 1
            output.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                verify("M5", run, output)

            source_file.write_bytes(trace + b"\n\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                verify("M5", run, output)

    def test_empirical_case_identifier_must_match_bound_worlds_and_all_methods(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            observation = _trace()
            observation.pop("case_id")
            methods = ("preserve_observed_order", "frozen_mappo", DETERMINISTIC, AGENTIC)
            observation["rows"] = [{"case_id": "case-a", "method": method} for method in methods]
            artifacts = {"worlds.jsonl": {"snapshot": {"id": "case-a"}}, "observations.jsonl": observation}

            def write_sources():
                hashes = {}
                for name, row in artifacts.items():
                    raw = json.dumps(row).encode() + b"\n"
                    (run / name).write_bytes(raw)
                    hashes[name] = sha256(raw).hexdigest()
                (run / "manifest.json").write_text(json.dumps({"sha256": hashes}), encoding="utf-8")

            write_sources()
            _, cases = _source_cases("M6", run.resolve(), None)
            self.assertEqual(cases[0][0], "case-a")
            observation["rows"][0]["case_id"] = "mismatched"
            write_sources()
            with self.assertRaisesRegex(ValueError, "unmatched empirical case"):
                _source_cases("M6", run.resolve(), None)


if __name__ == "__main__":
    unittest.main()
