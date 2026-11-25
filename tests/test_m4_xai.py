from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ecoagent_twin.m4_xai import explain, main, render_markdown


def candidate_record(served, travel_wh, charged_wh):
    events = [
        {"kind": "locker_outage", "time_min": 20, "locker_id": "central"},
        {"kind": "offer_slot_preassigned", "time_min": 22, "parcel_id": "p1", "locker_id": "east"},
        {"kind": "routing_started", "time_min": 22},
    ]
    if charged_wh:
        events.append({"kind": "charging_completed", "time_min": 24, "charged_wh": charged_wh})
    events.extend(
        {
            "kind": "parcel_deposited", "time_min": 25 + index,
            "parcel_id": f"p{index}", "locker_id": "east" if index == 1 else "north",
        }
        for index in range(1, served + 1)
    )
    return {
        "actions": [],
        "result": {
            "metrics": {
                "total_parcels": 4, "served_parcels": served, "unserved_parcels": 4 - served,
                "service_rate": served / 4, "recovery_95_min": None, "travel_min": 12,
                "travel_wh": travel_wh, "charged_wh": charged_wh,
            },
            "events": events,
            "unserved_parcel_ids": [f"p{index}" for index in range(served + 1, 5)],
        },
    }


def saved_case(case_id):
    mappo = candidate_record(2, 800, 100)
    repair = candidate_record(3, 850, 0)
    return {
        "case_id": case_id,
        "traffic": "heavy_traffic",
        "severity": "severe",
        "chaos_kind": "compound",
        "alerts": [
            {"kind": "locker_outage", "time_min": 20, "payload": {"locker_id": "central"}},
            {"kind": "locker_saturation", "time_min": 20, "payload": {
                "capacity_changes": [{"locker_id": "west", "free_slots_before": 2, "free_slots_after": 1}],
            }},
        ],
        "alert": {
            "alert_min": 20, "dispatch_min": 22, "horizon_min": 42,
            "accepted_reassignments": 1, "offer_budget_cents": 400, "credit_spent_cents": 200,
        },
        "offers": {
            "assignments": {"p1": "east"},
            "events": [
                {"kind": "offer_issued", "time_min": 20, "parcel_id": "p1", "locker_id": "east"},
                {"kind": "offer_accepted", "time_min": 21, "parcel_id": "p1", "locker_id": "east",
                 "paid_cents": 200},
            ],
            "metrics": {
                "issued": 1, "accepted": 1, "declined": 0, "expired": 0, "reassignments": 1,
                "spent_cents": 200, "budget_cents": 400,
            },
        },
        "fallbacks": [
            {"parcel_id": "p4", "mode": "deferred_home_delivery", "available_min": 55},
        ],
        "candidates": {"continue_mappo": mappo, "route_repair": repair},
        "menu": [
            {"id": "continue_mappo", "label": "Continue MAPPO", "served": 2,
             "unserved": 2, "travel_wh": 800, "restricted_recovery_min": 22},
            {"id": "route_repair", "label": "Repair route", "served": 3,
             "unserved": 1, "travel_wh": 850, "restricted_recovery_min": 22},
        ],
        "repair_receipt": {
            "engine": "ortools_cp_sat", "strategy": "cp_sat", "solver_status": "OPTIMAL",
            "columns_per_vehicle_limit": 64, "fallback_reason": None,
            "certified_metrics": deepcopy(repair["result"]["metrics"]),
        },
        "model_scores": {"continue_mappo": 1.5, "route_repair": 0.1},
        "mappo_alone": {
            "status": "selected", "decision_id": "continue_mappo", "end_to_end_decision_ms": 7.5,
        },
        "selections": {
            "deterministic_supervisor": {
                "status": "selected", "decision_id": "route_repair", "fallback_used": False,
                "elapsed_ms": 1.5, "reason": "Private-looking string.",
            },
            "agentic_supervisor": {
                "status": "selected", "decision_id": "continue_mappo", "fallback_used": False,
                "elapsed_ms": 200.0, "reason": "Private-looking string.",
            },
        },
    }


class XAITest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)

    def write(self, *rows):
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (self.run / "traces.jsonl").write_bytes(payload)
        (self.run / "manifest.json").write_text(
            json.dumps({"sha256": {"traces.jsonl": sha256(payload).hexdigest()}}),
            encoding="utf-8",
        )

    def test_explains_selected_case_with_verified_scope_and_outcome_delta(self):
        self.write(saved_case("first"), saved_case("second"))
        evidence = explain(self.run, "second")
        self.assertEqual(evidence["case"]["case_id"], "second")
        self.assertEqual(evidence["source"]["trace_line"], 2)
        self.assertEqual(evidence["agentic_minus_deterministic"], {
            "served": -1, "travel_wh": -50, "charged_wh": 100, "restricted_recovery_min": 0,
        })
        markdown = render_markdown(evidence)
        self.assertIn("west went from 2 to 1 free slots", markdown)
        self.assertIn("Paid **200** of **400 cents**", markdown)
        self.assertIn("held a slot at east", markdown)
        self.assertIn("Operational fallback for parcel p4", markdown)
        self.assertIn("**-1** on-time parcels", markdown)
        self.assertIn("Agentic minus MAPPO alone was **+0** on-time parcels", markdown)
        self.assertNotIn("Private-looking string", markdown)
        self.assertEqual(explain(self.run)["case"]["case_id"], "first")

    def test_rejects_modified_trace_and_nonunique_case_ids(self):
        case = saved_case("case")
        self.write(case)
        with (self.run / "traces.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            explain(self.run)
        self.write(case, case)
        with self.assertRaisesRegex(ValueError, "duplicate case ID"):
            explain(self.run)

    def test_refingerprinted_inconsistencies_do_not_receive_an_explanation(self):
        for changed, message in (
            (("menu", 0, "served", 4), "menu disagrees"),
            (("offers", "metrics", "spent_cents", 0), "offer commitments"),
            (("selections", "agentic_supervisor", "decision_id", "other"), "invalid recorded"),
            (("repair_receipt", "certified_metrics", "travel_wh", 900), "repair receipt"),
        ):
            with self.subTest(changed=changed):
                row = saved_case("case")
                record = row
                for key in changed[:-2]:
                    record = record[key]
                record[changed[-2]] = changed[-1]
                self.write(row)
                with self.assertRaisesRegex(ValueError, message):
                    explain(self.run)

    def test_timeout_fallback_distinguishes_executed_and_proposed_actions(self):
        row = saved_case("case")
        row["selections"]["agentic_supervisor"].update(
            status="timeout", decision_id="route_repair", proposed_id="continue_mappo",
            fallback_used=True,
        )
        self.write(row)
        evidence = explain(self.run)
        self.assertTrue(evidence["selections"]["agentic_supervisor"]["fallback_used"])
        self.assertEqual(evidence["agentic_minus_deterministic"]["served"], 0)
        self.assertIn("timeout, deterministic fallback", render_markdown(evidence))

    def test_preselection_timeout_keeps_mappo_and_records_planning_time(self):
        row = saved_case("case")
        row["preselection_planning_ms"] = 2200.0
        for selection in row["selections"].values():
            selection.update(
                status="planning_timeout", decision_id="continue_mappo", fallback_used=True,
                elapsed_ms=0.0, end_to_end_decision_ms=2200.0,
            )
        self.write(row)
        evidence = explain(self.run)
        self.assertEqual(evidence["agentic_minus_mappo"]["served"], 0)
        report = render_markdown(evidence)
        self.assertIn("MAPPO continuation fallback", report)
        self.assertIn("0.00 ms selection, 2200.00 ms end to end", report)

    def test_json_cli_and_exclusive_output_file(self):
        self.write(saved_case("case"))
        output = self.run / "explanation.json"
        with patch("sys.argv", ["m4_xai", str(self.run), "--format", "json"]):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main()
        self.assertEqual(json.loads(stdout.getvalue())["case"]["case_id"], "case")
        with patch("sys.argv", ["m4_xai", str(self.run), "--output", str(output)]):
            main()
            original = output.read_bytes()
            self.assertEqual(json.loads(original)["case"]["case_id"], "case")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
