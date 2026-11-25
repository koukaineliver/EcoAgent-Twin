import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ecoagent_twin.m1_benchmark import DEFAULT_PROTOCOL, DEFAULT_SCENARIO, run, verify


class M1BenchmarkTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "run"
        self.protocol_path = self.root / "protocol.json"
        self.protocol = json.loads(DEFAULT_PROTOCOL.read_bytes())
        self.protocol.update(development_days=[1], evaluation_days=[2], seeds=[1, 2])

    def write_protocol(self):
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")

    def test_matched_runs_score_censoring_and_replay(self):
        self.write_protocol()
        report = run(self.protocol_path, DEFAULT_SCENARIO, self.output)
        self.assertEqual((report["matched_snapshots"], report["episode_rows"]), (12, 36))
        self.assertEqual(verify(self.output), report)
        with (self.output / "episodes.csv").open(newline="", encoding="utf-8") as stream:
            episodes = list(csv.DictReader(stream))
        censored = [row for row in episodes if not row["recovery_minutes"]]
        observed = [row for row in episodes if row["recovery_minutes"]]
        self.assertTrue(censored and observed)
        self.assertTrue(all(row["restricted_recovery_minutes"] == "22" for row in censored))
        self.assertTrue(all(row["restricted_recovery_minutes"] == row["recovery_minutes"] for row in observed))

    def test_verifier_detects_event_change_even_with_updated_file_hash(self):
        self.write_protocol()
        run(self.protocol_path, DEFAULT_SCENARIO, self.output)
        cases_path = self.output / "cases.jsonl"
        lines = cases_path.read_text(encoding="utf-8").splitlines()
        case = json.loads(lines[0])
        case["controllers"]["keep_plan"]["events"][0]["time_min"] += 1
        lines[0] = json.dumps(case)
        cases_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["sha256"]["cases.jsonl"] = sha256(cases_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "saved events do not replay"):
            verify(self.output)

    def test_development_and_evaluation_days_cannot_overlap(self):
        self.protocol["evaluation_days"] = [1]
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "day IDs overlap"):
            run(self.protocol_path, DEFAULT_SCENARIO, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
