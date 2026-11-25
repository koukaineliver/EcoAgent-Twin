from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ecoagent_twin.m2_experiment import DEFAULT_PROTOCOL, DEFAULT_SCENARIO, _worlds, run, verify


class M2ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(DEFAULT_PROTOCOL.read_bytes())
        self.base = json.loads(DEFAULT_SCENARIO.read_bytes())

    def test_rejects_overlap_between_training_and_evaluation_day_blocks(self):
        self.protocol["partitions"]["evaluation"]["days"] = self.protocol["partitions"]["training"]["days"]
        with self.assertRaisesRegex(ValueError, "day IDs overlap"):
            _worlds(self.protocol, self.base)

    @unittest.skipUnless(find_spec("torch"), "PyTorch is required for the MAPPO round trip")
    def test_trains_checkpoint_and_replays_matched_episodes(self):
        for partition in self.protocol["partitions"].values():
            partition["days"] = partition["days"][:1]
            partition["seeds"] = partition["seeds"][:1]
        self.protocol["training"].update(policy_seeds=[731], updates=1, episodes_per_update=2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "protocol.json"
            config.write_text(json.dumps(self.protocol), encoding="utf-8")
            output = root / "run"
            result = run(config, DEFAULT_SCENARIO, output)
            self.assertEqual(result["evaluation_episode_rows"], 6)
            self.assertTrue((output / "checkpoints" / "mappo_731.pt").is_file())
            self.assertEqual(verify(output), result)


if __name__ == "__main__":
    unittest.main()
