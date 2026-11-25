from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ecoagent_twin.m2_sumo_experiment import (
    DEFAULT_PROTOCOL, DEFAULT_SCENARIO, measure, replay, verify_measurement, verify_replay,
)
from ecoagent_twin.m2_sumo_runner import read_trips


class SumoTripTests(unittest.TestCase):
    def test_round_delay_and_gross_energy_without_crediting_regeneration(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tripinfo.xml"
            path.write_text(
                '<tripinfos><tripinfo id="probe" depart="10" arrival="70" duration="60" '
                'departDelay="0.1" routeLength="500"><battery depleted="0" '
                'totalEnergyConsumed="10.1" totalEnergyRegenerated="9.3"/></tripinfo></tripinfos>',
                encoding="utf-8",
            )
            probes = {"probe": ("A", "B")}
            row, = read_trips(path, probes)
            self.assertEqual((row["travel_min"], row["travel_wh"], row["regenerated_wh"]), (2, 11, 9.3))
            with self.assertRaisesRegex(ValueError, "Missing completed SUMO probes"):
                read_trips(path, {**probes, "absent": ("B", "A")})

    @unittest.skipUnless(find_spec("sumo"), "Eclipse SUMO binaries required")
    def test_sumo_outputs_feed_shared_fleet_replay(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            measured = measure(DEFAULT_SCENARIO, root / "measure", 47)
            self.assertEqual(measured["completed_od_probes"], {"light_traffic": 42, "heavy_traffic": 42})
            self.assertEqual(verify_measurement(root / "measure"), measured)
            self.assertGreater(measured["changed_travel_min_pairs"], 0)
            protocol = json.loads(DEFAULT_PROTOCOL.read_bytes())
            protocol.update(days=[221], seeds=[41])
            config = root / "protocol.json"
            config.write_text(json.dumps(protocol), encoding="utf-8")
            result = replay(root / "measure", config, root / "replay", checkpoint_run=None)
            self.assertEqual((result["matched_outage_snapshots"], result["controller_episodes"]), (3, 6))
            self.assertEqual(verify_replay(root / "replay"), result)


if __name__ == "__main__":
    unittest.main()
