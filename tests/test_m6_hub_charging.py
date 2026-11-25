from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from functools import partial
from importlib.util import find_spec
import json
from pathlib import Path
import unittest

from ecoagent_twin.m6_hub_charging import Hub, HubOutageEnv, carbon, energy_ledger, replay, search
from ecoagent_twin.m6_hub_charging_experiment import (
    M5_ALERT_UTC, _lines, _project, _run_source, _state,
)
from ecoagent_twin.m3_cases import digest, read_json
from ecoagent_twin.model import Scenario


M5_RUN = Path(__file__).resolve().parents[1] / "runs" / "m5_eval_20261125T005646766426Z"


def archived_queue_contention(m5_run: Path = M5_RUN) -> dict:
    """Replay a prescribed simultaneous booking under one and two matched plugs."""
    _run_source(m5_run, ("traces.jsonl", "offer_protocol.json"))
    protocol = read_json(m5_run / "offer_protocol.json")
    panel = []
    for trace in _lines(m5_run / "traces.jsonl"):
        world = trace["world"]
        if ((world["fleet_size"], world["traffic_seed"], world["severity"]) != (2, 301, "severe")
                or world["chaos_kind"] not in ("locker_outage", "charger_outage")):
            continue
        scenario, settings, offers, fallback, alert = _state(world, protocol)
        assert (offers, fallback, alert) == (trace["offers"], trace["fallbacks"], trace["alert"])
        case = _project(scenario, settings, offers, M5_ALERT_UTC - timedelta(minutes=scenario.outage_min),
                        study="m5_queue", efficiency=0.90, nominal_rate=480, stress=True, panel={})
        arms = {}
        for plugs in (1, 2):
            factory = partial(HubOutageEnv, case.scenario, **case.settings,
                              hubs=(Hub(case.hub, 480, plugs),), max_battery_wh=case.battery_cap)
            env = factory()
            actions = []
            while (view := env.observation()) is not None:
                if len(actions) < 2:
                    index = next(i for i, action in enumerate(view.actions) if action.kind == "charge")
                else:
                    index = next((i for i, action in enumerate(view.actions) if action.kind == "deposit"),
                                 len(view.actions) - 1)
                actions.append({"vehicle_id": view.vehicle_id, "time_min": view.time_min,
                                "action": asdict(view.actions[index])})
                env.step(index)
            result = env.result()
            assert replay(factory, actions) == result
            booked = [event for event in result["events"] if event["kind"] == "charge_reserved"]
            assert len(booked) == 2 and all(event["hub_id"] == case.hub for event in booked)
            assert actions[0]["time_min"] == actions[1]["time_min"]
            ledger = energy_ledger(result, case.scenario, terminal_target_wh=case.terminal_target,
                                   terminal_rate_wh_per_min=case.registered_rate)
            arms[str(plugs)] = {"actions": actions, "result": result, "charging_intervals": ledger,
                               "queue_min": sum(event["queue_wait_min"] for event in booked)}
        assert arms["1"]["queue_min"] > 0 and arms["2"]["queue_min"] == 0
        panel.append({"case_id": case.scenario.id, "UTC_alert_clock": M5_ALERT_UTC.isoformat(),
                      "modeled_hub": case.hub, "shared_dispatch_settings": case.settings, "arms": arms})
    assert len(panel) == 4
    return {"schema": "m6_archived_queue_contention_v1", "scope": "modeled 480 Wh/min locker hub; prescribed paired stress replay",
            "m5_manifest_sha256": digest(m5_run / "manifest.json"),
            "source_sha256": digest(Path(__file__)), "rows": panel}


class HubChargingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "ecoagent_twin" / "scenarios" / "locker_outage.json"
        scenario = Scenario.from_dict(json.loads(path.read_bytes()))
        cls.scenario = replace(scenario, vehicles=tuple(replace(v, battery_wh=800) for v in scenario.vehicles))

    def test_booking_queue_and_energy_reconciliation(self):
        environment = HubOutageEnv(self.scenario, hubs=(Hub("west", 480),), max_battery_wh=2000)
        for vehicle in self.scenario.vehicles:
            view = environment.observation()
            self.assertEqual(view.vehicle_id, vehicle.id)
            action = next(i for i, item in enumerate(view.actions) if item.kind == "charge")
            environment.step(action)
        while (view := environment.observation()) is not None:
            environment.step(len(view.actions) - 1)
        result = environment.result()
        reservations = [event for event in result["events"] if event["kind"] == "charge_reserved"]
        self.assertEqual(len(reservations), 2)
        self.assertGreater(reservations[1]["queue_wait_min"], 0)
        self.assertEqual(reservations[0]["completion_min"], reservations[1]["start_min"])
        self.assertGreaterEqual(min(event["arrival_soc_wh"] for event in reservations), 500)
        ledger = energy_ledger(result, self.scenario, terminal_target_wh=2000, terminal_rate_wh_per_min=480)
        self.assertEqual(sum(row["battery_wh"] for row in ledger), result["metrics"]["travel_wh"] + 2 * 1200)
        self.assertTrue(all(row["hub_id"] == "west" for row in ledger if row["kind"] == "in_window"))

    def test_closed_hub_and_search_budget(self):
        with self.assertRaisesRegex(ValueError, "available locker"):
            HubOutageEnv(self.scenario, hubs=(Hub("central", 240),), max_battery_wh=2000)

        def factory():
            return HubOutageEnv(self.scenario, hubs=(Hub("west", 480),), max_battery_wh=2000)

        def objective(result):
            return -result["metrics"]["served_parcels"], result["metrics"]["travel_wh"]

        plan = search(factory, objective, transition_limit=1)
        self.assertFalse(plan["exact_in_simulator"])
        self.assertLessEqual(plan["transitions"], 1)
        self.assertEqual(plan["result"], replay(factory, plan["actions"]))

    def test_carbon_splits_hour_and_accounts_for_losses(self):
        origin = datetime(2018, 7, 16, 18, 50, tzinfo=timezone.utc)
        ledger = [{"start_min": 5, "end_min": 15, "battery_wh": 1000, "kind": "in_window"}]
        end = origin.replace(minute=0) + timedelta(hours=1)
        self.assertAlmostEqual(carbon(ledger, origin, {end: 0.1, end + timedelta(hours=1): 0.5}, 0.75), 0.4)

    @unittest.skipUnless(M5_RUN.is_dir() and find_spec("ortools") and find_spec("mesa"),
                         "requires the archived M5 run, OR-Tools and Mesa")
    def test_archived_queue_contention(self):
        result = archived_queue_contention()
        self.assertEqual(sorted(row["arms"]["1"]["queue_min"] for row in result["rows"]), [2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
