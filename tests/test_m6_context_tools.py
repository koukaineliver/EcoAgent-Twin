"""Verify clock, traffic evidence and certified-menu gates without remote calls."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import unittest

from ecoagent_twin.m6_context_tools import _run_graph, _updated_world, _weather_receipt


def _world() -> dict:
    return {
        "schema": 1, "data_basis": "synthetic", "id": "fixture", "depot": "depot",
        "outage": {"time_min": 0, "locker_id": "locker"},
        "horizon_min": 20, "service_min_per_parcel": 1, "locker_free_capacity": {"locker": 0},
        "vehicles": [{"id": "ev", "node": "depot", "battery_wh": 100, "reserve_wh": 5}],
        "parcels_in_original_plan_order": [{"id": "p", "vehicle_id": "ev", "original_locker": "locker",
                                            "alternatives": [], "deadline_min": 10}],
        "travel_minutes": {"depot": {"depot": 0, "locker": 1}, "locker": {"depot": 1, "locker": 0}},
        "travel_wh": {"depot": {"depot": 0, "locker": 30}, "locker": {"depot": 30, "locker": 0}},
    }


def _traffic() -> tuple[dict, dict]:
    network = {
        "access_depot_out": ("depot", "j"), "access_depot_in": ("j", "depot"),
        "access_locker_out": ("locker", "j"), "access_locker_in": ("j", "locker"),
    }
    return {
        "status": "available", "version": "SUMO 1.28.0", "seed": 3, "simulation_second": 180,
        "edges": {
            edge: {"travel_time_s": 20.0, "mean_speed_m_s": 4.0, "vehicle_count": 1}
            for edge in network
        },
        "routes": [{
            "origin": a, "destination": b, "edges": [f"access_{a}_out", f"access_{b}_in"],
            "sumo_route_time_s": seconds,
        } for a, b, seconds in (("depot", "locker", 110), ("locker", "depot", 40))],
    }, network


class ContextToolsTests(unittest.TestCase):
    def test_weather_rejects_stale_or_wrong_units(self):
        payload = {
            "utc_offset_seconds": 0,
            "current": {"time": 1800, "temperature_2m": 17.5, "precipitation": 0,
                        "weather_code": 2, "wind_speed_10m": 2.2},
            "current_units": {"time": "unixtime", "temperature_2m": "°C", "precipitation": "mm",
                              "weather_code": "wmo code", "wind_speed_10m": "km/h"},
        }
        fetched = datetime.fromtimestamp(2000, tz=timezone.utc)
        valid = _weather_receipt(json.dumps(payload).encode(), fetched)
        self.assertEqual(valid["role"], "advisory_only")
        self.assertEqual(valid["values"]["weather_code"], 2)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            _weather_receipt(json.dumps(payload).encode(), datetime.fromtimestamp(7000, tz=timezone.utc))
        payload["current_units"]["precipitation"] = "inch"
        with self.assertRaisesRegex(ValueError, "units"):
            _weather_receipt(json.dumps(payload).encode(), fetched)

    def test_verified_measured_delay_changes_directed_world_only(self):
        raw = _world()
        before = deepcopy(raw)
        evidence, network = _traffic()
        updated, changes = _updated_world(raw, evidence, network, 3, 180)
        self.assertEqual(raw, before)
        self.assertEqual(updated["travel_minutes"]["depot"]["locker"], 2)
        self.assertEqual(updated["travel_wh"]["depot"]["locker"], 36)
        self.assertEqual(updated["travel_minutes"]["locker"]["depot"], 1)
        self.assertEqual(len(changes), 1)

    def test_missing_route_or_wrong_clock_cannot_update(self):
        evidence, network = _traffic()
        with self.assertRaisesRegex(ValueError, "clock"):
            _updated_world(_world(), evidence, network, 3, 181)
        evidence["routes"].pop()
        with self.assertRaisesRegex(ValueError, "missing travel arc"):
            _updated_world(_world(), evidence, network, 3, 180)

    def test_standstill_proxy_is_past_the_window_without_extrapolated_energy(self):
        evidence, network = _traffic()
        evidence["routes"][0]["sumo_route_time_s"] = 100_000.0
        updated, changes = _updated_world(_world(), evidence, network, 3, 180)
        self.assertEqual(updated["travel_minutes"]["depot"]["locker"], 21)
        self.assertEqual(updated["travel_wh"]["depot"]["locker"], 147)
        self.assertTrue(changes[0]["outside_dispatch_window"])

    def test_live_weather_unavailability_is_advisory_and_traffic_failure_holds(self):
        def unavailable():
            raise TimeoutError

        def unexpected(_):
            self.fail("an uncertified menu must not be selected")

        held = _run_graph(unavailable, unavailable, unexpected, unexpected)
        self.assertEqual([item["status"] for item in held["tools"]], ["unavailable", "held"])
        self.assertNotIn("decision", held)
        chosen = _run_graph(
            unavailable, lambda: {"status": "available"}, lambda _: {"status": "certified"},
            lambda _: {"status": "selected"},
        )
        self.assertEqual([item["tool"] for item in chosen["tools"]], [
            "get_weather", "query_traffic_twin", "certify_route_repair", "choose_approved_route",
        ])

    def test_invalid_plan_never_reaches_chooser(self):
        def plan(_):
            raise ValueError("synthetic malformed matrix")

        def choose(_):
            self.fail("the chooser received an uncertified matrix")

        held = _run_graph(lambda: {"status": "available"}, lambda: {"status": "available"}, plan, choose)
        self.assertEqual(held["plan"]["status"], "held")
        self.assertNotIn("decision", held)


if __name__ == "__main__":
    unittest.main()
