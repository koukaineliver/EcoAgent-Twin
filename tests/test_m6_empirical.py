from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import patch
import unittest

from ecoagent_twin.empirical_amazon import _ObjectPrefix
from ecoagent_twin.empirical_energy import PROFILES, carbon_receipt, leg_wh, road_proxy_km
from ecoagent_twin.m6_empirical_benchmark import make_case
from ecoagent_twin.model import Scenario, evaluate_route


class EmpiricalTests(unittest.TestCase):
    def test_official_nonstandard_nan_normalizes_in_chunked_parser(self):
        text = b'{ "Route1":{"alpha":NaN},"Route2":{"coordinates":[0.0,1.5]}}'
        with patch("ecoagent_twin.empirical_amazon.CHUNK", 7):
            source = _ObjectPrefix(BytesIO(text))
            self.assertEqual(list(source.records()), [
                ("Route1", {"alpha": None}), ("Route2", {"coordinates": [0, 1.5]}),
            ])
        self.assertEqual(source.bytes, len(text))

    def test_operational_carbon_reconciles_stored_and_grid_energy(self):
        profile = PROFILES[0]
        inputs = {
            "traction_wh": 800, "fallback_wh": 500, "initial_wh": 6000, "all_car_pickup_km": 1.5,
            "profile": profile,
        }
        no_charge = carbon_receipt(**inputs, charged_wh=0, terminal_wh=5200)
        charged = carbon_receipt(**inputs, charged_wh=1400, terminal_wh=6600)
        self.assertEqual(no_charge["total_grid_generation_kwh"], charged["total_grid_generation_kwh"])
        self.assertAlmostEqual(charged["total_grid_generation_kwh"],
                               charged["in_window_grid_generation_kwh"] +
                               charged["inventory_adjustment_and_fallback_grid_generation_kwh"])
        self.assertGreater(charged["operational_kg_co2e_all_car"],
                           charged["operational_kg_co2e_at_assumed_share"])
        with self.assertRaisesRegex(ValueError, "reconcile"):
            carbon_receipt(**inputs, charged_wh=1400, terminal_wh=6500)

    def test_window_uses_observed_order_directed_travel_and_safe_return(self):
        keys = ["station", *[f"drop{i}" for i in range(1, 13)]]
        stops = {}
        packages = {}
        for rank, stop in enumerate(keys):
            stops[stop] = {
                "type": "Dropoff" if rank else "Station", "lat": 33.9 + rank / 200, "lng": -118.2,
            }
            packages[stop] = {} if rank == 0 else {
                f"p{rank}-{i}": {
                    "planned_service_time_seconds": 30,
                    "time_window": {
                        "start_time_utc": None, "end_time_utc": "2018-06-17 08:44:00" if rank == 5 else None,
                    },
                }
                for i in range(3)
            }
        travel = {
            first: {second: 0 if i == j else 61 + i * 2 + j for j, second in enumerate(keys)}
            for i, first in enumerate(keys)
        }
        record = {
            "route_id": "fixture", "route": {
                "station_code": "DLA1", "date_YYYY_MM_DD": "2018-06-17", "departure_time_utc": "08:00:00",
                "stops": dict(reversed(list(stops.items()))),
            },
            "packages": packages, "travel_seconds": travel,
            "actual_sequence": {"actual": dict(zip(keys, range(len(keys)), strict=True))},
        }
        profile = PROFILES[0]
        raw, _, meta = make_case(record, profile, "low_soc_at_alert")
        scenario = Scenario.from_dict(raw)
        self.assertEqual(list(meta["source_stops"].values()), keys[:1] + keys[4:9])
        self.assertEqual(len(scenario.parcels), 8)
        self.assertEqual(meta["estimated_directed_travel_seconds_from_source"]["site-1"]["site-2"],
                         travel["drop5"]["drop6"])
        self.assertEqual(scenario.travel_energy["start", "depot"],
                         leg_wh(road_proxy_km((33.92, -118.2), (33.9, -118.2)),
                                travel["drop4"]["station"], profile))
        remaining = evaluate_route(scenario, scenario.vehicles[0], ()).remaining_wh
        self.assertEqual(remaining - scenario.vehicles[0].reserve_wh, 300)
        self.assertLess(datetime.fromisoformat(meta["projected_alert_utc"]),
                        datetime(2018, 6, 17, 8, 44, tzinfo=timezone.utc))
        self.assertLess(scenario.parcels[0].deadline_min, scenario.horizon_min)


if __name__ == "__main__":
    unittest.main()
