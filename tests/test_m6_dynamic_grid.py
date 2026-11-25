from datetime import datetime, timedelta, timezone
import unittest

from ecoagent_twin.m6_dynamic_grid import Interval, _energy_intervals, _hour_energy


class HourlyGridTests(unittest.TestCase):
    def test_hour_ending_split_preserves_energy(self):
        start = datetime(2018, 7, 12, 20, 45, tzinfo=timezone.utc)
        charge = Interval(start, start + timedelta(minutes=30), 1200, "in_route")
        first_end = start.replace(hour=21, minute=0)
        self.assertEqual(list(_hour_energy(charge)), [
            (first_end, 600), (first_end + timedelta(hours=1), 600),
        ])

    def test_saved_charge_excess_remains_inventory(self):
        world = {
            "metadata": {"projected_alert_utc": "2018-07-12T20:45:00+00:00"},
            "snapshot": {"horizon_min": 75},
            "charger": {"charger_rate_wh_per_min": 800},
        }
        events = [
            {"kind": "charging_started", "time_min": 10, "vehicle_id": "v"},
            {"kind": "charging_completed", "time_min": 12, "vehicle_id": "v", "charged_wh": 1600},
        ]
        row = {"travel_wh": 800, "carrier_fallback_traction_wh": 400, "onboard_recharge_wh": 1600,
               "initial_battery_wh": 1000, "terminal_battery_wh": 1800}
        interval, = _energy_intervals(world, events, row)
        self.assertEqual((interval.kind, interval.wh), ("in_route", 1200))
        self.assertEqual(interval.start, datetime(2018, 7, 12, 20, 55, tzinfo=timezone.utc))
        self.assertEqual(interval.end - interval.start, timedelta(minutes=2))
        residual, = _energy_intervals(world, [], {**row, "onboard_recharge_wh": 0, "terminal_battery_wh": 200})
        self.assertEqual((residual.kind, residual.wh), ("post_horizon", 1200))
        self.assertEqual(residual.start, datetime(2018, 7, 12, 22, 15, tzinfo=timezone.utc))
        self.assertEqual(residual.end - residual.start, timedelta(minutes=1.5))


if __name__ == "__main__":
    unittest.main()
