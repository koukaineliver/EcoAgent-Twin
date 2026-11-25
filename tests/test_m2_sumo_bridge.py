from copy import deepcopy
import json
from pathlib import Path
import unittest

from ecoagent_twin.m2_sumo_bridge import apply_travel_matrix
from ecoagent_twin.model import Scenario


FIXTURE = (
    Path(__file__).resolve().parents[1] / "ecoagent_twin" / "scenarios" / "locker_outage.json"
)


class SumoBridgeTest(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        nodes = sorted(self.raw["travel_minutes"])
        self.trips = [
            {
                "origin": origin,
                "destination": destination,
                "travel_min": 0 if origin == destination else 1 + (2 * i + j) % 5,
                "travel_wh": 0 if origin == destination else 50 + 10 * i + j,
            }
            for i, origin in enumerate(nodes)
            for j, destination in enumerate(nodes)
        ]

    def trip_index(self, origin, destination):
        return next(
            index for index, trip in enumerate(self.trips)
            if (trip["origin"], trip["destination"]) == (origin, destination)
        )

    def test_replaces_both_directed_matrices_without_mutating_inputs(self):
        original = deepcopy(self.raw)
        original_trips = deepcopy(self.trips)
        result = apply_travel_matrix(self.raw, self.trips)

        self.assertEqual(self.raw, original)
        self.assertEqual(self.trips, original_trips)
        unchanged = self.raw.keys() - {"travel_minutes", "travel_wh"}
        self.assertEqual(
            {key: result[key] for key in unchanged}, {key: original[key] for key in unchanged}
        )
        travel_wh = result["travel_wh"]
        self.assertNotEqual(travel_wh["depot"]["east"], travel_wh["east"]["depot"])
        scenario = Scenario.from_dict(result)
        for trip in self.trips:
            arc = trip["origin"], trip["destination"]
            self.assertEqual(scenario.travel_time[arc], trip["travel_min"])
            self.assertEqual(scenario.travel_energy[arc], trip["travel_wh"])

        reversed_result = apply_travel_matrix(self.raw, list(reversed(self.trips)))
        self.assertEqual(json.dumps(result), json.dumps(reversed_result))
        result["vehicles"][0]["battery_wh"] = 0
        self.assertEqual(self.raw, original)

    def test_rejects_incomplete_duplicate_and_unknown_pairs(self):
        missing = self.trip_index("depot", "east")
        for trips, message in (
            (self.trips[:missing] + self.trips[missing + 1 :], "missing travel arc"),
            ([*self.trips, self.trips[0]], "duplicate travel arc"),
            ([*self.trips, {**self.trips[0], "destination": "East"}], "unexpected travel arc"),
            ([{}, *self.trips[1:]], "each trip must provide"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                apply_travel_matrix(self.raw, trips)

    def test_fills_unprobed_diagonal_values_with_zero(self):
        off_diagonal = [trip for trip in self.trips if trip["origin"] != trip["destination"]]
        explicit_zeros = apply_travel_matrix(self.raw, self.trips)
        self.assertEqual(len(off_diagonal), 42)
        self.assertEqual(apply_travel_matrix(self.raw, off_diagonal), explicit_zeros)
        self.assertEqual(
            apply_travel_matrix(self.raw, [*off_diagonal, self.trips[0]]), explicit_zeros
        )

    def test_rejects_invalid_units_and_nonzero_diagonals(self):
        index = self.trip_index("depot", "east")
        for field in ("travel_min", "travel_wh"):
            for value, message in (
                (1.5, "must be integers"), (False, "must be integers"),
                (0, "must be positive"), (-1, "must be positive"),
            ):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    altered = list(self.trips)
                    altered[index] = {**altered[index], field: value}
                    apply_travel_matrix(self.raw, altered)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "diagonal"):
                altered = list(self.trips)
                diagonal = self.trip_index("depot", "depot")
                altered[diagonal] = {**altered[diagonal], field: 1}
                apply_travel_matrix(self.raw, altered)

    def test_validates_feasible_return_after_replacement(self):
        index = self.trip_index("start-1", "depot")
        for field, value in (("travel_min", 23), ("travel_wh", 1_501)):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "return deadline or battery"),
            ):
                altered = list(self.trips)
                altered[index] = {**altered[index], field: value}
                apply_travel_matrix(self.raw, altered)


if __name__ == "__main__":
    unittest.main()
