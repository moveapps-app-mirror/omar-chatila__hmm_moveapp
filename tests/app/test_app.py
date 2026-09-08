import os
import unittest
from tempfile import TemporaryDirectory

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd

from app.app import App, _normalise_unmodelled
from app.configuration import parse_features, parse_rules
from hmmcma import AnnotationResult, Feature
from sdk.moveapps_io import MoveAppsIo


def trajectory_collection():
    times = pd.date_range("2026-01-01", periods=12, freq="min", name="timestamp")
    x = np.r_[np.arange(4), np.arange(4) + 50, np.arange(4) + 54].astype(float)
    points = gpd.GeoDataFrame(
        {
            "animal_id": ["fox"] * len(times),
            "utm_x": x,
            "utm_y": np.zeros(len(times)),
        },
        geometry=gpd.points_from_xy(x, np.zeros(len(times))),
        index=times,
        crs="EPSG:32632",
    )
    return mpd.TrajectoryCollection(points, traj_id_col="animal_id")


class AppTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = TemporaryDirectory()
        self.addCleanup(self.artifacts.cleanup)
        os.environ["APP_ARTIFACTS_DIR"] = self.artifacts.name
        self.sut = App(moveapps_io=MoveAppsIo())

    def test_hmm_adds_state_and_plot_artifact(self):
        output = self.sut.execute(
            trajectory_collection(),
            {"method": "hmm", "num_states": 2, "features": "1,3", "lfill": False},
        )

        points = output.to_point_gdf()
        self.assertIn("state", points)
        self.assertEqual(len(points), 12)
        self.assertTrue(set(points.state).issubset({-1, 0, 1}))
        self.assertGreater(os.path.getsize(os.path.join(self.artifacts.name, "state_plot.png")), 0)

    def test_criteria_method_uses_cpp_rules_and_adds_state(self):
        output = self.sut.execute(
            trajectory_collection(),
            {
                "method": "criteria",
                "num_states": 2,
                "criteria_rules": "resting:disk=5; moving:!disk=5",
                "lfill": False,
            },
        )

        points = output.to_point_gdf()
        self.assertTrue({"state", "segment_name", "segment_id", "change_point"}.issubset(points.columns))
        self.assertNotIn(0, set(points.state))
        self.assertGreater(os.path.getsize(os.path.join(self.artifacts.name, "state_plot.png")), 0)

    def test_lfill_is_per_trajectory_and_does_not_fill_leading_points(self):
        times = pd.date_range("2026-01-01", periods=6, freq="min", name="timestamp")
        points = gpd.GeoDataFrame(
            {
                "animal_id": ["a"] * 3 + ["b"] * 3,
                "state": [-1, 2, -1, -1, 4, -1],
                "cluster": [-1, 2, -1, -1, 4, -1],
                "segment_id": [-1, 0, -1, -1, 1, -1],
                "change_point": [False] * 6,
                "utm_x": range(6),
                "utm_y": [0] * 6,
            },
            geometry=gpd.points_from_xy(range(6), [0] * 6),
            index=times,
            crs="EPSG:32632",
        )
        collection = mpd.TrajectoryCollection(points, traj_id_col="animal_id")
        result = AnnotationResult(collection, np.asarray(points.state), {}, {})

        actual = _normalise_unmodelled(result, "hmm", True).trajectory_collection.to_point_gdf()

        self.assertEqual(actual.state.tolist(), [-1, 2, 2, -1, 4, 4])


class ConfigurationTest(unittest.TestCase):
    def test_feature_numbers_are_parsed_and_deduplicated(self):
        self.assertEqual(parse_features("1, 3 1"), (Feature.SPEED, Feature.TURN_ANGLE))

    def test_unknown_feature_number_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown feature number"):
            parse_features("1,9")

    def test_compound_criteria_and_state_count_are_parsed(self):
        rules = parse_rules(
            "resting:disk=100&minimum_duration=3600; moving:speed_with_outliers=5,1",
            2,
        )

        self.assertEqual([rule.label for rule in rules], ["resting", "moving"])

    def test_criteria_rule_count_must_match_states(self):
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            parse_rules("resting:disk=100; moving:speed=5", 3)

    def test_mixed_boolean_operators_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "mixes"):
            parse_rules("state:disk=100&minimum_hops=2|heading=30", 1)


if __name__ == "__main__":
    unittest.main()
