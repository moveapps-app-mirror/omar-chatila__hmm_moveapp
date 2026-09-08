"""MoveApps entry point for behavioural-state annotation."""

from __future__ import annotations

import logging

import pandas as pd
from movingpandas import TrajectoryCollection

from hmmcma import HMM, CriterionAnnotator
from sdk.moveapps_io import MoveAppsIo
from sdk.moveapps_spec import hook_impl

from app.configuration import parse_features, parse_rules


LOGGER = logging.getLogger(__name__)


class App:
    def __init__(self, moveapps_io: MoveAppsIo):
        self.moveapps_io = moveapps_io

    @hook_impl
    def execute(self, data: TrajectoryCollection, config: dict) -> TrajectoryCollection:
        """Annotate input fixes and return them with a numeric ``state`` column."""
        method = str(config.get("method", "hmm")).strip().lower()
        num_states = _positive_integer(config.get("num_states", 3), "num_states")

        if method == "hmm":
            features = parse_features(config.get("features", "1,3"))
            annotator = HMM(features=features, num_states=num_states)
            LOGGER.info(
                "Running HMM with %s states and features: %s",
                num_states,
                ", ".join(feature.value for feature in features),
            )
        elif method == "criteria":
            rules = parse_rules(config.get("criteria_rules", ""), num_states)
            annotator = CriterionAnnotator(rules)
            LOGGER.info(
                "Running criteria-based segmentation with %s states: %s",
                num_states,
                ", ".join(rule.label for rule in rules),
            )
        else:
            raise ValueError("Unknown method. Choose 'hmm' or 'criteria'.")

        result = annotator.annotate(data)
        result = _normalise_unmodelled(result, method, _as_boolean(config.get("lfill", False)))

        plot_path = self.moveapps_io.create_artifacts_file("state_plot.png")
        annotator.plot(plot_path, annotation=result, title="Behavioural states")
        LOGGER.info("Saved state plot artifact to %s", plot_path)
        return result.trajectory_collection


def _positive_integer(value, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _as_boolean(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1", "yes", "on"}:
            return True
        if normalised in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _normalise_unmodelled(result, method: str, lfill: bool):
    """Use -1 for unmodelled points and optionally carry states forward per track."""
    collection = result.trajectory_collection
    points = collection.to_point_gdf().copy()
    id_col = collection.get_traj_id_col()
    states = pd.to_numeric(points["state"], errors="coerce").fillna(-1).astype(int)

    # CriterionAnnotator reserves zero for its "unclassified" result. HMM zero
    # is a valid model state and must therefore remain untouched.
    if method == "criteria":
        states = states.mask(states == 0, -1)
        if "segment_name" in points.columns:
            points.loc[states < 0, "segment_name"] = pd.NA

    if lfill:
        modelled = states.mask(states < 0)
        states = modelled.groupby(points[id_col], sort=False, dropna=False).ffill().fillna(-1).astype(int)
        if "segment_name" in points.columns:
            names = points["segment_name"].mask(states < 0)
            points["segment_name"] = names.groupby(points[id_col], sort=False, dropna=False).ffill()

    points["state"] = states
    points["cluster"] = states
    result.labels = states.to_numpy()
    result.metadata.setdefault("state_names", {})[-1] = "unmodelled"
    result.metadata["lfill"] = lfill
    result.trajectory_collection = TrajectoryCollection(points, traj_id_col=id_col)
    return result
