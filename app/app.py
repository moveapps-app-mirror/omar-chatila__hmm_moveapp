"""MoveApps entry point for behavioural-state annotation."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from movingpandas import TrajectoryCollection
from pyproj import CRS

from app.move_apps_patch import (
    apply_moveapps_id_dtype_patch,
    restore_original_dtypes_inplace,
)

from hmmcma import HMM, CriterionAnnotator
from sdk.moveapps_io import MoveAppsIo
from sdk.moveapps_spec import hook_impl

from app.configuration import parse_features, parse_rules


LOGGER = logging.getLogger(__name__)
apply_moveapps_id_dtype_patch()


class App:
    def __init__(self, moveapps_io: MoveAppsIo):
        self.moveapps_io = moveapps_io

    @hook_impl
    def execute(self, data: TrajectoryCollection, config: dict) -> TrajectoryCollection:
        """Annotate input fixes and return them with a numeric ``state`` column."""
        data = _add_utm(data)
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


def _add_utm(data: TrajectoryCollection) -> TrajectoryCollection:
    """Add per-animal WGS84 UTM coordinates without changing the geometry."""
    annotated = deepcopy(data)
    for trajectory in annotated.trajectories:
        points = trajectory.df.copy()
        if points.crs is None:
            raise ValueError("Every trajectory needs a CRS for UTM conversion.")
        if points.empty:
            continue

        lonlat = points.to_crs("EPSG:4326")
        valid = lonlat.geometry.notna() & ~lonlat.geometry.is_empty
        if not valid.any():
            raise ValueError(f"Trajectory {trajectory.id!r} has no valid geometry for UTM conversion.")

        # Find the centre of the shortest longitude arc. This remains correct
        # for tracks crossing the antimeridian, unlike a plain bounding-box centre.
        longitudes = np.sort(np.mod(lonlat.loc[valid].geometry.x.to_numpy(), 360.0))
        wrapped = np.append(longitudes, longitudes[0] + 360.0)
        largest_gap = int(np.argmax(np.diff(wrapped)))
        arc_start = wrapped[largest_gap + 1]
        arc_length = 360.0 - (wrapped[largest_gap + 1] - wrapped[largest_gap])
        centre_lon = float((arc_start + 0.5 * arc_length) % 360.0)
        if centre_lon > 180.0:
            centre_lon -= 360.0
        latitudes = lonlat.loc[valid].geometry.y
        centre_lat = float(0.5 * (latitudes.min() + latitudes.max()))

        zone = min(60, max(1, int((centre_lon + 180.0) // 6.0) + 1))
        utm_crs = CRS.from_epsg((32600 if centre_lat >= 0 else 32700) + zone)
        projected = points.to_crs(utm_crs)
        points["utm_x"] = projected.geometry.x
        points["utm_y"] = projected.geometry.y
        points["utm_crs"] = utm_crs.to_string()
        trajectory.df = points

    restore_original_dtypes_inplace(data, annotated)
    return annotated

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
