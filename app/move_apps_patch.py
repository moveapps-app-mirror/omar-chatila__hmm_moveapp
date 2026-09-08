import numpy as np
import pandas as pd
import movingpandas as mpd
import geopandas as gpd
import movingpandas.trajectory_collection as mpd_tc
import movingpandas.trajectory as mpd_traj

_APPLIED = False

def _force_object_series(s):
    # kill pandas StringDtype / ArrowStringArray reliably
    arr = np.asarray(s.to_list(), dtype=object)
    return pd.Series(arr, index=s.index, name=s.name, dtype=object)

def force_tc_id_object_inplace(tc):
    id_col = tc.get_traj_id_col()

    for tr in tc.trajectories:
        if id_col in tr.df.columns:
            tr.df[id_col] = _force_object_series(tr.df[id_col])

def apply_moveapps_id_dtype_patch():
    global _APPLIED
    if _APPLIED:
        return

    pd.options.mode.string_storage = "python"
    try:
        pd.options.future.infer_string = False
    except Exception:
        pass

    # Patch Trajectory._set_traj_id_column
    orig_set = mpd_traj.Trajectory._set_traj_id_column
    def patched_set(self, traj_id_col):
        orig_set(self, traj_id_col)
        if traj_id_col and traj_id_col in self.df.columns:
            self.df[traj_id_col] = _force_object_series(self.df[traj_id_col])
    mpd_traj.Trajectory._set_traj_id_column = patched_set
    mpd.Trajectory._set_traj_id_column = patched_set  # safety

    # Patch TrajectoryCollection.to_point_gdf
    orig_to_point = mpd_tc.TrajectoryCollection.to_point_gdf
    def patched_to_point_gdf(self, *args, **kwargs):
        gdf = orig_to_point(self, *args, **kwargs)
        id_col = self.get_traj_id_col()
        if id_col in gdf.columns:
            gdf[id_col] = _force_object_series(gdf[id_col])
        return gdf
    mpd_tc.TrajectoryCollection.to_point_gdf = patched_to_point_gdf
    mpd.TrajectoryCollection.to_point_gdf = patched_to_point_gdf
    _APPLIED = True


def restore_original_dtypes_inplace(original_tc, result_tc):
    """Restore dtypes for input columns that are also present in the result."""
    original = original_tc.to_point_gdf()
    original_dtypes = original.dtypes.to_dict()

    for trajectory in result_tc.trajectories:
        for column, dtype in original_dtypes.items():
            if column not in trajectory.df.columns or column == "geometry":
                continue
            if dtype == object:
                trajectory.df[column] = _force_object_series(trajectory.df[column])
            else:
                trajectory.df[column] = trajectory.df[column].astype(dtype)

    force_tc_id_object_inplace(result_tc)
    return result_tc

def debug_patch_state():
    import movingpandas.trajectory_collection as mpd_tc
    import movingpandas as mpd
    print("mpd_tc.TrajectoryCollection.to_point_gdf =", mpd_tc.TrajectoryCollection.to_point_gdf.__name__)
    print("mpd.TrajectoryCollection.to_point_gdf    =", mpd.TrajectoryCollection.to_point_gdf.__name__)


def merge_traj_collections(original_tc, result_gdf, fill_method="ffill", nearest_tolerance=None):
    orig = original_tc.to_point_gdf().copy()
    res = result_gdf.copy()

    traj_id_col = original_tc.get_traj_id_col()
    t_col = orig.index.name

    # reset index -> time column _t
    orig = orig.reset_index().rename(columns={t_col: "_t"})
    if t_col in res.columns:
        res = res.rename(columns={t_col: "_t"})
    else:
        res = res.reset_index().rename(columns={res.index.name: "_t"})

    # ensure datetime
    orig["_t"] = pd.to_datetime(orig["_t"])
    res["_t"]  = pd.to_datetime(res["_t"])

    # ensure object dtypes
    orig[traj_id_col] = _force_object_series(orig[traj_id_col])
    res[traj_id_col]  = _force_object_series(res[traj_id_col])

    tol = pd.Timedelta(nearest_tolerance) if nearest_tolerance is not None else None
    direction = "backward" if fill_method == "ffill" else "nearest"
    if fill_method not in ("ffill", "nearest"):
        raise ValueError("fill_method must be 'ffill' or 'nearest'")

    # merge per id
    value_cols = [c for c in orig.columns if c not in ["_t", "geometry"]]
    merged_parts = []

    for tid, res_part in res.groupby(traj_id_col, sort=False):
        orig_part = orig[orig[traj_id_col] == tid]
        if orig_part.empty:
            tmp = res_part[["_t", traj_id_col, "geometry"]].copy()
            for c in value_cols:
                if c not in tmp.columns:
                    tmp[c] = None
            merged_parts.append(tmp)
            continue

        res_part  = res_part.sort_values("_t")
        orig_part = orig_part.sort_values("_t")

        tmp = pd.merge_asof(
            res_part[["_t", traj_id_col, "geometry"]],
            orig_part[[traj_id_col, "_t"] + [c for c in value_cols if c != traj_id_col]],
            by=traj_id_col,
            on="_t",
            direction=direction,
            tolerance=tol,
        )
        merged_parts.append(tmp)

    merged = pd.concat(merged_parts, ignore_index=True)

    # force object again after concat/merge
    merged[traj_id_col] = _force_object_series(merged[traj_id_col])

    # build gdf + tc
    merged = merged.rename(columns={"_t": t_col})
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=original_tc.get_crs())
    gdf[t_col] = pd.to_datetime(gdf[t_col])
    gdf = gdf.set_index(t_col)

    # restore old column-order
    orig_cols = original_tc.to_point_gdf().columns
    gdf = gdf.reindex(columns=orig_cols)

    return mpd.TrajectoryCollection(
        gdf,
        traj_id_col=traj_id_col,
        t=gdf.index.name,
        crs=original_tc.get_crs(),
    )
