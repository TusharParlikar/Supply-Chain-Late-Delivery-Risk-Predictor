"""Load the Delhivery data, roll it up to trip legs and build the leakage-safe feature table."""
import shutil
from pathlib import Path

import pandas as pd

from geo import PinLocator, build_centers, haversine_km, load_pincodes
from paths import RAW_DIR

RAW_CSV = RAW_DIR / "delhivery_data.csv"
KAGGLE_DATASET = "santanukundu/delhivery-dataset"

# A leg is late when it takes more than this many times the OSRM route-planner estimate.
LATE_FACTOR = 2.5
TARGET = "late"
LEG_KEY = ["trip_uuid", "source_center", "destination_center"]

# Known only after the leg finishes, or computed from the outcome.
LEAKAGE = [
    "actual_time",
    "od_end_time",
    "start_scan_to_end_scan",
    "actual_distance_to_destination",
    "factor",
    "segment_actual_time",
    "segment_osrm_time",
    "segment_osrm_distance",
    "segment_factor",
    "is_cutoff",
    "cutoff_factor",
    "cutoff_timestamp",
]

# Everything here is known when the leg is dispatched.
CATEGORICAL = [
    "route_type",
    "source_state",
    "destination_state",
    "source_type",
    "destination_type",
    "source_center",
    "destination_center",
]
NUMERIC = [
    "osrm_time",
    "osrm_distance",
    "center_distance_km",
    "same_state",
    "start_hour",
    "start_dayofweek",
]
FEATURES = CATEGORICAL + NUMERIC


def download():
    if not RAW_CSV.exists():
        import kagglehub

        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(kagglehub.dataset_download(KAGGLE_DATASET)) / RAW_CSV.name, RAW_CSV)
    return RAW_CSV


def load_raw():
    return pd.read_csv(download(), parse_dates=["trip_creation_time", "od_start_time", "od_end_time"])


def load_centers(raw=None):
    return build_centers(load_raw() if raw is None else raw, PinLocator(load_pincodes()))


def to_legs(raw):
    """One row per trip leg. Time and distance columns are cumulative within a leg, so take the max."""
    legs = raw.groupby(LEG_KEY, as_index=False).agg(
        split=("data", "first"),
        route_type=("route_type", "first"),
        od_start_time=("od_start_time", "first"),
        osrm_time=("osrm_time", "max"),
        osrm_distance=("osrm_distance", "max"),
        actual_time=("actual_time", "max"),
    )
    legs[TARGET] = (legs["actual_time"] > LATE_FACTOR * legs["osrm_time"]).astype(int)
    return legs


def add_features(rows, centers):
    """Add center, distance and time features. rows needs source_center, destination_center, od_start_time."""
    rows = rows.copy()
    for side in ("source", "destination"):
        info = centers.reindex(rows[f"{side}_center"])
        rows[f"{side}_state"] = info["state"].fillna("Unknown").values
        rows[f"{side}_type"] = info["ctype"].fillna("Other").values
        rows[f"{side}_lat"] = info["latitude"].values
        rows[f"{side}_lon"] = info["longitude"].values
    rows["center_distance_km"] = haversine_km(rows["source_lat"], rows["source_lon"],
                                              rows["destination_lat"], rows["destination_lon"])
    rows["same_state"] = (rows["source_state"] == rows["destination_state"]).astype(int)
    rows["start_hour"] = rows["od_start_time"].dt.hour
    rows["start_dayofweek"] = rows["od_start_time"].dt.dayofweek
    return rows


def build(raw=None):
    """Return (X, y, legs). X holds only dispatch-time columns."""
    raw = load_raw() if raw is None else raw
    legs = add_features(to_legs(raw), load_centers(raw))
    X = legs[FEATURES]
    assert not set(LEAKAGE) & set(X.columns), "leakage column in features"
    assert TARGET not in X.columns
    return X, legs[TARGET], legs


if __name__ == "__main__":
    raw = load_raw()
    X, y, legs = build(raw)
    centers = load_centers(raw)
    print("rows, cols:", raw.shape)
    print("trips:", raw["trip_uuid"].nunique(), "| legs:", len(legs), "| centers:", len(centers))
    print("late rate:", round(y.mean(), 4), "| late:", int(y.sum()), "| on time:", int((1 - y).sum()))
    print("legs by split:", legs["split"].value_counts().to_dict())
    print("states:", centers["state"].nunique(), "| route types:", legs["route_type"].unique().tolist())
    print("centers with exact pincode match:", round(centers["pincode"].isin(load_pincodes()["pin"]).mean(), 3))
    print("feature matrix:", X.shape, "| missing distance:", int(X["center_distance_km"].isna().sum()))
