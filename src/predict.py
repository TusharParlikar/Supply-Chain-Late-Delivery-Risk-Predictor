"""Score new shipments given as pincodes: find nearest Delhivery centers, estimate the route, predict late risk."""
import json

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, add_features, load_centers
from geo import PinLocator, haversine_km, load_pincodes, nearest_centers
from paths import MODELS_DIR

REQUIRED_COLUMNS = ["source_pincode", "destination_pincode", "route_type", "departure_time"]


class Predictor:
    def __init__(self):
        self.pipe = joblib.load(MODELS_DIR / "xgb_pipeline.joblib")
        self.stats = json.loads((MODELS_DIR / "route_stats.json").read_text())
        self.centers = load_centers()
        self.locator = PinLocator(load_pincodes())
        legs = pd.read_parquet(MODELS_DIR / "scored_legs.parquet")
        self.history = legs.groupby(["source_center", "destination_center", "route_type"])[
            ["osrm_distance", "osrm_time"]].median()

    def nearest(self, pincode, k=5):
        """Nearest k centers to a pincode, or None if the pincode cannot be located."""
        loc = self.locator.locate(pincode)
        return None if loc is None else nearest_centers(self.centers, *loc, k=k)

    def score(self, requests):
        """requests: REQUIRED_COLUMNS, optionally source_center / destination_center to override the nearest.
        Returns one row per request with chosen centers, route estimate, risk_score and error."""
        rows = requests.reset_index(drop=True).copy()
        missing = [c for c in REQUIRED_COLUMNS if c not in rows]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        errors = pd.Series("", index=rows.index)

        for side in ("source", "destination"):
            pins = rows[f"{side}_pincode"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            rows[f"{side}_pincode"] = pins
            locs = pins.map(self.locator.locate)
            errors[locs.isna()] += f"{side} pincode not found; "
            chosen = rows.get(f"{side}_center", pd.Series(None, index=rows.index)).astype(object)
            chosen = chosen.where(chosen.isin(self.centers.index), None)
            for i, loc in locs.items():
                if loc is not None and pd.isna(chosen[i]):
                    chosen[i] = nearest_centers(self.centers, *loc, k=1).index[0]
            rows[f"{side}_center"] = chosen
            c = self.centers.reindex(chosen)
            lat = locs.map(lambda v: v[0] if v else np.nan)
            lon = locs.map(lambda v: v[1] if v else np.nan)
            rows[f"{side}_center_km"] = haversine_km(lat, lon, c["latitude"].values, c["longitude"].values).round(1)

        rows["route_type"] = rows["route_type"].astype(str).str.strip()
        errors[~rows["route_type"].isin(self.stats)] += f"route_type must be one of {sorted(self.stats)}; "
        rows["od_start_time"] = pd.to_datetime(rows["departure_time"], errors="coerce")
        errors[rows["od_start_time"].isna()] += "bad departure_time; "
        rows["error"] = errors.str.rstrip("; ")

        ok = rows["error"] == ""
        rows["risk_score"] = np.nan
        if ok.any():
            feats = self.estimate_route(add_features(rows[ok], self.centers))
            for col in ("source_state", "destination_state", "center_distance_km", "osrm_distance", "osrm_time",
                        "route_from_history"):
                rows.loc[ok, col] = feats[col].values
            rows.loc[ok, "risk_score"] = self.pipe.predict_proba(feats[FEATURES])[:, 1]
        return rows

    def estimate_route(self, feats):
        """OSRM distance/time: median of past legs on the same center pair, else a distance-based estimate."""
        key = pd.MultiIndex.from_frame(feats[["source_center", "destination_center", "route_type"]])
        past = self.history.reindex(key)
        factor = feats["route_type"].map(lambda rt: self.stats[rt]["road_factor"])
        speed = feats["route_type"].map(lambda rt: self.stats[rt]["km_per_min"])
        same_point = (feats["center_distance_km"] < 0.5).values
        guess_km = np.where(same_point, feats["route_type"].map(lambda rt: self.stats[rt]["same_point_km"]),
                            feats["center_distance_km"].values * factor.values)
        guess_min = np.where(same_point, feats["route_type"].map(lambda rt: self.stats[rt]["same_point_min"]),
                             guess_km / speed.values)
        feats = feats.copy()
        feats["route_from_history"] = past["osrm_distance"].notna().values
        feats["osrm_distance"] = np.where(feats["route_from_history"], past["osrm_distance"].values, guess_km)
        feats["osrm_time"] = np.where(feats["route_from_history"], past["osrm_time"].values, guess_min)
        return feats


if __name__ == "__main__":
    p = Predictor()
    req = pd.DataFrame({
        "source_pincode": ["421302", "110001", "999999", "560001"],
        "destination_pincode": ["400011", "560001", "110001", "600001"],
        "route_type": ["FTL", "FTL", "FTL", "Bike"],
        "departure_time": ["2018-10-01 08:00", "2018-10-01 22:00", "2018-10-01 08:00", "2018-10-01 08:00"],
    })
    out = p.score(req)
    print(out[["source_pincode", "source_center", "destination_center", "osrm_distance", "osrm_time",
               "route_from_history", "risk_score", "error"]].to_string())
    assert out.loc[:1, "risk_score"].between(0, 1).all() and out.loc[:1, "error"].eq("").all()
    assert "source pincode not found" in out.loc[2, "error"] and pd.isna(out.loc[2, "risk_score"])
    assert "route_type" in out.loc[3, "error"]
    print("predict checks passed")
