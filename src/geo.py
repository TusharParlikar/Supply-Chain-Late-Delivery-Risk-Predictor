"""Pincode geocoding, Delhivery center table and nearest-center search."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from paths import RAW_DIR

PINCODES_CSV = RAW_DIR / "india_pincodes.csv"
PINCODES_DATASET = "somesh24/indias-pincode-wise-districts"
EARTH_KM = 6371.0

CENTER_TYPES = {"HB", "H", "D", "I", "IP", "L", "DC", "DPC", "PC", "C", "DPP", "CP"}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def load_pincodes():
    """pin (6-digit str) -> latitude, longitude, place_name, state."""
    if not PINCODES_CSV.exists():
        import kagglehub

        PINCODES_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(kagglehub.dataset_download(PINCODES_DATASET)) / "IN.csv", PINCODES_CSV)
    p = pd.read_csv(PINCODES_CSV).dropna(subset=["latitude", "longitude"])
    p["pin"] = p["key"].str[3:]
    return p.rename(columns={"admin_name1": "state"})[["pin", "place_name", "state", "latitude", "longitude"]]


class PinLocator:
    """Exact pincode match, else the mean of its 3-digit sorting district."""

    def __init__(self, pins):
        self.exact = pins.set_index("pin")[["latitude", "longitude"]]
        self.district = pins.assign(d=pins["pin"].str[:3]).groupby("d")[["latitude", "longitude"]].mean()

    def locate(self, pin):
        pin = str(pin).strip()
        if len(pin) != 6 or not pin.isdigit():
            return None
        for table, key in ((self.exact, pin), (self.district, pin[:3])):
            if key in table.index:
                lat, lon = table.loc[key]
                return float(lat), float(lon)
        return None


def parse_center_name(name):
    """'Bhiwandi_Mankoli_HB (Maharashtra)' -> ('Bhiwandi', 'HB', 'Maharashtra')."""
    if not isinstance(name, str) or "(" not in name:
        return None, "Other", None
    body, state = name.rsplit(" (", 1)
    tokens = body.replace(" ", "_").split("_")
    ctype = next((t for t in reversed(tokens) if t in CENTER_TYPES), None)
    if ctype is None and tokens[-1].lower() == "hub":
        ctype = "HB"
    return tokens[0], ctype or "Other", state.rstrip(")")


def build_centers(raw, locator):
    """One row per Delhivery center: code, name, city, type, state, pincode, lat/lon."""
    both = pd.concat([
        raw[["source_center", "source_name"]].set_axis(["center", "name"], axis=1),
        raw[["destination_center", "destination_name"]].set_axis(["center", "name"], axis=1),
    ])
    c = both.dropna().drop_duplicates("center")
    c = pd.concat([c, both[~both["center"].isin(c["center"])].drop_duplicates("center")])
    parsed = c["name"].map(parse_center_name)
    c["city"] = parsed.str[0]
    c["ctype"] = parsed.str[1]
    c["state"] = parsed.str[2].fillna("Unknown")
    c["pincode"] = c["center"].str[3:9]
    c["geo_exact"] = c["pincode"].isin(locator.exact.index)
    loc = c["pincode"].map(locator.locate)
    c["latitude"] = loc.map(lambda v: v[0] if v else np.nan)
    c["longitude"] = loc.map(lambda v: v[1] if v else np.nan)
    # Last resort: average location of geocoded centers in the same state.
    for col in ("latitude", "longitude"):
        c[col] = c[col].fillna(c.groupby("state")[col].transform("mean"))
    return c.dropna(subset=["latitude", "longitude"]).set_index("center")


def nearest_centers(centers, lat, lon, k=5):
    d = haversine_km(lat, lon, centers["latitude"].values, centers["longitude"].values)
    return centers.assign(distance_km=d).nsmallest(k, "distance_km")


if __name__ == "__main__":
    # Delhi to Mumbai is about 1,150 km in a straight line.
    assert 1100 < haversine_km(28.6139, 77.2090, 19.0760, 72.8777) < 1200
    assert parse_center_name("Bhiwandi_Mankoli_HB (Maharashtra)") == ("Bhiwandi", "HB", "Maharashtra")
    assert parse_center_name("Mumbai Hub (Maharashtra)") == ("Mumbai", "HB", "Maharashtra")
    assert parse_center_name("Kanpur_Central_H_6 (Uttar Pradesh)") == ("Kanpur", "H", "Uttar Pradesh")
    assert parse_center_name("Haridwar (Uttarakhand)") == ("Haridwar", "Other", "Uttarakhand")
    loc = PinLocator(load_pincodes())
    assert loc.locate("110001") is not None and loc.locate("abc") is None and loc.locate("12345") is None
    print("geo checks passed")
