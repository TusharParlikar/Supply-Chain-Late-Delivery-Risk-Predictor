"""Load the DataCo CSV and build the leakage-safe feature table."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
KAGGLE_DATASET = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"

TARGET = "Late_delivery_risk"

# Known only after the order ships, or a direct restatement of the target.
LEAKAGE = [
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status",
    "shipping date (DateOrders)",
]

# Everything here is known when the order is placed.
CATEGORICAL = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Category Name",
    "Department Name",
    "Customer Segment",
    "Type",
]
NUMERIC = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Product Price",
    "Sales",
    "Latitude",
    "Longitude",
    "order_month",
    "order_dayofweek",
    "order_hour",
]
FEATURES = CATEGORICAL + NUMERIC


def download():
    """Fetch the CSV from Kaggle into data/raw/ if it is not there yet."""
    if RAW_CSV.exists():
        return RAW_CSV
    import shutil

    import kagglehub

    src = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / RAW_CSV.name
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, RAW_CSV)
    return RAW_CSV


def load_raw():
    return pd.read_csv(download(), encoding="latin-1")


def build(df):
    """Return (X, y, df_with_order_date_parts). X holds only order-time columns."""
    df = df.copy()
    order_date = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
    df["order_date"] = order_date
    df["order_month"] = order_date.dt.month
    df["order_dayofweek"] = order_date.dt.dayofweek
    df["order_hour"] = order_date.dt.hour

    X = df[FEATURES]
    assert not set(LEAKAGE) & set(X.columns), "leakage column in features"
    assert TARGET not in X.columns
    return X, df[TARGET], df


if __name__ == "__main__":
    df = load_raw()
    X, y, _ = build(df)
    print("rows, cols:", df.shape)
    print("late rate:", round(y.mean(), 4), "| late:", int(y.sum()), "| not late:", int((1 - y).sum()))
    for col in ["Market", "Order Region", "Shipping Mode", "Category Name"]:
        print(f"{col}: {df[col].nunique()} unique")
    print("feature matrix:", X.shape)
