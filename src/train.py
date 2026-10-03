"""Train and compare Logistic Regression, Random Forest and XGBoost."""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CATEGORICAL, NUMERIC, build, load_centers
from paths import MODELS_DIR, REPORTS_DIR

SEED = 42


def preprocessor():
    # Centers seen fewer than 20 times share one "infrequent" column; unseen centers land there too.
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
         CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ], verbose_feature_names_out=False)


def models(pos_weight):
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1, random_state=SEED,
        ),
    }


def route_stats(legs):
    """Medians the app uses to estimate OSRM distance and time for a new shipment."""
    # Centers geocoded to the same district point show 0 km apart; keep their typical real route instead.
    same_point = legs[legs["center_distance_km"] < 0.5].groupby("route_type")[["osrm_distance", "osrm_time"]].median()
    # Only legs whose two centers have exact pincode coordinates; district-level guesses are too coarse.
    exact = set(load_centers().query("geo_exact").index)
    legs = legs[legs["source_center"].isin(exact) & legs["destination_center"].isin(exact)]
    stats = {}
    for rt, g in legs.groupby("route_type"):
        g = g[g["center_distance_km"] > 5]
        stats[rt] = {
            "road_factor": float((g["osrm_distance"] / g["center_distance_km"]).median()),
            "km_per_min": float((g["osrm_distance"] / g["osrm_time"]).median()),
            "same_point_km": float(same_point.loc[rt, "osrm_distance"]),
            "same_point_min": float(same_point.loc[rt, "osrm_time"]),
        }
    return stats


def main():
    X, y, legs = build()
    # Kaggle's split is by time: training trips run Sep 12-26 2018, test trips Sep 27-Oct 3.
    train = (legs["split"] == "training").values
    X_train, X_test, y_train, y_test = X[train], X[~train], y[train], y[~train]
    assert not set(legs.loc[train, "trip_uuid"]) & set(legs.loc[~train, "trip_uuid"])
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"train {len(X_train)}, test {len(X_test)}, scale_pos_weight {pos_weight:.3f}", flush=True)

    rows, fitted = [], {}
    for name, clf in models(pos_weight).items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", clf)]).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })
        fitted[name] = pipe
        print(rows[-1], flush=True)

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    table = pd.DataFrame(rows).round(4).sort_values("roc_auc", ascending=False)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    # XGBoost is the model SHAP explains and the dashboard serves.
    best = fitted["XGBoost"]
    joblib.dump(best, MODELS_DIR / "xgb_pipeline.joblib")
    (MODELS_DIR / "route_stats.json").write_text(json.dumps(route_stats(legs[train]), indent=2))

    scored = legs[["trip_uuid", "source_center", "destination_center", "od_start_time", "route_type",
                   "source_state", "destination_state", "source_type", "destination_type",
                   "osrm_time", "osrm_distance", "center_distance_km", "late"]].copy()
    scored["risk_score"] = best.predict_proba(X)[:, 1]
    scored["in_test"] = ~train
    scored.to_parquet(MODELS_DIR / "scored_legs.parquet")


if __name__ == "__main__":
    main()
