"""Train and compare Logistic Regression, Random Forest and XGBoost."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CATEGORICAL, NUMERIC, ROOT, build, load_raw

SEED = 42
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ], verbose_feature_names_out=False)


def models(pos_weight):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1, random_state=SEED,
        ),
    }


def main():
    df = load_raw()
    X, y, df = build(df)
    # Lines of one order share timestamp, location and outcome, so split by Order Id:
    # an order is entirely in train or entirely in test. One fold of 5 = 80/20, stratified.
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(folds.split(X, y, groups=df["Order Id"]))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    assert not set(df["Order Id"].iloc[train_idx]) & set(df["Order Id"].iloc[test_idx])
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"train {len(X_train)}, test {len(X_test)}, scale_pos_weight {pos_weight:.3f}")

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
        print(rows[-1])

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    table = pd.DataFrame(rows).round(4).sort_values("roc_auc", ascending=False)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    # XGBoost is the model SHAP explains and the dashboard serves.
    best = fitted["XGBoost"]
    joblib.dump(best, MODELS_DIR / "xgb_pipeline.joblib")

    scored = df[["Order Id", "order_date", "Market", "Order Region", "Shipping Mode", "Category Name",
                 "Customer Segment", "Days for shipment (scheduled)", "Late_delivery_risk"]].copy()
    scored["risk_score"] = best.predict_proba(X)[:, 1]
    scored["in_test"] = scored.index.isin(X_test.index)
    scored.to_parquet(MODELS_DIR / "scored_orders.parquet")


if __name__ == "__main__":
    main()
