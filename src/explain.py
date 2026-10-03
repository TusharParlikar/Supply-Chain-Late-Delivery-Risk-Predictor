"""SHAP explanations for the XGBoost pipeline."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from features import CATEGORICAL, NUMERIC, build
from paths import MODELS_DIR, REPORTS_DIR

MODEL_PATH = MODELS_DIR / "xgb_pipeline.joblib"


def explainer_for(pipe):
    return shap.TreeExplainer(pipe.named_steps["clf"])


def explain_rows(pipe, X, explainer=None):
    """SHAP Explanation for raw feature rows X (log-odds scale)."""
    prep = pipe.named_steps["prep"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)
    sv = (explainer or explainer_for(pipe))(Xt)
    # Plots show real values (e.g. 4 scheduled days), not standard-scaled ones.
    display = Xt.copy()
    display[NUMERIC] = X[NUMERIC].values
    sv.display_data = display.values
    return sv


def original_feature(name):
    """Map a one-hot column like 'route_type_FTL' back to 'route_type'."""
    return next((c for c in CATEGORICAL if name.startswith(c + "_")), name)


def main():
    pipe = joblib.load(MODEL_PATH)
    X, _, legs = build()
    sample = X.sample(5000, random_state=42)
    sv = explain_rows(pipe, sample)

    REPORTS_DIR.mkdir(exist_ok=True)
    shap.summary_plot(sv, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150)
    plt.close()

    importance = (
        pd.DataFrame(abs(sv.values), columns=sv.feature_names)
        .mean()
        .groupby(original_feature)
        .sum()
        .sort_values(ascending=False)
        .round(4)
    )
    importance.rename("mean_abs_shap").rename_axis("feature").to_csv(REPORTS_DIR / "shap_importance.csv")
    print(importance.to_string())

    # One example leg: the highest-risk leg in the sample.
    proba = pipe.predict_proba(sample)[:, 1]
    i = int(proba.argmax())
    leg = legs.loc[sample.index[i]]
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    plt.title(f"{leg.source_center} to {leg.destination_center}: late risk {proba[i]:.2f}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_waterfall_example.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
