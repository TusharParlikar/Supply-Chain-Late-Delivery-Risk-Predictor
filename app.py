"""Streamlit dashboard: late-delivery risk by region, shipping mode and product category."""
import sys
import threading
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from explain import explain_rows, explainer_for  # noqa: E402
from features import ROOT, build, load_raw  # noqa: E402

st.set_page_config(page_title="Late-Delivery Risk", layout="wide")


@st.cache_resource
def load_model():
    pipe = joblib.load(ROOT / "models" / "xgb_pipeline.joblib")
    return pipe, explainer_for(pipe)


@st.cache_resource
def plot_lock():
    # pyplot's current figure is global; reruns run in parallel threads and would draw on each other's figure.
    return threading.Lock()


@st.cache_data
def load_data():
    scored = pd.read_parquet(ROOT / "models" / "scored_orders.parquet")
    X, _, _ = build(load_raw())
    return scored, X


pipe, explainer = load_model()
scored, X = load_data()

st.title("Supply Chain Late-Delivery Risk")

with st.sidebar:
    st.header("Filters")
    filters = {}
    for col in ["Market", "Order Region", "Shipping Mode", "Category Name"]:
        filters[col] = st.multiselect(col, sorted(scored[col].unique()))
    test_only = st.checkbox("Test-set orders only", help="Orders the model never saw in training")

view = scored
for col, picked in filters.items():
    if picked:
        view = view[view[col].isin(picked)]
if test_only:
    view = view[view["in_test"]]

if view.empty:
    st.warning("No orders match these filters.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Order lines", f"{len(view):,}")
c2.metric("Actual late rate", f"{view['Late_delivery_risk'].mean():.1%}")
c3.metric("Mean predicted risk", f"{view['risk_score'].mean():.1%}")


def risk_by(col, top=None):
    g = (
        view.groupby(col)
        .agg(predicted_risk=("risk_score", "mean"), actual_late_rate=("Late_delivery_risk", "mean"),
             orders=("risk_score", "size"))
        .sort_values("predicted_risk", ascending=False)
        .reset_index()
    )
    if top:
        g = g.head(top)
    fig = px.bar(g, x="predicted_risk", y=col, orientation="h", hover_data=["actual_late_rate", "orders"],
                 labels={"predicted_risk": "Mean predicted late risk"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(300, 28 * len(g)))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return fig


tab_region, tab_mode, tab_cat, tab_order, tab_models = st.tabs(
    ["By region", "By shipping mode", "By product category", "Order lookup", "Model comparison"]
)
with tab_region:
    st.plotly_chart(risk_by("Order Region"), width="stretch")
with tab_mode:
    st.plotly_chart(risk_by("Shipping Mode"), width="stretch")
with tab_cat:
    st.plotly_chart(risk_by("Category Name", top=25), width="stretch")
    st.caption("Top 25 categories by predicted risk.")

with tab_order:
    order_id = st.selectbox("Order Id", sorted(view["Order Id"].unique()))
    lines = view[view["Order Id"] == order_id]
    st.dataframe(lines.drop(columns=["in_test"]), hide_index=True)
    idx = lines.index[0]
    st.write(f"Explaining the first line of this order (risk {lines.loc[idx, 'risk_score']:.2f}).")
    sv = explain_rows(pipe, X.loc[[idx]], explainer)
    with plot_lock():
        fig = plt.figure()
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Bars in log-odds. Red pushes toward late, blue toward on time.")

with tab_models:
    st.dataframe(pd.read_csv(ROOT / "reports" / "model_comparison.csv"), hide_index=True)
