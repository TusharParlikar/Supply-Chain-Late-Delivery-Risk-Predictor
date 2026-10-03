"""Streamlit app: late-delivery risk for Indian shipments on the Delhivery network."""
import sys
import threading
from datetime import date, time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from explain import explain_rows, explainer_for  # noqa: E402
from features import FEATURES, LATE_FACTOR, add_features, build  # noqa: E402
from paths import MODELS_DIR, REPORTS_DIR  # noqa: E402
from predict import REQUIRED_COLUMNS, Predictor  # noqa: E402

st.set_page_config(page_title="Late-Delivery Risk India", layout="wide")


@st.cache_resource
def load_predictor():
    p = Predictor()
    return p, explainer_for(p.pipe)


@st.cache_resource
def plot_lock():
    # pyplot's current figure is global; reruns run in parallel threads and would draw on each other's figure.
    return threading.Lock()


@st.cache_data
def load_legs():
    scored = pd.read_parquet(MODELS_DIR / "scored_legs.parquet")
    X, _, _ = build()
    return scored, X


def waterfall(pipe, explainer, X_row):
    sv = explain_rows(pipe, X_row, explainer)
    with plot_lock():
        fig = plt.figure()
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Bars in log-odds. Red pushes toward late, blue toward on time.")


def risk_label(p):
    return "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"


predictor, explainer = load_predictor()
centers = predictor.centers

st.title("Late-Delivery Risk: Delhivery Network, India")
st.caption(f"{len(centers):,} Delhivery centers across {centers['state'].nunique()} states and union territories. "
           f"A leg is late when it takes more than {LATE_FACTOR}x the OSRM route-planner time.")

tab_one, tab_batch, tab_dash, tab_models = st.tabs(
    ["Check a shipment", "Upload CSV", "Network dashboard", "Model comparison"])

# ---------------------------------------------------------------- single shipment
with tab_one:
    c1, c2, c3, c4 = st.columns(4)
    src_pin = c1.text_input("Pickup pincode", "421302")
    dst_pin = c2.text_input("Delivery pincode", "400011")
    route_type = c3.selectbox("Route type", sorted(predictor.stats),
                              help="FTL = full truck load, Carting = smaller vehicles")
    k = c4.slider("Centers to show", 3, 10, 5)
    c5, c6 = st.columns(2)
    dep_date = c5.date_input("Departure date", date(2018, 10, 1))
    dep_time = c6.time_input("Departure time", time(9, 0))

    src_loc, dst_loc = predictor.locator.locate(src_pin), predictor.locator.locate(dst_pin)
    bad = [label for label, loc in (("Pickup", src_loc), ("Delivery", dst_loc)) if loc is None]
    if bad:
        st.error(f"{' and '.join(bad)} pincode not found. Enter a valid 6-digit Indian pincode.")
    else:
        near = {"source": predictor.nearest(src_pin, k), "destination": predictor.nearest(dst_pin, k)}
        left, right = st.columns(2)
        pick = {}
        for col, side, label in ((left, "source", "pickup"), (right, "destination", "delivery")):
            with col:
                st.subheader(f"Centers near {label}")
                table = near[side][["name", "city", "ctype", "state", "distance_km"]].round({"distance_km": 1})
                st.dataframe(table, column_config={"ctype": "type", "distance_km": "km"})
                pick[side] = st.selectbox(
                    f"{label.title()} center", table.index, key=f"pick_{side}",
                    format_func=lambda c, t=table: f"{t.loc[c, 'name']} ({t.loc[c, 'distance_km']} km)")

        map_df = pd.concat([
            pd.DataFrame([src_loc, dst_loc], columns=["latitude", "longitude"],
                         index=["pickup pincode", "delivery pincode"]).assign(what="your pincode"),
            near["source"][["name", "latitude", "longitude"]].set_index("name").assign(what="center near pickup"),
            near["destination"][["name", "latitude", "longitude"]].set_index("name")
            .assign(what="center near delivery"),
        ])
        fig = px.scatter_map(map_df.reset_index(names="place"), lat="latitude", lon="longitude", color="what",
                             hover_name="place", zoom=5, height=420)
        fig.update_traces(marker={"size": 12})
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        st.plotly_chart(fig, width="stretch")

        request = pd.DataFrame([{
            "source_pincode": src_pin, "destination_pincode": dst_pin, "route_type": route_type,
            "departure_time": pd.Timestamp.combine(dep_date, dep_time),
            "source_center": pick["source"], "destination_center": pick["destination"],
        }])
        result = predictor.score(request).iloc[0]
        if result["error"]:
            st.error(result["error"])
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Late risk", f"{result['risk_score']:.0%}", risk_label(result["risk_score"]),
                      delta_color="off")
            m2.metric("Route distance (est.)", f"{result['osrm_distance']:.0f} km")
            m3.metric("Planned time (est.)", f"{result['osrm_time'] / 60:.1f} h")
            m4.metric("Counts as late after", f"{LATE_FACTOR * result['osrm_time'] / 60:.1f} h")
            st.caption("Route estimate comes from past legs between these two centers."
                       if result["route_from_history"]
                       else "No past legs between these centers; route estimated from straight-line distance.")
            if pick["source"] == pick["destination"]:
                st.info("Pickup and delivery use the same center: this is a local delivery.")
            feats = predictor.estimate_route(
                add_features(request.assign(od_start_time=request["departure_time"]), centers))
            with st.expander("Why this risk? (SHAP)"):
                waterfall(predictor.pipe, explainer, feats[FEATURES])

# ---------------------------------------------------------------- batch upload
with tab_batch:
    st.write("Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
             + ". Optional: `source_center`, `destination_center` to override the nearest center.")
    sample = pd.DataFrame({
        "source_pincode": ["421302", "110037", "560067", "500032"],
        "destination_pincode": ["400011", "122001", "600001", "530001"],
        "route_type": ["Carting", "FTL", "FTL", "FTL"],
        "departure_time": ["2018-10-01 09:00", "2018-10-01 22:30", "2018-10-02 06:00", "2018-10-02 14:00"],
    })
    st.download_button("Download sample CSV", sample.to_csv(index=False), "shipments_sample.csv", "text/csv")
    upload = st.file_uploader("Shipments CSV", type="csv")
    if upload is not None:
        try:
            scored = predictor.score(pd.read_csv(upload, dtype=str))
        except (ValueError, pd.errors.ParserError) as e:
            st.error(f"Could not read file: {e}")
        else:
            b1, b2, b3 = st.columns(3)
            b1.metric("Shipments", len(scored))
            b2.metric("Scored", int((scored["error"] == "").sum()))
            b3.metric("High risk (60%+)", int((scored["risk_score"] >= 0.6).sum()))
            scored["risk_level"] = scored["risk_score"].map(lambda p: risk_label(p) if pd.notna(p) else "")
            scored["source_center_name"] = scored["source_center"].map(centers["name"])
            scored["destination_center_name"] = scored["destination_center"].map(centers["name"])
            cols = ["source_pincode", "destination_pincode", "route_type", "departure_time", "source_center_name",
                    "source_center_km", "destination_center_name", "destination_center_km", "osrm_distance",
                    "osrm_time", "risk_score", "risk_level", "error"]
            out = scored[[c for c in cols if c in scored]]
            st.dataframe(out.style.format({"risk_score": "{:.0%}", "osrm_distance": "{:.0f}",
                                           "osrm_time": "{:.0f}"}, na_rep=""), hide_index=True)
            st.download_button("Download results", out.to_csv(index=False), "shipments_scored.csv", "text/csv")

# ---------------------------------------------------------------- dashboard over historical legs
with tab_dash:
    legs, X = load_legs()
    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("Source state", sorted(legs["source_state"].unique()))
    rtypes = f2.multiselect("Route type", sorted(legs["route_type"].unique()))
    test_only = f3.checkbox("Test week only", help="Legs from Sep 27 to Oct 3 2018, never seen in training")
    view = legs
    if states:
        view = view[view["source_state"].isin(states)]
    if rtypes:
        view = view[view["route_type"].isin(rtypes)]
    if test_only:
        view = view[view["in_test"]]

    if view.empty:
        st.warning("No legs match these filters.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Trip legs", f"{len(view):,}")
        d2.metric("Actual late rate", f"{view['late'].mean():.1%}")
        d3.metric("Mean predicted risk", f"{view['risk_score'].mean():.1%}")

        def risk_by(col):
            g = (view.groupby(col).agg(predicted_risk=("risk_score", "mean"), actual_late_rate=("late", "mean"),
                                       legs=("risk_score", "size"))
                 .query("legs >= 20").sort_values("predicted_risk", ascending=False).reset_index())
            fig = px.bar(g, x="predicted_risk", y=col, orientation="h", hover_data=["actual_late_rate", "legs"],
                         labels={"predicted_risk": "Mean predicted late risk"})
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(300, 26 * len(g)))
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            return fig

        s1, s2, s3, s4 = st.tabs(["By source state", "By destination state", "By route type", "By center type"])
        with s1:
            st.plotly_chart(risk_by("source_state"), width="stretch")
        with s2:
            st.plotly_chart(risk_by("destination_state"), width="stretch")
        with s3:
            st.plotly_chart(risk_by("route_type"), width="stretch")
        with s4:
            st.plotly_chart(risk_by("source_type"), width="stretch")
            st.caption("HB/H = hub, D/DC = delivery center, I = intermediate, IP/DPC/DPP/PC = smaller points.")
        st.caption("Groups with fewer than 20 legs are hidden.")

        st.subheader("Leg lookup")
        trip = st.selectbox("Trip", sorted(view["trip_uuid"].unique()))
        trip_legs = view[view["trip_uuid"] == trip]
        st.dataframe(trip_legs.drop(columns=["in_test", "trip_uuid"]), hide_index=True)
        idx = st.selectbox("Leg", trip_legs.index, format_func=lambda i: (
            f"{trip_legs.loc[i, 'source_center']} to {trip_legs.loc[i, 'destination_center']}"))
        waterfall(predictor.pipe, explainer, X.loc[[idx]])

with tab_models:
    st.dataframe(pd.read_csv(REPORTS_DIR / "model_comparison.csv"), hide_index=True)
    st.caption("Test set = every trip from Sep 27 to Oct 3 2018. Training used only earlier trips.")
