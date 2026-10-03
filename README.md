# Supply Chain Late-Delivery Risk Predictor (India)

This project predicts whether a shipment leg on Delhivery's Indian logistics network will run late. It uses only information available at dispatch. You give it a pickup and a delivery pincode, and it finds the nearest Delhivery centers, estimates the route, predicts late risk and explains the prediction with SHAP.

**Stack:** Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit, Plotly

## Data

| Source | Use |
|---|---|
| [Delhivery dataset](https://www.kaggle.com/datasets/santanukundu/delhivery-dataset) (Kaggle) | Trip segments between Delhivery centers, Sep 12 to Oct 3 2018 |
| [India's pincode-wise districts](https://www.kaggle.com/datasets/somesh24/indias-pincode-wise-districts) (Kaggle) | Latitude and longitude for 11,042 Indian pincodes |

Both files download automatically with `kagglehub` on the first run.

| Fact | Value |
|---|---|
| Segment records (rows) | 144,867 |
| Trips | 14,817 |
| Trip legs (source center to destination center) | 26,368 |
| Delhivery centers | 1,657 |
| States and union territories | 33 |
| Route types | FTL (full truck load), Carting |
| Late legs | 7,249 (27.5%) |

### What "late" means

Each leg carries an **OSRM time**: the route planner's estimate of the drive, made before the leg starts. A leg is labelled late when its actual time is more than **2.5 times** the OSRM time (`LATE_FACTOR` in `src/features.py`).

- The median actual-to-OSRM ratio is 2.0, so normal handling time is about double the drive time.
- A ratio of 2.5 marks the slowest 27.5% of legs.

Late rate by route type: Carting 36.6%, FTL 19.4%.

The source states with the highest late rates (at least 200 legs each) are West Bengal (53%), Assam (48%) and Bihar (45%). The lowest are Kerala (13%), Punjab (13%) and Andhra Pradesh (15%).

## Preventing leakage

**Grouping rows into legs.** The time and distance columns add up across a leg's segments. The pipeline groups segments into one row per leg (`trip_uuid`, `source_center`, `destination_center`) and takes the final totals. A leg is never split across rows.

**Dropped columns.** These are known only after the leg finishes, or are computed from the outcome:

| Column | Reason |
|---|---|
| `actual_time`, `od_end_time`, `start_scan_to_end_scan` | Actual duration; the label is built from these |
| `actual_distance_to_destination` | Measured during the trip |
| `factor`, `segment_factor` | Actual time divided by OSRM time, which restates the label |
| `segment_actual_time`, `segment_osrm_*` | Per-segment values recorded during the trip |
| `is_cutoff`, `cutoff_factor`, `cutoff_timestamp` | Set during processing |

`src/features.py` uses an explicit whitelist of features. An assert stops the pipeline if any leakage column gets into the feature matrix.

**Split by time.** The pipeline keeps Kaggle's own split. Training trips run from Sep 12 to Sep 26 2018, and every test trip comes from the following week (Sep 27 to Oct 3). An assert checks that no trip appears in both sets. The model is scored on a week it never saw, the same way it would be used in practice.

## Locating Delhivery centers

Center codes contain the center's pincode: `IND388121AAA` means pincode 388121. Each center gets a location in this order:

1. **Exact pincode match.** Covers 840 centers.
2. **District average.** The average location of the pincode's 3-digit sorting district. Covers 726 centers.
3. **State average.** The average location of the other centers in the same state. Covers 91 centers.

Center names follow the pattern `City_Place_Type (State)`. The pipeline reads the city, center type (HB = hub, D/DC = delivery center, I = intermediate, and so on) and state from the name.

## Features (known at dispatch)

| Type | Features |
|---|---|
| Categorical (one-hot) | route type, source and destination state, source and destination center type, source and destination center. Centers with fewer than 20 legs share one "infrequent" column. |
| Numeric (scaled) | OSRM time, OSRM distance, straight-line distance between centers, same-state flag, departure hour, departure day of week |

## Class imbalance

27.5% of legs are late:

- Logistic Regression and Random Forest use `class_weight="balanced"`.
- XGBoost uses `scale_pos_weight` = on-time legs / late legs in the training set (2.66).

## Results

The test set is 7,421 legs from the unseen final week. The decision threshold is 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.786 | 0.588 | 0.785 | 0.673 | **0.865** |
| Random Forest | 0.768 | 0.561 | 0.780 | 0.653 | 0.858 |
| Logistic Regression | 0.702 | 0.480 | 0.755 | 0.587 | 0.799 |

XGBoost settings: `max_depth=8`, `n_estimators=600`, `learning_rate=0.03`. These were chosen on a validation slice taken from the training period (Sep 23 to 26), not on the test week. The balanced class weights favour recall: the model catches about 79% of late legs.

## Explainability (SHAP)

`src/explain.py` runs `shap.TreeExplainer` on 5,000 legs.

The top risk drivers are listed below, as mean |SHAP| summed over each feature's one-hot columns:

1. OSRM time (0.71). Short planned legs have little slack for loading and handling delays.
2. Destination state (0.46)
3. Source state (0.44)
4. Destination center type (0.33)
5. Departure hour (0.33)

![SHAP summary](reports/shap_summary.png)

![SHAP waterfall](reports/shap_waterfall_example.png)

## Streamlit app

```bash
streamlit run app.py
```

**Check a shipment.** Enter the pickup pincode, delivery pincode, route type and departure time. The app then:

- Lists the nearest Delhivery centers to each pincode, with their distances, and shows them on a map.
- Uses the nearest center by default. You can pick a different one.
- Estimates the route. If past legs exist between the two centers, it uses their median OSRM distance and time. Otherwise it estimates from straight-line distance, using a road factor and an average speed learned from training legs.
- Shows the late risk, the planned time, the hour after which the leg counts as late, and a SHAP explanation.

**Upload CSV.** Upload a file with the columns `source_pincode`, `destination_pincode`, `route_type` and `departure_time`. `source_center` and `destination_center` are optional and override the nearest center. Every row gets its nearest centers, a route estimate, a risk score, a risk level and an error message if the row is invalid. You can download the scored file, and a sample CSV is provided.

**Network dashboard.** Shows predicted and actual late rates by source state, destination state, route type and center type. You can filter by state, route type and test week. A leg lookup shows any leg's SHAP waterfall.

**Model comparison.** Shows the metrics table.

## Project structure

```
.
├── src/
│   ├── paths.py         # project folders
│   ├── geo.py           # pincode geocoding, center table, haversine, nearest centers
│   ├── features.py      # download, legs, label, leakage-safe features
│   ├── train.py         # time split, 3 models, metrics, route stats for the app
│   ├── explain.py       # SHAP summary, importance, example waterfall
│   └── predict.py       # score new shipments from pincodes (used by the app)
├── app.py               # Streamlit app
├── reports/             # model_comparison.csv, shap_importance.csv, plots
├── data/raw/            # Kaggle CSVs (not committed)
├── models/              # pipeline, route stats, scored legs (not committed)
└── requirements.txt
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

cd src
python geo.py                     # self-checks: haversine, name parsing, pincode lookup
python features.py                # downloads data, prints dataset facts
python train.py                   # trains and compares models (about 1 min)
python explain.py                 # SHAP plots
python predict.py                 # self-checks: scoring from pincodes
cd ..
streamlit run app.py
```

## Limitations

- The data covers three weeks of 2018, so it cannot capture seasonal effects such as festivals or the monsoon.
- Center locations come from pincode centroids, not street addresses. Centers in the same district can show 0 km apart. When that happens and no history exists, the app uses the typical length of such legs.
- For a new center pair, the OSRM time is estimated from distance, while the model was trained on real OSRM times. Predictions are most reliable for center pairs seen in the data.
- The "late" label is relative to the route planner's time. Delhivery's actual delivery promises are not in the data.
