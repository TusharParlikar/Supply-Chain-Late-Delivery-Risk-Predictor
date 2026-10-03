# Supply Chain Late-Delivery Risk Predictor

**Stack:** Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit
**Date:** Oct 2026

## Resume bullets (the project must back these up)

- Trained and compared Logistic Regression, Random Forest, and XGBoost classifiers to flag late-delivery risk across 180,000+ supply chain orders, with leakage-safe features and class-imbalance handling.
- Applied SHAP to explain each order's risk drivers and built a Streamlit dashboard showing risk by region, shipping mode, and product category.

## Data source (Kaggle)

- Dataset: **DataCo Smart Supply Chain for Big Data Analysis**
  https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
- File to use: `DataCoSupplyChainDataset.csv` (read with `encoding="latin-1"`)
- Column reference: `DescriptionDataCoSupplyChain.csv`
- Skip: `tokenized_access_logs.csv` (web logs, not needed)
- Target column: `Late_delivery_risk` (1 = late, 0 = not late)

### Numbers to match

| Claim on resume | Actual from dataset | Verified? |
|---|---|---|
| 180,000+ orders | 180,519 rows (order lines); 65,752 unique orders | [x] (see note) |
| Late-delivery target | `Late_delivery_risk`: 98,977 late (54.8%) / 81,542 not late (45.2%) | [x] |
| Regions | `Order Region` 23, `Market` 5 | [x] |
| Shipping modes | `Shipping Mode` 4 | [x] |
| Product categories | `Category Name` 50 | [x] |
| 53 columns | 53 | [x] |

**Note:** each row is one order *line*. For accurate wording on the resume, use "180,000+ supply chain order lines" or "65,000+ orders (180,000+ line items)".

Fill in the actual values after loading. If any number differs, update the resume or this table, never leave them mismatched.

## TODO

### 1. Setup
- [x] Create `requirements.txt`: pandas, numpy, scikit-learn, xgboost, shap, streamlit, plotly, matplotlib, joblib
- [x] Create a virtual environment and install requirements
- [x] Download dataset from Kaggle into `data/raw/` (manual download or `kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`)
- [x] Add `data/` to `.gitignore` (file is about 95 MB, do not commit it)

### 2. Data check (`src/features.py`; run it directly to print the facts)
- [x] Load CSV and confirm row count = 180,519 and 53 columns
- [x] Confirm target balance of `Late_delivery_risk` and record exact counts in the table above
- [x] Count unique values for `Order Region`, `Market`, `Shipping Mode`, `Category Name`
- [x] Nulls: not an issue, because the feature whitelist excludes `Product Description` and `Order Zipcode`
- [x] Plot late rate by shipping mode, region, category (in the dashboard)

### 3. Leakage-safe features (`src/features.py`)
- [x] Drop columns that are only known after shipping or that directly encode the target:
  - `Days for shipping (real)`
  - `Delivery Status`
  - `Order Status`
  - `shipping date (DateOrders)`
- [x] Drop PII and ID columns: `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street`, `Product Image`, `Order Id`, `Order Item Id`, `Customer Id`, `Product Card Id`, etc.
- [x] Keep order-time features: `Days for shipment (scheduled)`, `Shipping Mode`, `Order Region`, `Market`, `Category Name`, `Department Name`, `Customer Segment`, `Type` (payment), `Order Item Quantity`, `Order Item Discount Rate`, `Product Price`, `Sales`, `Latitude`, `Longitude`
- [x] Derive from `order date (DateOrders)`: month, day of week, hour
- [x] Write a short note in the README listing every dropped column and why
- [x] One assert check: no leakage column survives in the final feature matrix

### 4. Train and compare models (`src/train.py`)
- [x] Train/test split: grouped by `Order Id` with `StratifiedGroupKFold` (80/20). A random row split leaked: AUC went from 0.876 to 0.759 once lines from the same order could no longer sit in both sets
- [ ] Optional: time-based split on order date to show results hold over time
- [x] Preprocessing pipeline: `ColumnTransformer` with one-hot for categoricals, scaling for numerics (scaling needed for Logistic Regression)
- [x] Class-imbalance handling:
  - Logistic Regression and Random Forest: `class_weight="balanced"`
  - XGBoost: `scale_pos_weight = negatives / positives`
- [x] Train Logistic Regression, Random Forest, XGBoost
- [x] Report for each model: accuracy, precision, recall, F1, ROC-AUC
- [x] Save comparison table to `reports/model_comparison.csv`
- [x] Save best model with `joblib` to `models/`

### 5. SHAP explanations (`src/explain.py`)
- [x] `shap.TreeExplainer` on the XGBoost model
- [x] Global summary plot (top risk drivers) saved to `reports/`
- [x] Per-order explanation (waterfall plot) for a chosen order
- [x] Map one-hot feature names back to readable names (`reports/shap_importance.csv`)

### 6. Streamlit dashboard (`app.py`)
- [x] Load saved model and a scored copy of the dataset
- [x] Late-risk rate by **region** (`Order Region` / `Market`)
- [x] Late-risk rate by **shipping mode**
- [x] Late-risk rate by **product category**
- [x] Sidebar filters for region, shipping mode, category
- [x] Order lookup: pick an order and show its risk score plus SHAP waterfall
- [x] Model comparison table from `reports/model_comparison.csv`

### 7. Finish
- [x] README: problem, dataset link, leakage notes, results table, run steps
- [ ] Add a dashboard screenshot to the README
- [ ] Re-check every number in the "Numbers to match" table against the final run
- [x] Push to GitHub: https://github.com/TusharParlikar/Supply-Chain-Late-Delivery-Risk-Predictor
- [ ] Add the GitHub link to the resume
- [ ] Optional: deploy the dashboard on Streamlit Community Cloud
