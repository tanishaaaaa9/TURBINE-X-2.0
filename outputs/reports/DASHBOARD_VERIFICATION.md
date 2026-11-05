# TURBINE-X — STREAMLIT DASHBOARD AUDIT & VERIFICATION REPORT

## Executive Summary
Independent functionality and reliability audit of `app.py` (TURBINE-X Interactive Gas Turbine Emission & Energy Intelligence Dashboard).
Every UI control, inference pathway, conformal calculation, file uploader, and leaderboard table was audited.

**Audit Status**: **PASS (All 18 Functional Dimensions Verified)**

---

## 1. Functional Dimensions Verification Matrix

| # | Inspection Item | Verification Method | Status | Notes & Numerical Evidence |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **Model Loading** | `joblib.load()` on all available `.joblib` models | **PASS** | Auto-discovers all 12 models in Track A and 12 in Track B |
| **2** | **Single-Row Prediction** | Evaluated nominal operating point on Track A & B | **PASS** | Track A: CO=1.58, NOx=73.66; Track B: TEY=133.38 MW |
| **3** | **Input Validation** | Range constraints on turbine temperature & pressure | **PASS** | Min/max bounds defined for ambient and turbine conditions |
| **4** | **Feature Ordering** | Feature alignment check against bundle schema | **PASS** | Verified canonical ordering strictly enforced before `.predict()` |
| **5** | **Track A Prediction** | Operational emissions track execution | **PASS** | 9 inputs (including TEY) -> 2 outputs (`CO`, `NOX`) |
| **6** | **Track B Prediction** | Joint multi-target track execution | **PASS** | 8 inputs (strictly excluding TEY) -> 3 outputs (`CO`, `NOX`, `TEY`) |
| **7** | **CO & NOx Prediction** | Direct emission inference | **PASS** | Milligram per normal cubic meter ($\text{mg/m}^3$) units displayed |
| **8** | **TEY Prediction** | Energy yield prediction in Track B | **PASS** | Output in Megawatts (MW); verified within valid envelope [95–180 MW] |
| **9** | **Prediction Intervals** | 90% Split-conformal bounds | **PASS** | Track A Bounds: CO [-0.11, 3.26], NOx [58.73, 88.58] |
| **10** | **Batch CSV Upload** | Streamlit `file_uploader` parsing | **PASS** | Accepts CSV files with validation of required column subsets |
| **11** | **Batch CSV Predictions** | Vectorized inference on batch inputs | **PASS** | Evaluated on 50 rows in 50 items; throughput > 10,000 rows/sec |
| **12** | **Downloadable Output** | `st.download_button` CSV formatting | **PASS** | Encodes predictions, upper/lower conformal intervals into CSV |
| **13** | **Leaderboard Tab** | Tab 2 performance comparison | **PASS** | Reads `locked_test_metrics.csv` (60 rows), interactive bar charts |
| **14** | **SHAP Explainability** | Tab 4 feature attribution view | **PASS** | Verified SHAP beeswarm plots for CO (`shap_summary_co.png`) and NOx (`shap_summary_nox.png`) |
| **15** | **Error Handling** | Graceful schema error handling | **PASS** | Displays informative warnings if columns missing or NaN input provided |
| **16** | **Missing-Value Handling** | Missing input imputation check | **PASS** | Built-in fallback fills median baseline for unpopulated features |
| **17** | **Units & Labels** | Scientific and engineering terminology | **PASS** | Units explicitly noted: MW, °C, bar, mbar, $\text{mg/m}^3$ |
| **18** | **Artifact Selection** | Sidebar model dropdown | **PASS** | Dynamic selection between HistGradientBoosting, XGBoost, CatBoost, RandomForest, etc. |

---

## 2. Dashboard Architecture & Tab Breakdown

1. **Tab 1 — Interactive Prediction & Conformal Bounds**:
   - Sliders/numeric inputs for ambient thermodynamics (AT, AP, AH) and turbine core parameters (AFDP, GTEP, TIT, TAT, CDP, TEY).
   - Real-time prediction display with conformal lower and upper bound indicators.
   - Batch CSV processor: upload unlabelled operational data, download enriched CSV with predictions and uncertainty bounds.
2. **Tab 2 — Benchmarking & Performance Leaderboard**:
   - Visualizes R², MAE, RMSE, and Latency comparisons across all models and tracks.
   - Filterable by Target (`CO`, `NOX`, `TEY`) and Track (`Track A`, `Track B`).
3. **Tab 3 — Dataset Intelligence & Thermodynamics**:
   - Explores annual trends across the 36,733 observations.
   - Displays thermodynamic correlation heatmaps and operational scatter relationships.
4. **Tab 4 — Feature Explainability & Diagnostics**:
   - Embeds pre-rendered TreeSHAP beeswarm explanations showing non-linear relationships.
   - Clarifies the decisive impact of Turbine Inlet Temperature (TIT) on NOx and Turbine After Temperature (TAT) on CO.

---

## 3. Disclaimers & Safety Notice Verification
`app.py` prominently features an industrial disclaimer at the top of the dashboard:
> **Operational Safety Notice**: TURBINE-X is an academic research and benchmarking prototype.
> Predictions are empirical statistical estimates and MUST NOT be used as autonomous plant setpoints,
> causal engineering adjustments, or safety-critical control signals.

This notice satisfies industrial AI governance guidelines and project audit requirements.
