# TURBINE-X: Interactive Dashboard User Guide

## 1. Application Architecture (`app.py`)
The TURBINE-X dashboard is built using Streamlit and Plotly to provide an intuitive interface for model inspection, live inference, and diagnostic review.

---

## 2. Main Dashboard Views & Capabilities

### Tab 1: Interactive Inference & Conformal Uncertainty Bounds
- **Operating Setpoint Sliders:** Provides intuitive sliders and numerical inputs for all 9 thermodynamic and ambient variables (`AT`, `AP`, `AH`, `AFDP`, `GTEP`, `TIT`, `TAT`, `TEY`, `CDP`), with default baseline values reflecting standard turbine operating setpoints.
- **Predictive Estimation:** Instantly computes multi-target predictions for the selected model.
- **90% Split-Conformal Uncertainty Intervals:** Displays empirical 90% confidence bounds $[y_{\text{lower}}, y_{\text{upper}}]$ derived from empirical 2014 validation errors.
- **Batch CSV Upload & Processing:** Allows users to upload a CSV file of sensor readings, generates batch predictions with conformal bounds, and provides a one-click CSV export button.

### Tab 2: Model Benchmarking Leaderboard
- Loads the locked 2015 test set evaluation table (`outputs/tables/locked_test_metrics.csv`).
- Renders an interactive leaderboard ranking models by RMSE, MAE, and $R^2$.
- Grouped bar charts dynamically compare architectures across targets.

### Tab 3: Dataset Intelligence & Thermodynamics
- Displays master dataset summary metrics (36,733 records, 2011–2015).
- Displays raw data sample inspection and correlation heatmaps across all 11 sensor channels.

### Tab 4: Feature Explainability & Diagnostics
- Renders horizontal bar charts of relative feature importances for tree-based models.
- Displays offline evaluation charts comparing model performances across both prediction tracks.

---

## 3. Launch Instructions
From PowerShell or Command Prompt:
```cmd
RUN_DASHBOARD.bat
```
Or directly via Python:
```powershell
streamlit run app.py
```
Open your browser to: `http://localhost:8501`.

