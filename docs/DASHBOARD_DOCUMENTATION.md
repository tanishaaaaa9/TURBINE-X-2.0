# TURBINE-X 2.0 — Streamlit Industrial Dashboard Documentation

## Overview
The TURBINE-X 2.0 Dashboard (`app.py`) is an interactive, responsive industrial Artificial Intelligence and Machine Learning interface designed for real-time turbine monitoring, point forecasting, risk classification, anomaly detection, explainability, and uncertainty quantification.

## Module Structure
1. **Home Portal**: Features the official TURBINE-X 2.0 logo, industrial hero section, platform KPI cards, dataset stats, and project safety notices.
2. **Dataset Explorer**: Scatter plots, thermodynamic interactions, Pearson correlation heatmaps, and summary statistical tables.
3. **Single Prediction**: Point predictions and 90% split-conformal intervals for Track A (operational emissions) and Track B (joint multi-target prediction).
4. **Batch CSV Prediction**: Vectorized file upload, multi-row inference (>10,000 rows/sec throughput), and downloadable enriched CSVs.
5. **Emission Risk Classification**: Interactive multi-class operational risk tiering (Normal, Elevated, High Risk) with class probability confidence gauges.
6. **Unsupervised Anomaly Detection**: Unsupervised Isolation Forest, LOF, and Deep Autoencoder anomaly scores and flags.
7. **Time-Series Forecasting**: Multi-step sequential forecasts ($h=1, 6, 24$ hours) for continuous emissions and turbine yield.
8. **Deep Neural Architectures**: Training curves, validation loss profiles, and neural parameter inspection.
9. **Multi-Task Model Leaderboards**: Interactive sortable tables for 16 regressors and 14 classifiers.
10. **Explainable AI (XAI)**: Pre-rendered TreeSHAP beeswarm, global importance, and waterfall local explanations.
11. **Uncertainty Analysis**: 90% split-conformal calibration diagnostics and non-stationary drift analysis.
12. **MLOps Registry**: Serialized artifact inspection, parameter counts, and versioning.
13. **About TURBINE-X**: System architecture, citations, and disclaimer notices.

## How to Launch
```powershell
streamlit run app.py
```
Or execute:
```powershell
.\RUN_DASHBOARD.bat
```

