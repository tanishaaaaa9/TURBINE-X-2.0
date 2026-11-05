# Changelog — TURBINE-X

All notable changes to the **TURBINE-X** platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-11-05

### Added
- **Multi-Task Machine Learning Architecture:** Expanded from single-task regression to an integrated industrial AI platform supporting regression, classification, unsupervised anomaly detection, and sequential forecasting.
- **16 Supervised Regression Algorithms:** Benchmarked across Track A (9 inputs $\to$ CO, NOx) and Track B (8 inputs $\to$ CO, NOx, TEY), including Linear, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, CatBoost, LightGBM, KNN, SVR, MLPRegressor, StackingRegressor, and VotingRegressor.
- **14 Operational Risk Classifiers:** Multi-class risk categorization into Tier 1 (Normal), Tier 2 (Elevated), and Tier 3 (High Risk) for both CO and NOx, with out-of-fold StackingClassifier achieving 55.34% accuracy and 88.24% top-2 accuracy.
- **5 Unsupervised Anomaly Detectors:** Reference-data baseline modeling using IsolationForest, LocalOutlierFactor, OneClassSVM, EllipticEnvelope, and a bottleneck Deep Autoencoder.
- **Sequential Multi-Horizon Forecasting:** Evaluation across observation horizons $h=1, 6, 24$ with Persistence, Rolling Moving Average, Holt Exponential Smoothing, and Autoregressive Lag Ridge.
- **Explainable AI (SHAP):** TreeExplainer attribution and beeswarm visualizations for thermodynamic physical validation.
- **Conformal Prediction:** 90% non-exchangeable prediction intervals providing rigorous uncertainty quantification.
- **Interactive Streamlit Application:** 13 functional modules covering executive KPIs, telemetry simulator, SHAP explorer, and publication figure gallery.
- **36 Publication Figures:** Generated at 300 DPI adhering to industrial technical styling.

### Changed
- **Neural Network Optimization:** Tuned `MLPRegressor`, `MLPClassifier`, and `DeepAutoencoder` hyperparameters (`max_iter=500`, `learning_rate_init=0.003`, `early_stopping=True`), achieving full convergence in ~25 iterations with 0 warnings.
- **Scientific Figure Alignment:** Regenerated Figures 28–30 to depict genuine sequential models (Persistence, Holt, Autoregressive Lag Ridge) instead of synthetic curves.
- **Scientific Reframe of Time Series:** Formally documented that raw UCI data contains sequential operational steps without calendar timestamps.

### Removed
- Unimplemented deep sequence models (LSTM, GRU, 1D-CNN, Transformer) purged from experimental claims and formally documented as `NOT IMPLEMENTED`.

---

## [1.0.0] - 2026-11-04

### Added
- Initial baseline regression benchmarks for Gas Turbine CO, NOx, and TEY prediction.
- 19 exploratory Jupyter notebooks and initial temporal splitting framework.

