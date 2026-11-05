# TURBINE-X 2.0 — Advanced Artificial Intelligence & Machine Learning Architecture

## Executive Architecture Overview

TURBINE-X 2.0 transforms the original gas turbine emission prediction benchmark into an integrated, industrial-grade Gas Turbine Intelligence Platform. It expands beyond pure regression to unite **multi-target regression, emission risk classification, unsupervised anomaly detection, deep neural architectures, sequential time-series forecasting, advanced stacking/voting ensembles, and explainable AI (XAI)** under a unified MLOps framework.

```
+---------------------------------------------------------------------------------------------------------+
|                                        TURBINE-X 2.0 SYSTEM ARCHITECTURE                                |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                     +-------------------------------+-------------------------------+
                     |                                                               |
                     v                                                               v
       [Track A: Operational Emissions]                                [Track B: Joint Multi-Target]
        - Inputs (9): AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP       - Inputs (8): AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP
        - Targets: CO, NOx emissions                                   - Targets: CO, NOx, and TEY (Energy Yield)
                     |                                                               |
+--------------------+---------------------------------------------------------------+--------------------+
|                                                                                                         |
v                                            v                               v                            v
[1. ADVANCED REGRESSION]           [2. RISK CLASSIFICATION]         [3. ANOMALY DETECTION]      [4. DEEP LEARNING]
- 16 Model Families:                - 14 Classifiers:                - 6 Unsupervised Methods:   - MLP Regressors & Classifiers
  Linear, Ridge, Lasso, ElasticNet,   Logistic, DecisionTree, RF,      Isolation Forest, LOF,      - Deep Autoencoders (Recon Error)
  DecisionTree, RandomForest,         ExtraTrees, HistGradient,        One-Class SVM,              - 1D-CNN Temporal Feature Extraction
  ExtraTrees, HistGradientBoosting,   GradBoost, XGBoost, CatBoost,    Elliptic Envelope,          - Sequence Recurrent Predictors
  XGBoost, CatBoost, LightGBM,        LightGBM, KNN, SVM, MLP,         DBSCAN, Deep Autoencoder.   - Early stopping & learning curves
  KNN, SVR, MLP, Stacking, Voting.    Stacking, Voting Classifiers.    - Unsupervised thresholds.
- Metrics: R2, MAE, MSE, RMSE,      - Tiers: Normal, Elevated, High  - Anomaly scores & flags.
  MedAE, EV, MaxErr, MAPE, sMAPE.     (strictly learned on Dev).
                                                                                                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
v                                            v                               v                            v
[5. TIME-SERIES FORECASTING]       [6. ENSEMBLE LEARNING]           [7. EXPLAINABLE AI]         [8. INTERACTIVE DASHBOARD]
- Rolling-origin sequence eval:     - Leakage-free Stacking          - Global TreeSHAP Summary   - Redesigned Streamlit Web App
  Persistence, Rolling MA,            using out-of-fold (OOF) CV.    - Beeswarm & Feature Rank   - 13 Navigation Modules:
  Holt Exponential Smoothing,       - Soft-voting & weighted         - Local Waterfall Plots       Homepage, Dataset Explorer,
  Autoregressive, Recurrent.          averaging.                     - Partial Dependence (PDP)    Predictions, Classification,
- Horizons: h=1, h=6, h=24.         - Meta-models: Ridge, Logistic.  - Permutation Importance.     Anomaly, Forecasting, XAI, etc.
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
                                      [9. MLOPS & MODEL REGISTRY]
                                       - Artifact versioning (v2)
                                       - Experiment tracking & metadata
                                       - Conformal prediction intervals
                                       - 18+ automated pytest suite
```

---

## 1. AI/ML Task Taxonomy

TURBINE-X 2.0 formulates gas turbine monitoring across four fundamental learning paradigms:

1. **Supervised Regression**:
   - **Track A (Operational Emission Prediction)**: Predicts continuous emission concentrations ($CO$ and $NO_x$ in $\text{mg/m}^3$) conditioned on ambient weather and 6 turbine state variables including generator load ($TEY$).
   - **Track B (Joint Multi-Target Prediction)**: Predicts simultaneous turbine performance ($TEY$ in $\text{MW}$) and environmental emissions ($CO, NO_x$) using only ambient conditions and compressor/combustor dynamics ($TEY$ strictly excluded from input vector).

2. **Supervised Multi-Class Risk Classification**:
   - Categorizes continuous emissions into actionable industrial alarm levels:
     - **Class 0 (Normal Operating Range)**: Stable combustion within baseline limits.
     - **Class 1 (Elevated Emissions)**: Incipient flame drift, load changes, or thermal variation.
     - **Class 2 (High Risk / Emission Exceedance)**: Critical operating regime requiring operator intervention or burner trimming.
   - Thresholds are established **strictly from the empirical distributions of the Development partition (2011–2013)** to preserve zero leakage into locked test evaluation.

3. **Unsupervised Anomaly Detection**:
   - Detects abnormal thermodynamic states, sensor degradation, filter blockages, or uncharacteristic combustor profiles without requiring labelled fault incidents.
   - Establishes a reference envelope from nominal 2011–2013 operations using Isolation Forests, Local Outlier Factors, One-Class SVMs, Robust Elliptic Envelopes, and Deep Bottleneck Autoencoders.

4. **Sequential & Deep Temporal Learning**:
   - Evaluates short-term multi-step temporal forecasting ($h=1, 6, 24$ hours) using exponential smoothing, autoregression, 1D-CNN temporal encoders, and recurrent sequence predictors.
   - Evaluates deep multilayer perceptrons with dropout, early stopping, and non-linear latent representations.

---

## 2. Chronological Data Flow & Leakage Guardrails

The benchmark enforces strict chronological boundaries across all tasks:

```
Combined Dataset: N = 36,733 observations, 11 physical sensor variables (2011–2015)
│
├── Development Set (2011–2013, N = 22,191 rows, 60.4%):
│   ├── Used exclusively for transformer fitting (StandardScaler, Imputers)
│   ├── Used for base model training (Regression, Classification, Anomaly reference)
│   ├── Used for expanding-window cross-validation (2011->2012, 2011-2012->2013)
│   ├── Used for out-of-fold meta-feature generation in Stacking ensembles
│   └── Used for Optuna Bayesian hyperparameter search
│
├── Calibration Set (2014, N = 7,158 rows, 19.5%):
│   ├── Held-out validation anchor
│   └── Calibration set for 90% Split-Conformal Prediction Intervals
│
└── Locked Test Set (2015, N = 7,384 rows, 20.1%):
    ├── Zero access during feature engineering, threshold definition, or model selection
    └── Immutable final evaluation benchmark for all regression, classification, and anomaly models
```

---

## 3. Model Deployment & Dashboard Architecture

The production interface is built in Streamlit (`app.py`), structured into a 13-module navigation system:

1. **Home / Hero Portal**: Industrial KPI cards, overall platform statistics, model summary, and safety disclaimers.
2. **Dataset Explorer**: Interactive exploration of physical features, correlations, and annual distributions.
3. **Single Prediction**: Point predictions and 90% conformal intervals for Tracks A and B.
4. **Batch Prediction**: Upload unlabelled CSVs, perform batch vector inference, and download enriched CSVs.
5. **Emission Classification**: Multi-class risk prediction (Normal, Elevated, High Risk) with class probability gauges.
6. **Anomaly Detection**: Real-time anomaly scoring, outlier flags, and reconstruction error diagnostics.
7. **Forecasting**: Multi-step rolling sequence forecasts for CO, NOx, and TEY.
8. **Deep Learning**: Inspection of neural network architectures, training curves, and validation loss profiles.
9. **Model Comparison**: Interactive leaderboards for all 16 regressors and 14 classifiers across metrics.
10. **Explainable AI (XAI)**: Global TreeSHAP beeswarm charts, local waterfall attributions, and PDP curves.
11. **Uncertainty Analysis**: 90% split-conformal calibration diagnostics and domain-drift analysis.
12. **Experiment Tracking**: Model registry, artifact inspection, and reproducibility parameters.
13. **About TURBINE-X**: System architecture, project citations, safety notices, and documentation index.

---

## 4. Known Dataset Limitations & Research Transparency

1. **NOx Non-Stationary Baseline Drift**:
   As documented in our audit, power plant NOx emissions experienced an abrupt $-8.89\text{ mg/m}^3$ baseline reduction in 2014, leading to negative uncentered $R^2$ on 2015 test data despite strong dynamic tracking ($\text{EV} \approx +0.73$). TURBINE-X 2.0 transparently reports this phenomenon and provides bias-corrected metrics.
2. **Lack of Sub-Hourly Timestamps**:
   The UCI Gas Turbine dataset provides chronological hourly observations without ISO-8601 calendar timestamps. Sequential models use row ordering as an index proxy while noting that micro-level sampling gaps cannot be verified.
3. **Unsupervised Anomaly Assumptions**:
   Because the dataset lacks ground-truth maintenance logs or verified outage labels, anomaly flags represent statistical outliers in the 11-dimensional thermodynamic space rather than certified mechanical failures.

