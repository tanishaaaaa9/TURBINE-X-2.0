# TURBINE-X 2.0 — Model Registry & MLOps Governance Report

## 1. Registry Architecture & Versioning Standards

The TURBINE-X 2.0 Model Registry organizes machine learning artifacts across four functional task families under an immutable semantic versioning standard (`v2.0`):

- `artifacts/models/v2/regression/` (16 model families across Track A and Track B)
- `artifacts/models/v2/classification/` (14 model families across CO and NOx risk tiers)
- `artifacts/models/v2/anomaly/` (6 unsupervised reference detectors)
- `models/final_model.joblib` (Consolidated production bundle with conformal bounds)

```
Model Versioning Scheme:
  [Algorithm]_[Task]_[Track]_[v2.0].joblib
  Example: HistGradientBoosting_Regression_TrackA_v2.0.joblib
```

---

## 2. Supervised Regression Registry (Locked 2015 Test Set)

| Model Name | Task / Track | Architecture Details | Parameters / Estimators | Artifact Path | Test R² | Test MAE | Latency (ms/row) | Production Status |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **ExtraTrees** | Reg / Track A | `Pipeline(StandardScaler -> ExtraTreesRegressor)` | $n=150$, depth=14 | `artifacts/models/v2/regression/A_emissions/ExtraTrees.joblib` | **0.5769** | **0.8855** | 0.0057 | **CHAMPION (CO)** |
| **CatBoost** | Reg / Track A | `Pipeline(StandardScaler -> CatBoostRegressor)` | iter=150, depth=6 | `artifacts/models/v2/regression/A_emissions/CatBoost.joblib` | **0.5487** | **0.9024** | 0.0010 | **CONTENDER** |
| **HistGradientBoosting** | Reg / Track A | `Pipeline(StandardScaler -> HistGradientBoosting)` | iter=150, leaves=31 | `artifacts/models/v2/regression/A_emissions/HistGradientBoosting.joblib` | 0.4980 | 0.9508 | 0.0019 | **PRODUCTION BUNDLE** |
| **XGBoost** | Reg / Track B | `Pipeline(StandardScaler -> XGBRegressor)` | $n=150$, depth=6 | `artifacts/models/v2/regression/B_joint/XGBoost.joblib` | **0.9890** | **1.4070** | 0.0015 | **CHAMPION (TEY)** |
| **HistGradientBoosting** | Reg / Track B | `Pipeline(StandardScaler -> HistGradientBoosting)` | iter=150, leaves=31 | `artifacts/models/v2/regression/B_joint/HistGradientBoosting.joblib` | **0.9881** | **1.4000** | 0.0029 | **CHAMPION (TEY)** |
| **StackingRegressor** | Reg / Track A | `Pipeline(StandardScaler -> StackingRegressor)` | RF+HGB+Ridge | `artifacts/models/v2/regression/A_emissions/StackingRegressor.joblib` | 0.5210 | 0.9320 | 0.0133 | VERIFIED |
| **VotingRegressor** | Reg / Track A | `Pipeline(StandardScaler -> VotingRegressor)` | RF+ET+HGB | `artifacts/models/v2/regression/A_emissions/VotingRegressor.joblib` | 0.5412 | 0.9105 | 0.0265 | VERIFIED |
| **MLPRegressor** | Reg / Track A | `Pipeline(StandardScaler -> MLPRegressor)` | (64, 32), ReLU, Adam | `artifacts/models/v2/regression/A_emissions/MLPRegressor.joblib` | 0.4650 | 1.0820 | 0.0006 | VERIFIED |
| **LightGBM** | Reg / Track A | `Pipeline(StandardScaler -> LGBMRegressor)` | $n=150$, depth=6 | `artifacts/models/v2/regression/A_emissions/LightGBM.joblib` | 0.4890 | 0.9650 | 0.0019 | VERIFIED |

---

## 3. Supervised Classification Registry (Multi-Class Risk)

| Model Name | Target | Architecture | Macro F1 | Test Accuracy | Artifact Path | Latency (ms/row) | Production Status |
| :--- | :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **MLPClassifier** | CO Risk | `Pipeline(StandardScaler -> MLPClassifier)` | **0.441** | **58.64%** | `artifacts/models/v2/classification/CO_risk/MLPClassifier.joblib` | 0.0008 | **CHAMPION (CO Risk)** |
| **StackingClassifier** | CO Risk | `Pipeline(StandardScaler -> StackingClassifier)` | **0.437** | **55.34%** | `artifacts/models/v2/classification/CO_risk/StackingClassifier.joblib` | 0.0052 | **CONTENDER** |
| **HistGradientBoosting** | CO Risk | `Pipeline(StandardScaler -> HistGradientBoosting)` | 0.398 | 53.21% | `artifacts/models/v2/classification/CO_risk/HistGradientBoostingClassifier.joblib` | 0.0022 | PRODUCTION BUNDLE |
| **LogisticRegression** | NOX Risk | `Pipeline(StandardScaler -> LogisticRegression)` | **0.340** | **45.40%** | `artifacts/models/v2/classification/NOX_risk/LogisticRegression.joblib` | 0.0001 | **BASELINE** |
| **KNNClassifier** | NOX Risk | `Pipeline(StandardScaler -> KNeighborsClassifier)` | **0.390** | **44.56%** | `artifacts/models/v2/classification/NOX_risk/KNNClassifier.joblib` | 0.0150 | **CHAMPION (NOx Risk)** |
| **ExtraTreesClassifier** | NOX Risk | `Pipeline(StandardScaler -> ExtraTreesClassifier)` | 0.359 | 44.60% | `artifacts/models/v2/classification/NOX_risk/ExtraTreesClassifier.joblib` | 0.0040 | VERIFIED |

---

## 4. Unsupervised Anomaly Detection Registry

| Model Name | Reference Training | Test Anomaly Rate | Latency (ms/row) | Stored Artifact Path | Operational Purpose |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **IsolationForest** | Dev (2011–13) | **5.74%** | 0.0012 | `artifacts/models/v2/anomaly/IsolationForest.joblib` | Tree-partitioned tree isolation |
| **LocalOutlierFactor** | Dev (2011–13) | **34.98%** | 0.0250 | `artifacts/models/v2/anomaly/LocalOutlierFactor.joblib` | Local k-NN density divergence |
| **OneClassSVM** | Dev (2011–13) | **15.33%** | 0.0820 | `artifacts/models/v2/anomaly/OneClassSVM.joblib` | RBF support vector boundary |
| **DeepAutoencoder** | Dev (2011–13) | **99.96%** | 0.0009 | `artifacts/models/v2/anomaly/DeepAutoencoder.joblib` | Neural bottleneck reconstruction error |
| **EllipticEnvelope** | Dev (2011–13) | **99.99%** | 0.0004 | `artifacts/models/v2/anomaly/EllipticEnvelope.joblib` | Robust Mahalanobis covariance envelope |

---

## 5. Artifact Integrity & Reproducibility Verification

1. **Deterministic Seed**: All stochastic algorithms locked to seed `42`.
2. **Serialization Format**: High-efficiency Python `joblib` compression (`protocol=4`).
3. **Automated Smoke Tests**: Verified via `tests/test_v2_suite.py` asserting non-empty binaries, correct input/output feature vectors, and sub-millisecond inference speeds.

