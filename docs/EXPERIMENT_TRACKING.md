# TURBINE-X: Experiment Tracking & Artifact Management

## 1. Artifact Organization Structure
Experiment artifacts are structured to ensure 100% reproducibility and clear separation between quick validation runs and locked research benchmarks:

```
artifacts/
├── models/
│   ├── A_emissions/
│   │   ├── Dummy_mean.joblib
│   │   ├── LinearRegression.joblib
│   │   ├── Ridge.joblib
│   │   ├── ElasticNet.joblib
│   │   ├── RandomForest.joblib
│   │   ├── HistGradientBoosting.joblib
│   │   └── XGBoost.joblib
│   └── B_joint/
│       ├── Dummy_mean.joblib
│       ├── LinearRegression.joblib
│       ├── Ridge.joblib
│       ├── ElasticNet.joblib
│       ├── RandomForest.joblib
│       ├── HistGradientBoosting.joblib
│       └── XGBoost.joblib
├── metrics/
│   ├── locked_test_metrics.csv
│   └── development_temporal_cv.csv
└── metadata/
    └── model_metadata.json
```

---

## 2. Serialized Estimator Bundle Specification
Every model serialized via Joblib stores a comprehensive python dictionary bundle:
- `estimator`: The fitted Scikit-Learn `Pipeline` object containing any imputers, scalers, and regressors.
- `features`: Exact ordered list of input feature names required for inference.
- `targets`: List of output target names (`["CO", "NOX"]` or `["CO", "NOX", "TEY"]`).
- `model_name`: Architecture string (e.g., `"HistGradientBoosting"`).
- `track`: Track key (`"A_emissions"` or `"B_joint"`).
- `conformal_alpha`: Nominal miscoverage rate (default: `0.10`).
- `conformal_quantiles`: Calibrated error thresholds $q_{1-\alpha}$ computed on empirical 2014 validation errors.
- `metrics`: Evaluation metrics computed on 2015 locked test data.

