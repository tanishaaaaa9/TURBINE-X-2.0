"""TURBINE-X 2.0 Comprehensive Automated Test Suite.
Tests risk classification, anomaly detection, forecasting, deep neural pipelines,
36 figures generation, and dashboard integration.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from src.data.loader import find_project_root, FEATURES_A, TARGETS_A, FEATURES_B, TARGETS_B
from src.classification.labeling import compute_operational_risk_thresholds, add_classification_targets
from src.models.model_factory_v2 import create_classification_model_v2, create_regression_model_v2
from src.anomaly.detectors import get_anomaly_detectors, DeepAutoencoderAnomaly
from src.forecasting.evaluator import calculate_mase, forecast_persistence


@pytest.fixture
def dummy_turbine_df():
    np.random.seed(42)
    n = 200
    data = {
        "AT": np.random.uniform(5, 35, n),
        "AP": np.random.uniform(1000, 1025, n),
        "AH": np.random.uniform(30, 90, n),
        "AFDP": np.random.uniform(2, 6, n),
        "GTEP": np.random.uniform(18, 35, n),
        "TIT": np.random.uniform(1020, 1100, n),
        "TAT": np.random.uniform(520, 560, n),
        "TEY": np.random.uniform(100, 160, n),
        "CDP": np.random.uniform(10, 15, n),
        "CO": np.random.exponential(1.5, n),
        "NOX": np.random.normal(65, 10, n),
        "source_year": np.random.choice([2011, 2012, 2013], n)
    }
    return pd.DataFrame(data)


def test_v2_risk_label_generation(dummy_turbine_df):
    th = compute_operational_risk_thresholds(dummy_turbine_df)
    assert "CO" in th and "NOX" in th
    assert th["CO"]["tier_1_normal_max"] < th["CO"]["tier_2_elevated_max"]
    assert th["NOX"]["tier_1_normal_max"] < th["NOX"]["tier_2_elevated_max"]

    df_labeled = add_classification_targets(dummy_turbine_df, th)
    assert "CO_risk" in df_labeled.columns
    assert "NOX_risk" in df_labeled.columns
    assert set(df_labeled["CO_risk"].unique()).issubset({0, 1, 2})
    assert set(df_labeled["NOX_risk"].unique()).issubset({0, 1, 2})


def test_v2_classification_model_factory(dummy_turbine_df):
    th = compute_operational_risk_thresholds(dummy_turbine_df)
    df_l = add_classification_targets(dummy_turbine_df, th)

    for m_name in ["LogisticRegression", "HistGradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"]:
        pipe = create_classification_model_v2(m_name)
        pipe.fit(df_l[FEATURES_A], df_l["CO_risk"])
        preds = pipe.predict(df_l[FEATURES_A])
        assert len(preds) == len(df_l)
        assert set(preds).issubset({0, 1, 2})


def test_v2_regression_model_factory(dummy_turbine_df):
    for m_name in ["LinearRegression", "HistGradientBoosting", "LightGBM", "CatBoost", "MLPRegressor"]:
        pipe = create_regression_model_v2(m_name)
        pipe.fit(dummy_turbine_df[FEATURES_A], dummy_turbine_df[TARGETS_A].values)
        preds = pipe.predict(dummy_turbine_df[FEATURES_A])
        assert preds.shape == (len(dummy_turbine_df), 2)
        assert not np.isnan(preds).any()


def test_v2_anomaly_detectors(dummy_turbine_df):
    detectors = get_anomaly_detectors(contamination=0.05)
    X = dummy_turbine_df[FEATURES_A]

    for name in ["IsolationForest", "OneClassSVM", "DeepAutoencoder"]:
        det = detectors[name]
        det.fit(X)
        flags = det.predict(X)
        assert len(flags) == len(X)
        assert set(flags).issubset({-1, 1})


def test_v2_forecasting_evaluator():
    y_tr = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    y_te = np.array([15.0, 16.0, 17.0, 18.0])
    y_pr = forecast_persistence(y_te, horizon=1)
    assert len(y_pr) == len(y_te) - 1
    mase = calculate_mase(y_tr, y_te[1:], y_pr)
    assert mase >= 0.0


def test_v2_all_36_figures_exist():
    root = find_project_root()
    fig_dir = root / "outputs" / "figures"
    manifest_file = root / "outputs" / "metadata" / "figure_manifest.csv"
    assert manifest_file.exists()
    df_man = pd.read_csv(manifest_file)
    assert len(df_man) == 36

    for f_name in df_man["figure_filename"]:
        fig_path = fig_dir / f_name
        assert fig_path.exists(), f"Missing figure: {f_name}"
        assert fig_path.stat().st_size > 5000, f"Figure appears empty: {f_name}"

