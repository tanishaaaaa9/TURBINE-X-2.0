"""TURBINE-X Evaluation Metrics & Conformal Uncertainty Unit Tests."""
import pytest
import numpy as np
from src.evaluation.metrics import regression_metrics, compute_target_metrics
from src.evaluation.conformal import conformal_quantiles, interval_metrics


def test_regression_metrics_calculation():
    y_true = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    y_pred = np.array([[11.0, 95.0], [19.0, 205.0], [31.0, 290.0]])
    targets = ["CO", "NOX"]

    metrics = regression_metrics(y_true, y_pred, targets)
    assert len(metrics) == 2
    assert metrics[0]["target"] == "CO"
    assert metrics[0]["MAE"] == 1.0
    assert metrics[1]["target"] == "NOX"
    assert metrics[1]["MAE"] == 6.6667 or np.isclose(metrics[1]["MAE"], 6.6667, atol=1e-3)


def test_conformal_quantiles_and_intervals():
    y_cal = np.array([[10.0, 50.0], [20.0, 60.0], [30.0, 70.0]])
    pred_cal = np.array([[10.5, 49.0], [19.8, 61.2], [30.1, 69.5]])
    q = conformal_quantiles(y_cal, pred_cal, alpha=0.10)
    assert len(q) == 2
    assert q[0] > 0
    assert q[1] > 0

    # Test intervals on evaluation set
    y_test = np.array([[10.2, 50.5]])
    pred_test = np.array([[10.0, 50.0]])
    ci = interval_metrics(y_test, pred_test, q)
    assert "marginal_coverage" in ci
    assert "simultaneous_coverage" in ci

