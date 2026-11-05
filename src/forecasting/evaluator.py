"""TURBINE-X 2.0 Time-Series Forecasting & Sequential Evaluation Module.
Implements rolling-origin sequential forecasting across multi-step horizons (h=1, 6, 24).
Provides Persistence, Rolling Moving Average, Holt Exponential Smoothing, Autoregressive,
and Recurrent Sequence Predictors.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def calculate_mase(y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Scaled Error (MASE) against in-sample naive persistence."""
    mae_model = mean_absolute_error(y_test, y_pred)
    mae_naive_insample = np.mean(np.abs(np.diff(y_train)))
    if mae_naive_insample == 0:
        return 1.0
    return float(mae_model / mae_naive_insample)


def forecast_persistence(y_series: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Persistence baseline: forecast y[t+h] using y[t]."""
    preds = np.zeros(len(y_series) - horizon)
    for i in range(len(preds)):
        preds[i] = y_series[i]
    return preds


def forecast_moving_average(y_series: np.ndarray, window: int = 6, horizon: int = 1) -> np.ndarray:
    """Rolling moving average baseline forecast."""
    preds = []
    for i in range(window, len(y_series) - horizon + 1):
        preds.append(np.mean(y_series[i-window:i]))
    return np.array(preds)


def forecast_holt_exponential_smoothing(y_series: np.ndarray, alpha: float = 0.3, beta: float = 0.1, horizon: int = 1) -> np.ndarray:
    """Holt linear exponential smoothing for level and trend."""
    level = y_series[0]
    trend = y_series[1] - y_series[0] if len(y_series) > 1 else 0.0
    preds = []
    
    for i in range(len(y_series) - horizon):
        val = y_series[i]
        last_level = level
        level = alpha * val + (1.0 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1.0 - beta) * trend
        pred_h = level + horizon * trend
        preds.append(pred_h)
    return np.array(preds)


def forecast_autoregressive_lags(y_train: np.ndarray, y_test: np.ndarray, lags: int = 12, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Supervised autoregressive lag regression model."""
    def create_lag_features(arr, p, h):
        X, y = [], []
        for i in range(p, len(arr) - h + 1):
            X.append(arr[i-p:i])
            y.append(arr[i+h-1])
        return np.array(X), np.array(y)

    X_tr, y_tr = create_lag_features(y_train, lags, horizon)
    # Combine last train elements with test to create test features
    combined = np.concatenate([y_train[-lags:], y_test])
    X_te, y_actual = create_lag_features(combined, lags, horizon)

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return y_actual, y_pred


def run_forecasting_benchmark(train_series: np.ndarray, test_series: np.ndarray, target_name: str, horizon: int = 1) -> List[Dict[str, Any]]:
    """Run full suite of forecasting baselines on a single target across horizon h."""
    results = []
    y_true_pers = test_series[horizon:]
    y_pred_pers = forecast_persistence(test_series, horizon=horizon)

    mae_p = mean_absolute_error(y_true_pers, y_pred_pers)
    rmse_p = float(np.sqrt(mean_squared_error(y_true_pers, y_pred_pers)))
    r2_p = float(r2_score(y_true_pers, y_pred_pers))
    mase_p = calculate_mase(train_series, y_true_pers, y_pred_pers)

    results.append({
        "target": target_name,
        "model": "Persistence (Naive)",
        "horizon_hours": horizon,
        "MAE": round(mae_p, 4),
        "RMSE": round(rmse_p, 4),
        "R2": round(r2_p, 4),
        "MASE": round(mase_p, 4),
        "improvement_over_naive_pct": 0.0
    })

    # Moving Average (window=6)
    w = 6
    y_true_ma = test_series[w + horizon - 1:]
    y_pred_ma = forecast_moving_average(test_series, window=w, horizon=horizon)
    min_len = min(len(y_true_ma), len(y_pred_ma))
    y_true_ma = y_true_ma[:min_len]
    y_pred_ma = y_pred_ma[:min_len]
    
    mae_ma = mean_absolute_error(y_true_ma, y_pred_ma)
    rmse_ma = float(np.sqrt(mean_squared_error(y_true_ma, y_pred_ma)))
    r2_ma = float(r2_score(y_true_ma, y_pred_ma))
    mase_ma = calculate_mase(train_series, y_true_ma, y_pred_ma)
    imp_ma = round((mae_p - mae_ma) / mae_p * 100, 2)

    results.append({
        "target": target_name,
        "model": "Rolling Moving Average (w=6)",
        "horizon_hours": horizon,
        "MAE": round(mae_ma, 4),
        "RMSE": round(rmse_ma, 4),
        "R2": round(r2_ma, 4),
        "MASE": round(mase_ma, 4),
        "improvement_over_naive_pct": imp_ma
    })

    # Holt Exponential Smoothing
    y_pred_holt = forecast_holt_exponential_smoothing(test_series, alpha=0.35, beta=0.08, horizon=horizon)
    y_true_holt = test_series[horizon:]
    mae_h = mean_absolute_error(y_true_holt, y_pred_holt)
    rmse_h = float(np.sqrt(mean_squared_error(y_true_holt, y_pred_holt)))
    r2_h = float(r2_score(y_true_holt, y_pred_holt))
    mase_h = calculate_mase(train_series, y_true_holt, y_pred_holt)
    imp_h = round((mae_p - mae_h) / mae_p * 100, 2)

    results.append({
        "target": target_name,
        "model": "Holt Exponential Smoothing",
        "horizon_hours": horizon,
        "MAE": round(mae_h, 4),
        "RMSE": round(rmse_h, 4),
        "R2": round(r2_h, 4),
        "MASE": round(mase_h, 4),
        "improvement_over_naive_pct": imp_h
    })

    # Autoregressive Lag Regression (p=12 lags)
    y_true_ar, y_pred_ar = forecast_autoregressive_lags(train_series, test_series, lags=12, horizon=horizon)
    mae_ar = mean_absolute_error(y_true_ar, y_pred_ar)
    rmse_ar = float(np.sqrt(mean_squared_error(y_true_ar, y_pred_ar)))
    r2_ar = float(r2_score(y_true_ar, y_pred_ar))
    mase_ar = calculate_mase(train_series, y_true_ar, y_pred_ar)
    imp_ar = round((mae_p - mae_ar) / mae_p * 100, 2)

    results.append({
        "target": target_name,
        "model": "Autoregressive Lag Ridge (p=12)",
        "horizon_hours": horizon,
        "MAE": round(mae_ar, 4),
        "RMSE": round(rmse_ar, 4),
        "R2": round(r2_ar, 4),
        "MASE": round(mase_ar, 4),
        "improvement_over_naive_pct": imp_ar
    })

    return results

