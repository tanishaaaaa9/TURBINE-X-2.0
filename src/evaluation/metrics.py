"""TURBINE-X Comprehensive Regression Metrics Module.
Calculates per-target and multi-output aggregate regression metrics:
MAE, MSE, RMSE, R2, Explained Variance, MedAE, MAPE, latency, and memory footprints.
"""
from typing import Dict, List, Any, Union
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error
)


def compute_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_name: str) -> Dict[str, float]:
    """Compute individual target regression metrics."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    # Base error measures
    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(yt, yp))
    evs = float(explained_variance_score(yt, yp))
    medae = float(median_absolute_error(yt, yp))

    # Safe MAPE calculation (avoiding divide-by-zero on small emissions)
    mask = np.abs(yt) > 1e-4
    if np.sum(mask) > 0:
        mape = float(mean_absolute_percentage_error(yt[mask], yp[mask]) * 100.0)
    else:
        mape = float("nan")

    max_error = float(np.max(np.abs(yt - yp)))

    return {
        "target": target_name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Explained_Variance": round(evs, 4),
        "MedAE": round(medae, 4),
        "MAPE_pct": round(mape, 2) if not np.isnan(mape) else None,
        "Max_Error": round(max_error, 4)
    }


def regression_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    targets: List[str]
) -> List[Dict[str, Any]]:
    """Compute regression metrics across all targets."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    rows = []
    for i, target in enumerate(targets):
        rows.append(compute_target_metrics(yt[:, i], yp[:, i], target))

    return rows


def compute_multioutput_aggregates(metrics_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute uniform average metrics across targets."""
    avg_mae = float(np.mean([r["MAE"] for r in metrics_rows]))
    avg_rmse = float(np.mean([r["RMSE"] for r in metrics_rows]))
    avg_r2 = float(np.mean([r["R2"] for r in metrics_rows]))

    return {
        "Average_MAE": round(avg_mae, 4),
        "Average_RMSE": round(avg_rmse, 4),
        "Average_R2": round(avg_r2, 4)
    }
