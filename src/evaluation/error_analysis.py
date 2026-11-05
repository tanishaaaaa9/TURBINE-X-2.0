"""TURBINE-X Error Analysis Module.
Profiles prediction errors, identifies worst-performing operating regimes, and evaluates error percentiles.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd


def compute_error_profile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_features: pd.DataFrame,
    targets: List[str],
    top_k_worst: int = 10
) -> Dict[str, Any]:
    """Profile error distributions and identify extreme outlier regimes."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    abs_errors = np.abs(yt - yp)
    profile = {}

    for j, tgt in enumerate(targets):
        err = abs_errors[:, j]
        p50 = float(np.percentile(err, 50))
        p90 = float(np.percentile(err, 90))
        p95 = float(np.percentile(err, 95))
        p99 = float(np.percentile(err, 99))
        max_err = float(np.max(err))

        # Identify worst indices
        worst_indices = np.argsort(err)[-top_k_worst:][::-1]
        worst_cases = []
        for idx in worst_indices:
            row_dict = X_features.iloc[idx].to_dict() if idx < len(X_features) else {}
            row_dict["actual"] = float(yt[idx, j])
            row_dict["predicted"] = float(yp[idx, j])
            row_dict["absolute_error"] = float(err[idx])
            worst_cases.append(row_dict)

        profile[tgt] = {
            "p50_error": round(p50, 4),
            "p90_error": round(p90, 4),
            "p95_error": round(p95, 4),
            "p99_error": round(p99, 4),
            "max_error": round(max_err, 4),
            "worst_cases": worst_cases
        }

    return profile

