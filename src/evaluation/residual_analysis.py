"""TURBINE-X Residual & Error Analysis Module.
Analyzes residual distributions, biases, variance stability, and extreme error cases.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str]) -> Dict[str, Any]:
    """Calculate thorough residual diagnostics for each target."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    residuals = yt - yp  # e = y - \hat{y}
    analysis = {}

    for i, target in enumerate(targets):
        res = residuals[:, i]
        mean_res = float(np.mean(res))
        std_res = float(np.std(res))
        skew_res = float(stats.skew(res))
        kurt_res = float(stats.kurtosis(res))

        # D'Agostino-Pearson normality test on residuals
        stat_norm, p_norm = stats.normaltest(res) if len(res) >= 20 else (0.0, 1.0)

        # Fraction of residuals within 1 and 2 standard deviations
        within_1sigma = float(np.mean(np.abs(res - mean_res) <= std_res))
        within_2sigma = float(np.mean(np.abs(res - mean_res) <= 2 * std_res))

        analysis[target] = {
            "mean_residual_bias": round(mean_res, 4),
            "residual_std": round(std_res, 4),
            "residual_skewness": round(skew_res, 4),
            "residual_kurtosis": round(kurt_res, 4),
            "d_agostino_normality_p": round(float(p_norm), 6),
            "pct_within_1sigma": round(within_1sigma * 100, 2),
            "pct_within_2sigma": round(within_2sigma * 100, 2),
            "p05_residual": round(float(np.percentile(res, 5)), 4),
            "p50_residual": round(float(np.percentile(res, 50)), 4),
            "p95_residual": round(float(np.percentile(res, 95)), 4)
        }

    return analysis

