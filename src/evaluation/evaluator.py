"""TURBINE-X Evaluator Module.
Standardized evaluator for single-output and multi-output regression pipelines.
Computes comprehensive metrics, conformal bounds, and inference latency.
"""
from typing import Dict, List, Any, Optional
import time
import numpy as np
import pandas as pd
from src.evaluation.metrics import regression_metrics, compute_multioutput_aggregates
from src.evaluation.conformal import conformal_quantiles, interval_metrics


class ModelEvaluator:
    """Evaluates fitted pipelines against validation and test sets."""

    def __init__(self, targets: List[str], conformal_alpha: float = 0.10):
        self.targets = targets
        self.conformal_alpha = conformal_alpha

    def evaluate(
        self,
        fitted_pipeline,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        X_cal: Optional[pd.DataFrame] = None,
        y_cal: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Perform full evaluation of model predictions."""
        # 1. Inference latency measurement
        t0 = time.perf_counter()
        y_pred = fitted_pipeline.predict(X_test)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_row_ms = elapsed_ms / len(X_test)

        # 2. Regression metrics
        metrics = regression_metrics(y_test, y_pred, self.targets)
        aggregates = compute_multioutput_aggregates(metrics)

        # 3. Conformal intervals if calibration set provided
        conformal_info = None
        if X_cal is not None and y_cal is not None:
            pred_cal = fitted_pipeline.predict(X_cal)
            q = conformal_quantiles(y_cal, pred_cal, alpha=self.conformal_alpha)
            ci = interval_metrics(y_test, y_pred, q)
            conformal_info = {
                "alpha": self.conformal_alpha,
                "quantiles": q,
                "marginal_coverage": ci["marginal_coverage"],
                "simultaneous_coverage": ci["simultaneous_coverage"],
                "mean_interval_width": ci["mean_interval_width"]
            }

        return {
            "metrics": metrics,
            "aggregates": aggregates,
            "latency_total_ms": round(elapsed_ms, 3),
            "latency_per_row_ms": round(per_row_ms, 5),
            "conformal": conformal_info,
            "predictions": y_pred
        }

