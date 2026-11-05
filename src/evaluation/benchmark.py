"""TURBINE-X Comprehensive Model Benchmark Engine.
Executes training, evaluation, timing, conformal calibration, and artifact persistence.
"""
from pathlib import Path
import time
import os
import joblib
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data.loader import (
    load_raw_yearly_data,
    FEATURES_A, TARGETS_A,
    FEATURES_B, TARGETS_B
)
from src.data.splitting import temporal_split, get_temporal_cv_folds
from src.models.model_registry import get_benchmark_models
from src.evaluation.metrics import regression_metrics, compute_multioutput_aggregates
from src.evaluation.conformal import conformal_quantiles, interval_metrics


class BenchmarkRunner:
    """Orchestrates comprehensive benchmarking across models and prediction tracks."""

    def __init__(self, root_dir: Optional[Path] = None, seed: int = 42, quick: bool = False):
        self.root_dir = root_dir or Path.cwd()
        self.seed = seed
        self.quick = quick
        self.models = get_benchmark_models(quick=quick, seed=seed)

    def run(self) -> Dict[str, pd.DataFrame]:
        """Execute full benchmark pipeline."""
        raw_df = load_raw_yearly_data(self.root_dir / "data" / "raw")
        dev_df, cal_df, test_df = temporal_split(raw_df)

        tracks = [
            ("Track_A_emissions", FEATURES_A, TARGETS_A),
            ("Track_B_joint", FEATURES_B, TARGETS_B)
        ]

        test_records: List[Dict[str, Any]] = []
        cv_records: List[Dict[str, Any]] = []

        models_dir = self.root_dir / "artifacts" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        # Also maintain compatibility with models/ directory
        (self.root_dir / "models").mkdir(parents=True, exist_ok=True)

        for track_name, features, targets in tracks:
            track_model_dir = models_dir / track_name
            track_model_dir.mkdir(parents=True, exist_ok=True)
            compat_model_dir = self.root_dir / "models" / track_name
            compat_model_dir.mkdir(parents=True, exist_ok=True)

            X_dev, y_dev = dev_df[features], dev_df[targets].values
            X_cal, y_cal = cal_df[features], cal_df[targets].values
            X_test, y_test = test_df[features], test_df[targets].values

            for name, estimator in self.models.items():
                # 1. Fit on development set (2011-2013)
                t0 = time.perf_counter()
                fitted = clone(estimator)
                try:
                    fitted.fit(X_dev, y_dev)
                    train_time_sec = time.perf_counter() - t0

                    # 2. Timing & Predict on test set (2015)
                    t_inf = time.perf_counter()
                    pred_test = fitted.predict(X_test)
                    latency_ms = (time.perf_counter() - t_inf) / len(test_df) * 1000.0

                    # 3. Calibration predictions & conformal quantiles (2014)
                    pred_cal = fitted.predict(X_cal)
                    q = conformal_quantiles(y_cal, pred_cal, alpha=0.10)
                    ci = interval_metrics(y_test, pred_test, q)

                    # 4. Metric calculation
                    metrics = regression_metrics(y_test, pred_test, targets)
                    aggs = compute_multioutput_aggregates(metrics)

                    # 5. Model serialization
                    bundle = {
                        "estimator": fitted,
                        "features": features,
                        "targets": targets,
                        "track": track_name,
                        "model_name": name,
                        "conformal_alpha": 0.10,
                        "conformal_quantiles": q,
                        "metrics": metrics,
                        "aggregates": aggs,
                        "train_time_sec": round(train_time_sec, 3),
                        "latency_ms_per_row": round(latency_ms, 5)
                    }
                    save_path = track_model_dir / f"{name}.joblib"
                    joblib.dump(bundle, save_path)
                    joblib.dump(bundle, compat_model_dir / f"{name}.joblib")
                    model_size_kb = os.path.getsize(save_path) / 1024.0

                    for m in metrics:
                        j = targets.index(m["target"])
                        test_records.append({
                            "track": track_name,
                            "model": name,
                            "train_years": "2011-2013",
                            "calibration_year": "2014",
                            "test_year": "2015",
                            "train_rows": len(dev_df),
                            "test_rows": len(test_df),
                            "train_time_sec": round(train_time_sec, 3),
                            "latency_ms_per_row": round(latency_ms, 4),
                            "model_size_kb": round(model_size_kb, 1),
                            "conformal_90_coverage": ci["marginal_coverage"][j],
                            "conformal_90_width": ci["mean_interval_width"][j],
                            "conformal_simultaneous_coverage": ci["simultaneous_coverage"],
                            **m
                        })
                except Exception as exc:
                    test_records.append({
                        "track": track_name,
                        "model": name,
                        "error": str(exc)
                    })

            # Within-development temporal CV
            cv_folds = get_temporal_cv_folds(dev_df)
            for name, estimator in self.models.items():
                for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds, start=1):
                    tr_sub = dev_df.iloc[tr_idx]
                    va_sub = dev_df.iloc[va_idx]
                    try:
                        fitted = clone(estimator)
                        fitted.fit(tr_sub[features], tr_sub[targets].values)
                        preds = fitted.predict(va_sub[features])
                        m_cv = regression_metrics(va_sub[targets].values, preds, targets)
                        for m in m_cv:
                            cv_records.append({
                                "track": track_name,
                                "model": name,
                                "fold": fold_idx,
                                "train_years": f"2011-{2010+fold_idx}",
                                "val_year": 2011 + fold_idx,
                                **m
                            })
                    except Exception as exc:
                        cv_records.append({
                            "track": track_name,
                            "model": name,
                            "fold": fold_idx,
                            "error": str(exc)
                        })

        df_test = pd.DataFrame(test_records)
        df_cv = pd.DataFrame(cv_records)

        # Save to outputs
        tables_dir = self.root_dir / "outputs" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        df_test.to_csv(tables_dir / "locked_test_metrics.csv", index=False)
        df_cv.to_csv(tables_dir / "development_temporal_cv.csv", index=False)

        # Also save to artifacts/metrics
        art_metrics = self.root_dir / "artifacts" / "metrics"
        art_metrics.mkdir(parents=True, exist_ok=True)
        df_test.to_csv(art_metrics / "locked_test_metrics.csv", index=False)
        df_cv.to_csv(art_metrics / "development_temporal_cv.csv", index=False)

        return {"locked_test": df_test, "temporal_cv": df_cv}

