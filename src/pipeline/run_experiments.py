"""TURBINE-X Main Experiment Runner.
Executes temporal evaluation, conformal calibration, model serialization, and metric reporting.
Supports both --quick (rapid validation) and full benchmark modes.
"""
from pathlib import Path
import argparse
import json
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.base import clone

from src.data.loader import (
    find_project_root,
    load_raw_yearly_data,
    FEATURES_A, TARGETS_A,
    FEATURES_B, TARGETS_B
)
from src.data.splitting import temporal_split, get_temporal_cv_folds
from src.models.model_registry import get_benchmark_models
from src.evaluation.metrics import regression_metrics, compute_multioutput_aggregates
from src.evaluation.conformal import conformal_quantiles, interval_metrics
from src.visualization.model_plots import plot_model_comparison_bar


def run(quick: bool = False, seed: int = 42, root_dir: Path = None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    root = root_dir or find_project_root()
    out = root / "outputs"
    tables_dir = out / "tables"
    reports_dir = out / "reports"
    figures_dir = out / "figures"
    exp_dir = root / "experiments"
    art_models_dir = root / "artifacts" / "models"
    compat_models_dir = root / "models"

    for d in [tables_dir, reports_dir, figures_dir, exp_dir, art_models_dir, compat_models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"--- TURBINE-X Experiment Run (mode={'QUICK' if quick else 'FULL'}, seed={seed}) ---")
    df = load_raw_yearly_data(root / "data" / "raw")
    dev, cal, test = temporal_split(df)
    print(f"Loaded {len(df)} total rows: Dev (2011-2013)={len(dev)}, Cal (2014)={len(cal)}, Locked Test (2015)={len(test)}")

    registry = get_benchmark_models(quick=quick, seed=seed)
    records = []
    uncertainty_records = []
    tracks = [
        ("A_emissions", FEATURES_A, TARGETS_A),
        ("B_joint", FEATURES_B, TARGETS_B)
    ]

    for track, features, targets in tracks:
        print(f"\nProcessing Track: {track} (Inputs: {len(features)}, Targets: {len(targets)})")
        track_art_dir = art_models_dir / track
        track_art_dir.mkdir(parents=True, exist_ok=True)
        track_compat_dir = compat_models_dir / track
        track_compat_dir.mkdir(parents=True, exist_ok=True)

        for name, estimator in registry.items():
            start = time.perf_counter()
            try:
                fitted = clone(estimator)
                fitted.fit(dev[features], dev[targets].values)
                train_time_sec = time.perf_counter() - start

                # Inference latency on test set
                t_inf = time.perf_counter()
                pred = fitted.predict(test[features])
                latency_ms = (time.perf_counter() - t_inf) / len(test) * 1000.0

                # Conformal calibration on 2014 validation set
                pred_cal = fitted.predict(cal[features])
                q = conformal_quantiles(cal[targets].values, pred_cal, alpha=0.10)
                ci = interval_metrics(test[targets].values, pred, q)

                metrics = regression_metrics(test[targets].values, pred, targets)
                for metric in metrics:
                    j = targets.index(metric["target"])
                    records.append({
                        "track": track,
                        "model": name,
                        "train_years": "2011-2013",
                        "calibration_year": "2014",
                        "test_year": "2015",
                        "n_train": len(dev),
                        "n_calibration": len(cal),
                        "n_test": len(test),
                        "train_time_sec": round(train_time_sec, 3),
                        "latency_ms_per_row": round(latency_ms, 5),
                        "conformal_90_marginal_coverage": ci["marginal_coverage"][j],
                        "conformal_90_width": ci["mean_interval_width"][j],
                        "conformal_90_simultaneous_coverage": ci["simultaneous_coverage"],
                        **metric
                    })

                uncertainty_records.append({
                    "track": track,
                    "model": name,
                    "simultaneous_coverage_90": ci["simultaneous_coverage"],
                    "target_coverages": ci["marginal_coverage"],
                    "interval_widths": ci["mean_interval_width"]
                })

                import joblib
                bundle = {
                    "estimator": fitted,
                    "features": features,
                    "targets": targets,
                    "model_name": name,
                    "track": track,
                    "conformal_alpha": 0.10,
                    "conformal_quantiles": q,
                    "metrics": metrics
                }
                joblib.dump(bundle, track_art_dir / f"{name}.joblib")
                joblib.dump(bundle, track_compat_dir / f"{name}.joblib")
                print(f"  [SUCCESS] {name:22s} | Train: {train_time_sec:.2f}s | Latency: {latency_ms:.4f}ms/row")
            except Exception as exc:
                print(f"  [ERROR]   {name:22s} | {exc}")
                records.append({"track": track, "model": name, "error": str(exc)})

    result = pd.DataFrame(records)
    result.to_csv(tables_dir / "locked_test_metrics.csv", index=False)
    pd.DataFrame(uncertainty_records).to_csv(exp_dir / "uncertainty_metrics.csv", index=False)

    # Split into baseline and advanced model metrics for task registry compliance
    baseline_names = ["Dummy_mean", "LinearRegression", "Ridge", "Lasso", "ElasticNet"]
    df_base = result[result["model"].isin(baseline_names)]
    df_adv = result[~result["model"].isin(baseline_names)]
    df_base.to_csv(exp_dir / "baseline_metrics.csv", index=False)
    df_adv.to_csv(exp_dir / "advanced_model_metrics.csv", index=False)

    # Temporal CV within development years (2011->2012, 2011-2012->2013)
    print("\nRunning within-development chronological cross-validation...")
    cv = []
    cv_folds = [(2011, 2012), (2012, 2013)]
    for track, features, targets in tracks:
        for model_name, estimator in registry.items():
            for train_end, valid_year in cv_folds:
                tr = dev[dev.source_year <= train_end]
                va = dev[dev.source_year == valid_year]
                if tr.empty or va.empty:
                    continue
                try:
                    fitted = clone(estimator)
                    fitted.fit(tr[features], tr[targets].values)
                    pred = fitted.predict(va[features])
                    for metric in regression_metrics(va[targets].values, pred, targets):
                        cv.append({
                            "track": track,
                            "model": model_name,
                            "train_through": train_end,
                            "validation_year": valid_year,
                            **metric
                        })
                except Exception as exc:
                    cv.append({"track": track, "model": model_name, "error": str(exc)})

    df_cv = pd.DataFrame(cv)
    df_cv.to_csv(tables_dir / "development_temporal_cv.csv", index=False)

    # Plot model comparison chart
    if "RMSE" in result.columns:
        valid_res = result.dropna(subset=["RMSE"])
        if not valid_res.empty:
            plot_model_comparison_bar(
                valid_res[valid_res["track"] == "A_emissions"],
                metric_name="RMSE",
                output_path=figures_dir / "model_comparison_rmse.png"
            )

    report = {
        "dataset_rows": len(df),
        "development_rows": len(dev),
        "calibration_rows": len(cal),
        "locked_test_rows": len(test),
        "years": {
            "development": [2011, 2012, 2013],
            "calibration": [2014],
            "locked_test": [2015]
        },
        "warning": "2015 is a locked temporal test. Models were trained strictly on 2011-2013.",
        "records_evaluated": len(result),
        "quick_mode": quick,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(reports_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Summary of Locked 2015 Test Set Performance ---")
    if "target" in result.columns:
        print(result.sort_values(["track", "target", "RMSE"]).to_string(index=False))
    print("\nSaved all metrics to outputs/tables/, experiments/, and fitted models to artifacts/models/ and models/.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run fast subset of models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run(args.quick, args.seed)
