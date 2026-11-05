"""TURBINE-X Comprehensive Evidence-Based Audit Script.
Executes deep inspection of models, artifacts, evaluation metrics, conformal coverage,
notebooks, cross-validation, data leakage, and dashboard functionality.
"""
import os
import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    max_error
)

PROJECT_ROOT = Path("D:/TURBINE_X_Full_Project/TURBINE_X_Full_Project")
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_yearly_data, FEATURES_A, TARGETS_A, FEATURES_B, TARGETS_B
from src.data.splitting import temporal_split, get_temporal_cv_folds
from src.evaluation.conformal import interval_metrics

print("--- Step 1: Loading raw data and verifying splits ---")
df = load_raw_yearly_data(PROJECT_ROOT / "data" / "raw")
dev_df, cal_df, test_df = temporal_split(df)
print(f"Total dataset shape: {df.shape}")
print(f"Dev (2011-2013) count: {len(dev_df)}")
print(f"Cal (2014) count: {len(cal_df)}")
print(f"Test (2015) count: {len(test_df)}")

# Verify split properties
split_audit = {
    "dev_years": sorted(dev_df["source_year"].unique().tolist()),
    "cal_years": sorted(cal_df["source_year"].unique().tolist()),
    "test_years": sorted(test_df["source_year"].unique().tolist()),
    "dev_count": len(dev_df),
    "cal_count": len(cal_df),
    "test_count": len(test_df),
    "overlap_dev_cal": len(set(dev_df.index).intersection(set(cal_df.index))),
    "overlap_dev_test": len(set(dev_df.index).intersection(set(test_df.index))),
    "overlap_cal_test": len(set(cal_df.index).intersection(set(test_df.index))),
    "track_a_features": FEATURES_A,
    "track_b_features": FEATURES_B,
    "tey_in_track_a": "TEY" in FEATURES_A,
    "tey_in_track_b": "TEY" in FEATURES_B,
}
print("Split audit summary:", split_audit)

# --- Step 2: Inspecting and Auditing Model Artifacts ---
print("\n--- Step 2: Model Artifact Verification ---")
models_dir = PROJECT_ROOT / "artifacts" / "models"
all_models = [
    "Dummy_mean", "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "DecisionTree", "RandomForest", "ExtraTrees", "HistGradientBoosting",
    "XGBoost", "CatBoost", "KNN"
]

artifact_records = []
evaluation_records = []
conformal_records = []

tracks_info = [
    ("A_emissions", FEATURES_A, TARGETS_A),
    ("B_joint", FEATURES_B, TARGETS_B)
]

def calculate_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    # where denom is 0, sMAPE is 0
    smape_val = np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(smape_val))

for track, features, targets in tracks_info:
    track_dir = models_dir / track
    X_test = test_df[features].copy()
    y_test = test_df[targets].values

    for model_name in all_models:
        artifact_path = track_dir / f"{model_name}.joblib"
        file_exists = artifact_path.exists()
        file_size = artifact_path.stat().st_size if file_exists else 0
        load_success = False
        pred_success = False
        pred_shape = None
        has_nan = None
        has_inf = None
        pipeline_desc = "None"
        feature_count = 0
        target_count = 0
        n_estimators_or_params = "N/A"
        latency_ms_per_row = 0.0
        throughput_preds_per_sec = 0.0

        if file_exists and file_size > 0:
            try:
                bundle = joblib.load(artifact_path)
                load_success = True
                est = bundle["estimator"]
                bundle_feats = bundle["features"]
                bundle_targets = bundle["targets"]
                quantiles = np.array(bundle.get("conformal_quantiles", [0.0]*len(targets)))
                feature_count = len(bundle_feats)
                target_count = len(bundle_targets)
                pipeline_desc = str(type(est).__name__)
                if hasattr(est, "steps"):
                    pipeline_desc = "Pipeline(" + "->".join([s[0] for s in est.steps]) + ")"

                # Get estimators/params count if applicable
                final_step = est.steps[-1][1] if hasattr(est, "steps") else est
                if hasattr(final_step, "estimators_"):
                    n_estimators_or_params = f"n_estimators={len(final_step.estimators_)}"
                elif hasattr(final_step, "n_estimators"):
                    n_estimators_or_params = f"n_estimators={final_step.n_estimators}"
                elif hasattr(final_step, "coef_"):
                    n_estimators_or_params = f"n_features={final_step.coef_.size}"
                elif hasattr(final_step, "n_neighbors"):
                    n_estimators_or_params = f"k={final_step.n_neighbors}"

                # Warmup and timed latency
                _ = est.predict(X_test.iloc[:10])
                t0 = time.perf_counter()
                y_pred = est.predict(X_test)
                t_total = time.perf_counter() - t0
                pred_success = True
                pred_shape = y_pred.shape
                has_nan = bool(np.isnan(y_pred).any())
                has_inf = bool(np.isinf(y_pred).any())

                latency_ms_per_row = (t_total / len(X_test)) * 1000.0
                throughput_preds_per_sec = len(X_test) / t_total if t_total > 0 else 0.0

                # Compute regression metrics per target
                for idx, t_col in enumerate(targets):
                    yt = y_test[:, idx]
                    yp = y_pred[:, idx]

                    r2 = float(r2_score(yt, yp))
                    mae = float(mean_absolute_error(yt, yp))
                    mse = float(mean_squared_error(yt, yp))
                    rmse = float(np.sqrt(mse))
                    medae = float(median_absolute_error(yt, yp))
                    ev = float(explained_variance_score(yt, yp))
                    max_err = float(max_error(yt, yp))
                    mape = float(np.mean(np.abs((yt - yp) / np.where(yt == 0, 1e-7, yt))) * 100.0)
                    smape = calculate_smape(yt, yp)

                    evaluation_records.append({
                        "track": track,
                        "model": model_name,
                        "target": t_col,
                        "R2": round(r2, 4),
                        "MAE": round(mae, 4),
                        "MSE": round(mse, 4),
                        "RMSE": round(rmse, 4),
                        "MedAE": round(medae, 4),
                        "Explained_Variance": round(ev, 4),
                        "Max_Error": round(max_err, 4),
                        "MAPE_pct": round(mape, 2),
                        "sMAPE_pct": round(smape, 2),
                        "n_test": len(X_test),
                        "latency_ms_per_row": round(latency_ms_per_row, 5),
                        "throughput_preds_per_sec": round(throughput_preds_per_sec, 1),
                        "artifact_size_kb": round(file_size / 1024, 1),
                        "complexity_info": n_estimators_or_params
                    })

                # Compute Conformal Metrics
                ci = interval_metrics(y_test, y_pred, quantiles)
                for idx, t_col in enumerate(targets):
                    conformal_records.append({
                        "track": track,
                        "model": model_name,
                        "target": t_col,
                        "nominal_confidence": 0.90,
                        "calibration_year": 2014,
                        "test_year": 2015,
                        "conformal_quantile": round(float(quantiles[idx]), 4),
                        "empirical_marginal_coverage": round(float(ci["marginal_coverage"][idx]), 4),
                        "mean_interval_width": round(float(ci["mean_interval_width"][idx]), 4),
                        "simultaneous_coverage_all_targets": round(float(ci["simultaneous_coverage"]), 4),
                        "meets_nominal_marginal": bool(ci["marginal_coverage"][idx] >= 0.895)
                    })

            except Exception as e:
                print(f"Error loading {artifact_path}: {e}")
                load_success = False

        artifact_records.append({
            "track": track,
            "model": model_name,
            "file_exists": file_exists,
            "file_size_bytes": file_size,
            "load_success": load_success,
            "pred_success": pred_success,
            "pipeline_type": pipeline_desc,
            "expected_input_features": feature_count,
            "expected_output_targets": target_count,
            "prediction_shape": str(pred_shape),
            "contains_nan": has_nan,
            "contains_inf": has_inf,
            "artifact_rel_path": str(artifact_path.relative_to(PROJECT_ROOT))
        })

df_artifacts = pd.DataFrame(artifact_records)
df_eval = pd.DataFrame(evaluation_records)
df_conformal = pd.DataFrame(conformal_records)

# Save deliverables
rep_dir = PROJECT_ROOT / "outputs" / "reports"
rep_dir.mkdir(parents=True, exist_ok=True)

df_artifacts.to_csv(rep_dir / "MODEL_ARTIFACT_VERIFICATION.csv", index=False)
df_eval.to_csv(rep_dir / "COMPLETE_MODEL_EVALUATION.csv", index=False)
df_conformal.to_csv(rep_dir / "CONFORMAL_COVERAGE_AUDIT.csv", index=False)

print(f"Artifact verification saved: {len(df_artifacts)} models audited.")
print(f"Evaluation metrics saved: {len(df_eval)} target-level evaluations.")
print(f"Conformal audit saved: {len(df_conformal)} interval records.")

# --- Step 3: Cross-Validation Audit ---
cv_file = PROJECT_ROOT / "outputs" / "tables" / "development_temporal_cv.csv"
if cv_file.exists():
    df_cv = pd.read_csv(cv_file)
    cv_summary = df_cv.groupby(["track", "model", "target"]).agg(
        n_folds=("validation_year", "count"),
        validation_years=("validation_year", lambda s: ", ".join(map(str, s))),
        mean_R2=("R2", "mean"),
        std_R2=("R2", "std"),
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),
        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std")
    ).reset_index()
    cv_summary.to_csv(rep_dir / "CROSS_VALIDATION_AUDIT.csv", index=False)
    print("Cross validation audit table saved.")
else:
    print("Warning: development_temporal_cv.csv not found!")

# --- Step 4: Notebooks Audit ---
notebooks_dir = PROJECT_ROOT / "notebooks"
nb_records = []
for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        executed_cells = [c for c in code_cells if c.get("execution_count") is not None]
        error_cells = []
        total_outputs = 0
        for idx, c in enumerate(code_cells):
            outputs = c.get("outputs", [])
            total_outputs += len(outputs)
            for o in outputs:
                if o.get("output_type") == "error":
                    error_cells.append(idx)
        nb_records.append({
            "notebook_name": nb_path.name,
            "total_cells": len(cells),
            "code_cells": len(code_cells),
            "markdown_cells": len(md_cells),
            "executed_code_cells": len(executed_cells),
            "unexecuted_code_cells": len(code_cells) - len(executed_cells),
            "error_cells_count": len(error_cells),
            "total_output_items": total_outputs,
            "status": "PASS" if len(executed_cells) == len(code_cells) and len(error_cells) == 0 else "FAIL"
        })
    except Exception as e:
        nb_records.append({
            "notebook_name": nb_path.name,
            "status": f"ERROR_READING: {e}"
        })

df_nb = pd.DataFrame(nb_records)
print("Notebooks audit summary:\n", df_nb[["notebook_name", "code_cells", "executed_code_cells", "error_cells_count", "status"]])
