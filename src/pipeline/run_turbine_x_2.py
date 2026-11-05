"""TURBINE-X 2.0 Master Execution Pipeline.
Executes Advanced Regression (16 models), Emission Risk Classification (14 models),
Unsupervised Anomaly Detection (6 models), Time-Series Forecasting, and 36 Publication Figures.
"""
import os
import sys
import subprocess
from pathlib import Path

# Auto-detect and relaunch with the project virtual environment if running under global system python
def _ensure_venv():
    try:
        import catboost  # noqa: F401
        return
    except ImportError:
        pass

    curr = Path(__file__).resolve().parent
    candidates = [
        curr.parents[2] / ".venv" / "Scripts" / "python.exe",
        curr.parents[1] / ".venv" / "Scripts" / "python.exe",
        Path("D:/TURBINE_X_Full_Project/.venv/Scripts/python.exe"),
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[TURBINE-X 2.0 Launcher] Auto-bridging to project virtual environment: {cand}")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [str(cand), str(Path(__file__).resolve())] + sys.argv[1:]
            res = subprocess.run(cmd, env=env, cwd=str(curr.parents[1] if curr.name == "pipeline" else curr))
            sys.exit(res.returncode)

_ensure_venv()

import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("D:/TURBINE_X_Full_Project/TURBINE_X_Full_Project")
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_yearly_data, FEATURES_A, TARGETS_A, FEATURES_B, TARGETS_B
from src.data.splitting import temporal_split
from src.classification.labeling import compute_operational_risk_thresholds, add_classification_targets
from src.models.model_factory_v2 import create_regression_model_v2, create_classification_model_v2
from src.anomaly.detectors import get_anomaly_detectors
from src.forecasting.evaluator import run_forecasting_benchmark
from src.visualization.figure_generator_v2 import generate_all_36_figures

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error,
    explained_variance_score, max_error, accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, confusion_matrix
)


def run_v2(quick: bool = False, seed: int = 42):
    print("=" * 70)
    print("  TURBINE-X 2.0: ADVANCED MULTI-TASK AI/ML SYSTEM EXECUTION")
    print(f"  Mode: {'QUICK' if quick else 'FULL'} | Seed: {seed}")
    print("=" * 70)

    # Output directories
    art_dir = PROJECT_ROOT / "artifacts" / "models" / "v2"
    art_reg_a = art_dir / "regression" / "A_emissions"
    art_reg_b = art_dir / "regression" / "B_joint"
    art_cls_co = art_dir / "classification" / "CO_risk"
    art_cls_nox = art_dir / "classification" / "NOX_risk"
    art_ano = art_dir / "anomaly"
    tables_v2 = PROJECT_ROOT / "outputs" / "tables" / "v2"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    for d in [art_reg_a, art_reg_b, art_cls_co, art_cls_nox, art_ano, tables_v2, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Temporal Splitting
    print("\n[Step 1/6] Loading UCI Gas Turbine dataset and verifying chronological splits...")
    df = load_raw_yearly_data(PROJECT_ROOT / "data" / "raw")
    dev_df, cal_df, test_df = temporal_split(df)
    print(f"Total rows: {len(df)} | Dev (2011-2013)={len(dev_df)} | Cal (2014)={len(cal_df)} | Locked Test (2015)={len(test_df)}")

    # 2. Risk Classification Labeling (Strictly on Dev)
    print("\n[Step 2/6] Computing operational risk categories strictly from Dev empirical percentiles...")
    risk_thresholds = compute_operational_risk_thresholds(dev_df)
    dev_df = add_classification_targets(dev_df, risk_thresholds)
    cal_df = add_classification_targets(cal_df, risk_thresholds)
    test_df = add_classification_targets(test_df, risk_thresholds)

    with open(PROJECT_ROOT / "configs" / "risk_thresholds.json", "w") as f:
        json.dump(risk_thresholds, f, indent=2)
    print(f"Thresholds configured: CO -> {risk_thresholds['CO']}, NOX -> {risk_thresholds['NOX']}")

    # 3. Advanced Supervised Regression (16 Models)
    print("\n[Step 3/6] Benchmarking 16 Advanced Regression Models across Track A and Track B...")
    reg_models_list = [
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "DecisionTree",
        "RandomForest", "ExtraTrees", "HistGradientBoosting", "XGBoost",
        "CatBoost", "LightGBM", "KNN", "SVR", "MLPRegressor",
        "StackingRegressor", "VotingRegressor"
    ]
    if quick:
        reg_models_list = ["LinearRegression", "Ridge", "HistGradientBoosting", "XGBoost", "CatBoost", "LightGBM", "MLPRegressor"]

    reg_records = []
    tracks = [
        ("A_emissions", FEATURES_A, TARGETS_A, art_reg_a),
        ("B_joint", FEATURES_B, TARGETS_B, art_reg_b)
    ]

    last_pred_reg = None
    for track_name, feats, targs, save_path in tracks:
        print(f"\n--- Training Track: {track_name} ({len(feats)} inputs -> {len(targs)} targets) ---")
        X_tr, y_tr = dev_df[feats], dev_df[targs].values
        X_te, y_te = test_df[feats], test_df[targs].values

        for m_name in reg_models_list:
            t0 = time.perf_counter()
            pipe = create_regression_model_v2(m_name, random_state=seed)
            pipe.fit(X_tr, y_tr)
            train_time = time.perf_counter() - t0

            t_inf = time.perf_counter()
            pred_te = pipe.predict(X_te)
            lat_ms = (time.perf_counter() - t_inf) / len(X_te) * 1000.0

            if track_name == "A_emissions" and m_name in ["HistGradientBoosting", "XGBoost"]:
                last_pred_reg = pred_te

            # Metrics per target
            for idx, t_col in enumerate(targs):
                yt, yp = y_te[:, idx], pred_te[:, idx]
                r2 = float(r2_score(yt, yp))
                mae = float(mean_absolute_error(yt, yp))
                mse = float(mean_squared_error(yt, yp))
                rmse = float(np.sqrt(mse))
                medae = float(median_absolute_error(yt, yp))
                ev = float(explained_variance_score(yt, yp))
                max_err = float(max_error(yt, yp))
                denom = np.abs(yt) + np.abs(yp)
                smape = float(np.mean(np.where(denom == 0, 0.0, 200.0 * np.abs(yp - yt) / denom)))

                reg_records.append({
                    "track": track_name,
                    "model": m_name,
                    "target": t_col,
                    "R2": round(r2, 4),
                    "MAE": round(mae, 4),
                    "MSE": round(mse, 4),
                    "RMSE": round(rmse, 4),
                    "MedAE": round(medae, 4),
                    "Explained_Variance": round(ev, 4),
                    "Max_Error": round(max_err, 4),
                    "sMAPE_pct": round(smape, 2),
                    "train_time_sec": round(train_time, 3),
                    "latency_ms_per_row": round(lat_ms, 5),
                    "artifact_size_kb": 0.0
                })

            # Save bundle
            art_file = save_path / f"{m_name}.joblib"
            bundle = {
                "estimator": pipe,
                "features": feats,
                "targets": targs,
                "model_name": m_name,
                "track": track_name,
                "train_time": train_time
            }
            joblib.dump(bundle, art_file)
            f_size = art_file.stat().st_size / 1024.0
            for r in reg_records:
                if r["track"] == track_name and r["model"] == m_name:
                    r["artifact_size_kb"] = round(f_size, 1)

            print(f"  [Reg] {m_name:24s} | Fit: {train_time:5.2f}s | Latency: {lat_ms:6.4f}ms/row | Saved ({f_size:.1f} KB)")

    df_reg = pd.DataFrame(reg_records)
    df_reg.to_csv(tables_v2 / "advanced_regression_metrics.csv", index=False)

    # 4. Supervised Risk Classification (14 Models)
    print("\n[Step 4/6] Benchmarking 14 Supervised Risk Classifiers for CO and NOx...")
    cls_models_list = [
        "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
        "ExtraTreesClassifier", "HistGradientBoostingClassifier", "GradientBoostingClassifier",
        "XGBClassifier", "CatBoostClassifier", "LGBMClassifier", "KNNClassifier",
        "SVC", "MLPClassifier", "StackingClassifier", "VotingClassifier"
    ]
    if quick:
        cls_models_list = ["LogisticRegression", "HistGradientBoostingClassifier", "XGBClassifier", "CatBoostClassifier", "LGBMClassifier", "MLPClassifier"]

    cls_records = []
    cls_targets = [("CO_risk", art_cls_co), ("NOX_risk", art_cls_nox)]
    last_co_pred, last_co_prob = None, None
    last_nox_pred = None

    for target_col, save_path in cls_targets:
        print(f"\n--- Training Classifiers for Target: {target_col} (3 Classes: Normal, Elevated, High Risk) ---")
        X_tr, y_tr = dev_df[FEATURES_A], dev_df[target_col].values
        X_te, y_te = test_df[FEATURES_A], test_df[target_col].values

        for m_name in cls_models_list:
            t0 = time.perf_counter()
            pipe = create_classification_model_v2(m_name, random_state=seed)
            pipe.fit(X_tr, y_tr)
            train_time = time.perf_counter() - t0

            t_inf = time.perf_counter()
            pred_te = pipe.predict(X_te)
            lat_ms = (time.perf_counter() - t_inf) / len(X_te) * 1000.0

            prob_te = None
            if hasattr(pipe, "predict_proba"):
                try:
                    prob_te = pipe.predict_proba(X_te)
                except Exception:
                    pass

            if target_col == "CO_risk" and m_name in ["HistGradientBoostingClassifier", "LGBMClassifier"]:
                last_co_pred = pred_te
                last_co_prob = prob_te
            if target_col == "NOX_risk" and m_name in ["HistGradientBoostingClassifier", "LGBMClassifier"]:
                last_nox_pred = pred_te

            acc = float(accuracy_score(y_te, pred_te))
            bal_acc = float(balanced_accuracy_score(y_te, pred_te))
            prec, rec, f1_macro, _ = precision_recall_fscore_support(y_te, pred_te, average="macro", zero_division=0)
            _, _, f1_weighted, _ = precision_recall_fscore_support(y_te, pred_te, average="weighted", zero_division=0)
            mcc = float(matthews_corrcoef(y_te, pred_te))

            roc_auc_val = 0.0
            if prob_te is not None:
                try:
                    roc_auc_val = float(roc_auc_score(y_te, prob_te, multi_class="ovr", average="macro"))
                except Exception:
                    roc_auc_val = 0.0

            cls_records.append({
                "target": target_col,
                "model": m_name,
                "Accuracy": round(acc, 4),
                "Balanced_Accuracy": round(bal_acc, 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "Macro_F1": round(float(f1_macro), 4),
                "Weighted_F1": round(float(f1_weighted), 4),
                "ROC_AUC": round(roc_auc_val, 4),
                "MCC": round(mcc, 4),
                "train_time_sec": round(train_time, 3),
                "latency_ms_per_row": round(lat_ms, 5)
            })

            art_file = save_path / f"{m_name}.joblib"
            bundle = {
                "estimator": pipe,
                "features": FEATURES_A,
                "target": target_col,
                "classes": [0, 1, 2],
                "class_names": ["Normal", "Elevated", "High Risk"],
                "train_time": train_time
            }
            joblib.dump(bundle, art_file)
            print(f"  [Cls] {m_name:28s} | Acc: {acc*100:5.2f}% | Macro F1: {f1_macro:5.3f} | Fit: {train_time:5.2f}s")

    df_cls = pd.DataFrame(cls_records)
    df_cls.to_csv(tables_v2 / "classification_metrics.csv", index=False)

    # 5. Unsupervised Anomaly Detection
    print("\n[Step 5/6] Training 5 Unsupervised Anomaly Detectors on Dev Reference...")
    anomaly_models = get_anomaly_detectors(contamination=0.03, random_state=seed)
    X_ano_tr = dev_df[FEATURES_A]
    X_ano_te = test_df[FEATURES_A]

    anomaly_records = []
    anomaly_scores_dict = {}

    for a_name, detector in anomaly_models.items():
        t0 = time.perf_counter()
        detector.fit(X_ano_tr)
        t_fit = time.perf_counter() - t0

        if hasattr(detector, "score_samples"):
            scores = detector.score_samples(X_ano_te)
        elif hasattr(detector, "decision_function"):
            scores = detector.decision_function(X_ano_te)
        else:
            scores = np.zeros(len(X_ano_te))

        flags_tr = detector.predict(X_ano_tr)
        flags = detector.predict(X_ano_te)
        dev_anomaly_rate = float(np.mean(flags_tr == -1))
        anomaly_rate = float(np.mean(flags == -1))
        anomaly_scores_dict[a_name] = scores

        anomaly_records.append({
            "model": a_name,
            "dev_anomaly_rate_pct": round(dev_anomaly_rate * 100, 2),
            "anomaly_rate_pct": round(anomaly_rate * 100, 2),
            "mean_anomaly_score": round(float(np.mean(scores)), 4),
            "score_std": round(float(np.std(scores)), 4),
            "train_time_sec": round(t_fit, 3),
            "dev_sample_count": len(X_ano_tr),
            "test_sample_count": len(X_ano_te),
            "drift_impact": "High Distribution Shift" if anomaly_rate > 0.50 else ("Moderate Shift" if anomaly_rate > 0.10 else "Low Shift")
        })

        joblib.dump(detector, art_ano / f"{a_name}.joblib")
        print(f"  [Anomaly] {a_name:22s} | Dev Rate: {dev_anomaly_rate*100:4.1f}% | Test Rate: {anomaly_rate*100:5.2f}% | Fit: {t_fit:5.2f}s")

    df_ano = pd.DataFrame(anomaly_records)
    df_ano.to_csv(tables_v2 / "anomaly_metrics.csv", index=False)

    # 6. Sequential Forecasting Benchmark
    print("\n[Step 6/6] Executing Time-Series Forecasting Benchmark (h=1, 6, 24 hours)...")
    forecast_results = []
    for tgt in ["CO", "NOX", "TEY"]:
        tr_s = dev_df[tgt].values
        te_s = test_df[tgt].values
        for h in [1, 6, 24]:
            res = run_forecasting_benchmark(tr_s, te_s, target_name=tgt, horizon=h)
            forecast_results.extend(res)

    df_fc = pd.DataFrame(forecast_results)
    df_fc.to_csv(tables_v2 / "forecasting_metrics.csv", index=False)
    print(f"Forecasting benchmark completed: {len(df_fc)} horizon evaluations recorded.")

    # Master Leaderboard Consolidation
    df_reg["task"] = "Regression"
    df_cls["task"] = "Classification"
    df_leaderboard = pd.concat([
        df_reg[["task", "track", "model", "target", "R2", "MAE", "RMSE", "latency_ms_per_row"]],
        df_cls[["task", "target", "model", "Accuracy", "Macro_F1", "ROC_AUC", "latency_ms_per_row"]].rename(columns={"Accuracy": "R2_or_Accuracy", "Macro_F1": "Macro_F1_or_EV"})
    ], ignore_index=True)
    df_leaderboard.to_csv(reports_dir / "TURBINE_X_2_MODEL_LEADERBOARD.csv", index=False)

    # Generate 36 Publication Figures
    print("\nGenerating all 36 Publication-Ready Figures in outputs/figures/...")
    if last_pred_reg is None:
        last_pred_reg = np.zeros((len(test_df), 2))
    if last_co_pred is None:
        last_co_pred = test_df["CO_risk"].values
        last_co_prob = np.eye(3)[last_co_pred]
    if last_nox_pred is None:
        last_nox_pred = test_df["NOX_risk"].values

    generate_all_36_figures(
        df=df,
        reg_df=df_reg,
        cls_df=df_cls,
        y_test_reg=test_df[TARGETS_A].values,
        y_pred_reg=last_pred_reg,
        y_test_co_cls=test_df["CO_risk"].values,
        y_pred_co_cls=last_co_pred,
        y_prob_co_cls=last_co_prob,
        y_test_nox_cls=test_df["NOX_risk"].values,
        y_pred_nox_cls=last_nox_pred,
        anomaly_scores=anomaly_scores_dict,
        forecast_results=forecast_results
    )

    print("\n" + "=" * 70)
    print("  TURBINE-X 2.0 ADVANCED EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TURBINE-X 2.0 Runner")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_v2(quick=args.quick, seed=args.seed)

