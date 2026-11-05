# TURBINE-X: Complete Jupyter Notebook Research Guide

This document outlines the systematic, research-grade computational workflow across all 19 executed notebooks in `notebooks/`.

| Notebook | Title | Core Objective | Key Outputs & Artifacts Produced |
|---|---|---|---|
| `00_environment_setup.ipynb` | Environment & Platform Verification | Python interpreter, hardware architecture, package imports, seed setup, directory verification. | `outputs/reports/environment_check.json`, `logs/project.log` |
| `01_dataset_inventory_and_audit.ipynb` | Dataset Inventory & Integrity Audit | Cryptographic SHA-256 hashes, schema conformance, float64 verification, duplicate detection. | `outputs/reports/dataset_inventory.csv`, `outputs/reports/schema_audit.json`, `outputs/reports/data_quality_report.json`, `outputs/tables/duplicate_audit.csv` |
| `02_data_understanding.ipynb` | Data Understanding & Thermodynamics | Physical data dictionary, Brayton cycle subsystem mapping, Track A vs Track B problem definition. | `docs/data_dictionary.md`, `docs/problem_definition.md`, `outputs/reports/chronology_audit.json` |
| `03_exploratory_data_analysis.ipynb` | Exploratory Data Analysis & Annual Shifts | Target emission distributions, skewness, collinearity heatmap, annual drift boxplots. | `outputs/figures/target_distributions.png`, `outputs/figures/correlation_heatmap.png`, `outputs/tables/annual_summary.csv`, `outputs/tables/annual_target_summary.csv` |
| `04_data_quality_and_cleaning.ipynb` | Data Quality, Cleaning & Anomaly Profiling | Zero-loss data cleaning, Tukey IQR outlier profiling, master Parquet export. | `data/processed/cleaned_master.parquet`, `outputs/reports/cleaning_log.json` |
| `05_feature_engineering.ipynb` | Thermodynamic Feature Engineering | Brayton cycle pressure ratios, temperature drops, density proxies, and track-isolated manifests. | `data/processed/feature_manifest.json` |
| `06_feature_selection.ipynb` | Feature Selection & Redundancy Diagnostics | Mutual information scoring against emissions (CO, NOx) and redundancy auditing. | `outputs/tables/feature_diagnostics.csv` |
| `07_train_validation_test_split.ipynb` | Chronological Partitioning & Leakage Audit | Freezing 2011–2013 (Dev), 2014 (Cal), and 2015 (Locked Test) with disjoint index verification. | `data/processed/split_manifest.json` |
| `08_baseline_models.ipynb` | Baseline & Regularized Linear Models | Benchmark Dummy mean/median, OLS, Ridge, Lasso, and ElasticNet. | `experiments/baseline_metrics.csv` |
| `09_advanced_ml_models.ipynb` | Advanced Machine Learning & Ensembles | Benchmark Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost. | `experiments/advanced_model_metrics.csv` |
| `10_hyperparameter_optimization.ipynb` | Hyperparameter Optimization (Optuna) | Optuna TPE sampling on chronological CV folds to optimize learning rate and regularization. | `experiments/optuna_trials.csv` |
| `11_automl_experiments.ipynb` | AutoML Experiments (FLAML) | Time-bounded automated model selection and hyperparameter exploration per target. | `experiments/automl_summary.json` |
| `12_ensemble_learning.ipynb` | Ensemble Learning & Stacking | Voting and weighted ensemble benchmarks vs. top single models. | `experiments/ensemble_comparison.csv` |
| `13_model_explainability.ipynb` | Model Explainability & SHAP Analysis | SHAP TreeExplainer beeswarm and summary feature importance plots for CO and NOx. | `outputs/figures/shap_summary_co.png`, `outputs/figures/shap_summary_nox.png` |
| `14_model_evaluation.ipynb` | Model Evaluation & Conformal Uncertainty | Split-conformal calibration on 2014 and single-shot evaluation on 2015 test set. | `experiments/uncertainty_metrics.csv` |
| `15_error_and_robustness_analysis.ipynb` | Error & Robustness Diagnostics | Residual bias, variance stability, normality tests, and homoscedasticity checks. | `outputs/reports/robustness_report.md` |
| `16_final_model_training_and_saving.ipynb` | Final Pipeline Training & Serialization | Serialization of production `final_model.joblib` and json operational metadata. | `models/final_model.joblib`, `models/model_metadata.json` |
| `17_final_results_and_reproducibility.ipynb` | Final Results & Reproducibility Manifest | Research benchmark tables and cryptographic reproducibility audit. | `outputs/reports/final_evaluation.md`, `outputs/reports/reproducibility.json` |
| `18_reproducible_pipeline.ipynb` | Reproducible Pipeline Execution | Verification of end-to-end command-line runner and automated pytest execution. | `outputs/reports/test_results.txt` |

