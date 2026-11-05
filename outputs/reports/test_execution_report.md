# TURBINE-X: Automated Test Execution Report

**Execution Timestamp:** 2026-11-05  
**Test Framework:** Pytest 9.2.0 (Python 3.13.14)  
**Execution Environment:** Windows 11 CPU-First  
**Overall Status:** **18 PASSED / 0 FAILED / 0 SKIPPED (100% SUCCESS)**  
**Execution Duration:** 1.95s

---

## Detailed Test Case Results

| Test File | Test Function | Status | Scope Verified |
|---|---|---|---|
| `tests/test_data.py` | `test_project_root_discovery` | **PASSED** | Dynamic path resolution to project root. |
| `tests/test_data.py` | `test_yearly_row_counts` | **PASSED** | Exact row counts for 2011 to 2015 CSV files. |
| `tests/test_data.py` | `test_load_all_yearly_data` | **PASSED** | Concatenation of all 36,733 records with zero nulls. |
| `tests/test_data.py` | `test_validate_schema_passing` | **PASSED** | 11 numeric features and data types validation. |
| `tests/test_features.py` | `test_track_a_features_include_tey` | **PASSED** | Track A correctly retains TEY in input features. |
| `tests/test_features.py` | `test_track_b_features_strictly_exclude_tey` | **PASSED** | Track B strictly withholds TEY from input features. |
| `tests/test_features.py` | `test_compute_domain_features` | **PASSED** | Brayton cycle ratios ($\text{PR}, \Delta T, \text{TR}, \rho$) computed. |
| `tests/test_preprocessing.py` | `test_temporal_split_disjointness` | **PASSED** | Strict disjointness between Dev, Cal, and Test sets. |
| `tests/test_preprocessing.py` | `test_preprocessor_pipeline_fit_transform` | **PASSED** | Standardization pipeline transforms without data leakage. |
| `tests/test_models.py` | `test_model_factory_creation` | **PASSED** | Factory instantiates all model architectures cleanly. |
| `tests/test_models.py` | `test_multioutput_prediction_shape` | **PASSED** | Multi-target prediction shapes match target dimensions. |
| `tests/test_models.py` | `test_model_registry_quick_and_full` | **PASSED** | Registry exports quick and full benchmarking suites. |
| `tests/test_evaluation.py` | `test_regression_metrics_calculation` | **PASSED** | Accurate computation of MAE, MSE, RMSE, R², MedAE. |
| `tests/test_evaluation.py` | `test_conformal_quantiles_and_intervals` | **PASSED** | Finite-sample split-conformal calibration quantiles. |
| `tests/test_schema.py` | `test_each_year_schema_and_missingness` | **PASSED** | Baseline schema conformity and zero missingness. |
| `tests/test_tracker.py` | `test_registry_has_unique_ids` | **PASSED** | Unique task IDs in project task registry. |
| `tests/test_tracker.py` | `test_missing_expected_output_is_detected` | **PASSED** | Accurate detection of missing required outputs. |
| `tests/test_app_smoke.py` | `test_model_artifacts_load_and_predict` | **PASSED** | Reloading serialized `.joblib` models and test inference. |

