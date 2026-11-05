# TURBINE-X — COMPLETE MODEL TRAINING & ARTIFACT AUDIT

## 1. Executive Summary & Verification Matrix
Independent verification of all 12 required machine learning algorithms across both Prediction Tracks.
A total of **24 distinct model artifacts** were inspected, deserialized with `joblib`, probed for pipeline structure,
and verified via actual inference against the locked 2015 test dataset (N=7,384 observations).

### Training Summary by Track
- **Track A (Operational Emission Prediction)**: 12 of 12 models trained, fitted, serialized, and successfully reloaded. Inputs: 9 features (including TEY). Targets: CO, NOx.
- **Track B (Joint Multi-Target Prediction)**: 12 of 12 models trained, fitted, serialized, and successfully reloaded. Inputs: 8 features (strictly excluding TEY). Targets: CO, NOx, TEY.
- **Total Artifacts Verified**: 24 model bundles in `artifacts/models/` and mirrored in `models/`.
- **Status**: **100% COMPLETE — ZERO PLACEHOLDERS**

| Algorithm | Track A Status | Track A Artifact Size | Track B Status | Track B Artifact Size | Pipeline Structure | Preprocessing Included | Reload & Prediction Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Regressor** (`Dummy_mean`) | FITTED & SAVED | 1.9 KB | FITTED & SAVED | 2.1 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **Linear Regression** (`LinearRegression`) | FITTED & SAVED | 2.7 KB | FITTED & SAVED | 2.8 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **Ridge Regression** (`Ridge`) | FITTED & SAVED | 2.5 KB | FITTED & SAVED | 2.7 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **Lasso Regression** (`Lasso`) | FITTED & SAVED | 2.9 KB | FITTED & SAVED | 3.2 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **ElasticNet** (`ElasticNet`) | FITTED & SAVED | 2.9 KB | FITTED & SAVED | 3.2 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **Decision Tree** (`DecisionTree`) | FITTED & SAVED | 181.7 KB | FITTED & SAVED | 218.8 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **Random Forest** (`RandomForest`) | FITTED & SAVED | 238970.1 KB | FITTED & SAVED | 262646.4 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **Extra Trees** (`ExtraTrees`) | FITTED & SAVED | 367959.0 KB | FITTED & SAVED | 401561.6 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **HistGradientBoosting** (`HistGradientBoosting`) | FITTED & SAVED | 1744.2 KB | FITTED & SAVED | 3016.7 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **XGBoost** (`XGBoost`) | FITTED & SAVED | 1698.4 KB | FITTED & SAVED | 2609.2 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **CatBoost** (`CatBoost`) | FITTED & SAVED | 670.7 KB | FITTED & SAVED | 1005.3 KB | StandardScaler -> MultiOutput | Yes | **PASS** |
| **KNN Regressor** (`KNN`) | FITTED & SAVED | 3819.7 KB | FITTED & SAVED | 3630.4 KB | StandardScaler -> MultiOutput | Yes | **PASS** |

---
## 2. Detailed Algorithmic & Hyperparameter Inspection

### Track A (Operational Emissions)
- **Directory**: `artifacts/models/A_emissions/` (mirrored in `models/A_emissions/`)
- **Input Dimensions**: 9 features -> `['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']`
- **Output Dimensions**: 2 targets -> `['CO', 'NOX']`
- **Training Dataset**: Years 2011–2013 (N = 22,191 observations)
- **Validation / Calibration Dataset**: Year 2014 (N = 7,158 observations)
- **Locked Test Dataset**: Year 2015 (N = 7,384 observations)
- **Random Seed**: 42

#### Dummy Regressor (`Dummy_mean`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\Dummy_mean.joblib`
- **File Size**: 1,931 bytes (1.9 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.807, 20.9225]
- **Key Hyperparameters**: `strategy=mean`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### Linear Regression (`LinearRegression`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\LinearRegression.joblib`
- **File Size**: 2,717 bytes (2.7 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `LinearRegression`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.925, 13.7287]
- **Key Hyperparameters**: `copy_X=True`, `fit_intercept=True`, `positive=False`, `tol=1e-06`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### Ridge Regression (`Ridge`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\Ridge.joblib`
- **File Size**: 2,589 bytes (2.5 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `Ridge`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.904, 13.7622]
- **Key Hyperparameters**: `alpha=1.0`, `copy_X=True`, `fit_intercept=True`, `positive=False`, `random_state=42`, `solver=auto`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### Lasso Regression (`Lasso`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\Lasso.joblib`
- **File Size**: 2,970 bytes (2.9 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.6917, 14.62]
- **Key Hyperparameters**: `alpha=0.01`, `copy_X=True`, `fit_intercept=True`, `max_iter=10000`, `positive=False`, `precompute=False`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### ElasticNet (`ElasticNet`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\ElasticNet.joblib`
- **File Size**: 2,970 bytes (2.9 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.6847, 15.8058]
- **Key Hyperparameters**: `alpha=0.01`, `copy_X=True`, `fit_intercept=True`, `l1_ratio=0.5`, `max_iter=10000`, `positive=False`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### Decision Tree (`DecisionTree`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\DecisionTree.joblib`
- **File Size**: 186,076 bytes (181.7 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `DecisionTreeRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [2.0699, 17.8422]
- **Key Hyperparameters**: `ccp_alpha=0.0`, `criterion=squared_error`, `max_depth=12`, `min_impurity_decrease=0.0`, `min_samples_leaf=5`, `min_samples_split=2`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### Random Forest (`RandomForest`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\RandomForest.joblib`
- **File Size**: 244,705,380 bytes (238970.1 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `RandomForestRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.6262, 15.0959]
- **Key Hyperparameters**: `ccp_alpha=0.0`, `criterion=squared_error`, `min_impurity_decrease=0.0`, `min_samples_leaf=1`, `min_samples_split=2`, `min_weight_fraction_leaf=0.0`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### Extra Trees (`ExtraTrees`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\ExtraTrees.joblib`
- **File Size**: 376,790,018 bytes (367959.0 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `ExtraTreesRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.5458, 15.337]
- **Key Hyperparameters**: `ccp_alpha=0.0`, `criterion=squared_error`, `max_features=1.0`, `min_impurity_decrease=0.0`, `min_samples_leaf=1`, `min_samples_split=2`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### HistGradientBoosting (`HistGradientBoosting`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\HistGradientBoosting.joblib`
- **File Size**: 1,786,086 bytes (1744.2 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.6837, 14.9242]
- **Key Hyperparameters**: `categorical_features=from_dtype`, `early_stopping=auto`, `l2_regularization=1.0`, `learning_rate=0.06`, `loss=squared_error`, `max_bins=255`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### XGBoost (`XGBoost`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\XGBoost.joblib`
- **File Size**: 1,739,126 bytes (1698.4 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.8164, 14.8792]
- **Key Hyperparameters**: `objective=reg:squarederror`, `colsample_bytree=0.8`, `enable_categorical=False`, `learning_rate=0.05`, `max_depth=6`, `missing=nan`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### CatBoost (`CatBoost`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\CatBoost.joblib`
- **File Size**: 686,773 bytes (670.7 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.5726, 14.6778]
- **Key Hyperparameters**: `iterations=300`, `learning_rate=0.06`, `depth=6`, `loss_function=RMSE`, `random_seed=42`, `verbose=0`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

#### KNN Regressor (`KNN`)
- **Saved Artifact Path**: `artifacts\models\A_emissions\KNN.joblib`
- **File Size**: 3,911,344 bytes (3819.7 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `KNeighborsRegressor`)
- **Input Features (9 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
- **Output Targets (2 expected)**: ['CO', 'NOX']
- **Conformal 90% Quantiles**: [1.6381, 15.7499]
- **Key Hyperparameters**: `algorithm=auto`, `leaf_size=30`, `metric=minkowski`, `n_jobs=-1`, `n_neighbors=7`, `p=2`
- **Test Inference Probing**: Output shape `(7384, 2)`, zero NaN, zero Inf, 100% verified.

### Track B (Joint Multi-Target)
- **Directory**: `artifacts/models/B_joint/` (mirrored in `models/B_joint/`)
- **Input Dimensions**: 8 features -> `['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']`
- **Output Dimensions**: 3 targets -> `['CO', 'NOX', 'TEY']`
- **Training Dataset**: Years 2011–2013 (N = 22,191 observations)
- **Validation / Calibration Dataset**: Year 2014 (N = 7,158 observations)
- **Locked Test Dataset**: Year 2015 (N = 7,384 observations)
- **Random Seed**: 42

#### Dummy Regressor (`Dummy_mean`)
- **Saved Artifact Path**: `artifacts\models\B_joint\Dummy_mean.joblib`
- **File Size**: 2,112 bytes (2.1 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.807, 20.9225, 23.8174]
- **Key Hyperparameters**: `strategy=mean`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### Linear Regression (`LinearRegression`)
- **Saved Artifact Path**: `artifacts\models\B_joint\LinearRegression.joblib`
- **File Size**: 2,842 bytes (2.8 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `LinearRegression`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.7023, 18.7016, 4.8542]
- **Key Hyperparameters**: `copy_X=True`, `fit_intercept=True`, `positive=False`, `tol=1e-06`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### Ridge Regression (`Ridge`)
- **Saved Artifact Path**: `artifacts\models\B_joint\Ridge.joblib`
- **File Size**: 2,730 bytes (2.7 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `Ridge`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.7028, 18.619, 4.7849]
- **Key Hyperparameters**: `alpha=1.0`, `copy_X=True`, `fit_intercept=True`, `positive=False`, `random_state=42`, `solver=auto`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### Lasso Regression (`Lasso`)
- **Saved Artifact Path**: `artifacts\models\B_joint\Lasso.joblib`
- **File Size**: 3,271 bytes (3.2 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.9146, 16.5384, 4.6697]
- **Key Hyperparameters**: `alpha=0.01`, `copy_X=True`, `fit_intercept=True`, `max_iter=10000`, `positive=False`, `precompute=False`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### ElasticNet (`ElasticNet`)
- **Saved Artifact Path**: `artifacts\models\B_joint\ElasticNet.joblib`
- **File Size**: 3,271 bytes (3.2 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.7841, 16.5001, 3.0262]
- **Key Hyperparameters**: `alpha=0.01`, `copy_X=True`, `fit_intercept=True`, `l1_ratio=0.5`, `max_iter=10000`, `positive=False`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### Decision Tree (`DecisionTree`)
- **Saved Artifact Path**: `artifacts\models\B_joint\DecisionTree.joblib`
- **File Size**: 224,073 bytes (218.8 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `DecisionTreeRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.9368, 18.0752, 5.542]
- **Key Hyperparameters**: `ccp_alpha=0.0`, `criterion=squared_error`, `max_depth=12`, `min_impurity_decrease=0.0`, `min_samples_leaf=5`, `min_samples_split=2`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### Random Forest (`RandomForest`)
- **Saved Artifact Path**: `artifacts\models\B_joint\RandomForest.joblib`
- **File Size**: 268,949,935 bytes (262646.4 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `RandomForestRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.7379, 15.4741, 2.6819]
- **Key Hyperparameters**: `ccp_alpha=0.0`, `criterion=squared_error`, `min_impurity_decrease=0.0`, `min_samples_leaf=1`, `min_samples_split=2`, `min_weight_fraction_leaf=0.0`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### Extra Trees (`ExtraTrees`)
- **Saved Artifact Path**: `artifacts\models\B_joint\ExtraTrees.joblib`
- **File Size**: 411,199,117 bytes (401561.6 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `ExtraTreesRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.5833, 15.395, 2.2619]
- **Key Hyperparameters**: `ccp_alpha=0.0`, `criterion=squared_error`, `max_features=1.0`, `min_impurity_decrease=0.0`, `min_samples_leaf=1`, `min_samples_split=2`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### HistGradientBoosting (`HistGradientBoosting`)
- **Saved Artifact Path**: `artifacts\models\B_joint\HistGradientBoosting.joblib`
- **File Size**: 3,089,113 bytes (3016.7 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.9433, 15.2613, 2.9172]
- **Key Hyperparameters**: `categorical_features=from_dtype`, `early_stopping=auto`, `l2_regularization=1.0`, `learning_rate=0.06`, `loss=squared_error`, `max_bins=255`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### XGBoost (`XGBoost`)
- **Saved Artifact Path**: `artifacts\models\B_joint\XGBoost.joblib`
- **File Size**: 2,671,839 bytes (2609.2 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.8991, 14.6172, 2.6499]
- **Key Hyperparameters**: `objective=reg:squarederror`, `colsample_bytree=0.8`, `enable_categorical=False`, `learning_rate=0.05`, `max_depth=6`, `missing=nan`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### CatBoost (`CatBoost`)
- **Saved Artifact Path**: `artifacts\models\B_joint\CatBoost.joblib`
- **File Size**: 1,029,442 bytes (1005.3 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `MultiOutputRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.5523, 14.5985, 3.3747]
- **Key Hyperparameters**: `iterations=300`, `learning_rate=0.06`, `depth=6`, `loss_function=RMSE`, `random_seed=42`, `verbose=0`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

#### KNN Regressor (`KNN`)
- **Saved Artifact Path**: `artifacts\models\B_joint\KNN.joblib`
- **File Size**: 3,717,557 bytes (3630.4 KB)
- **Pipeline Architecture**: `Pipeline` (`StandardScaler` -> `KNeighborsRegressor`)
- **Input Features (8 expected)**: ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']
- **Output Targets (3 expected)**: ['CO', 'NOX', 'TEY']
- **Conformal 90% Quantiles**: [1.6698, 15.7654, 3.3376]
- **Key Hyperparameters**: `algorithm=auto`, `leaf_size=30`, `metric=minkowski`, `n_jobs=-1`, `n_neighbors=7`, `p=2`
- **Test Inference Probing**: Output shape `(7384, 3)`, zero NaN, zero Inf, 100% verified.

---
## 3. Production Model Artifact (`models/final_model.joblib`)
An integrated production bundle is packaged at `models/final_model.joblib` alongside its schema metadata `models/model_metadata.json`.
- **Selected Architecture**: `HistGradientBoostingRegressor` wrapped with `StandardScaler`.
- **Conformal Calibrated Alpha**: `0.10` (90% nominal confidence intervals).
- **Target Quantiles**: `CO: 1.6837 mg/m³`, `NOX: 14.9242 mg/m³`.
- **Verification**: Loads in sub-second time, feeds the Streamlit dashboard, verified in automated test suite.