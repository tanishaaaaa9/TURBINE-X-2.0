# TURBINE-X 2.0 — Advanced Ensemble Learning Methodology

## 1. Out-of-Fold (OOF) Stacking Architecture
To prevent meta-model data leakage, TURBINE-X 2.0 employs out-of-fold cross-validation during meta-feature generation:
- **Base Estimators**: RandomForest, HistGradientBoosting, and Ridge.
- **Meta-Estimator**: Regularized Ridge regression (for regression) and L2-penalized Logistic Regression (for classification).
- **CV Folds**: 3-fold internal cross-validation strictly within Development data.

## 2. Soft-Voting Ensembles
Combines diverse predictive distributions from ExtraTrees, RandomForest, and HistGradientBoosting:
$$\hat{y}_{	ext{Voting}} = rac{1}{M} \sum_{m=1}^M \hat{y}_m$$
Voting reduces individual estimator variance, achieving stable generalizations across test regimes.
