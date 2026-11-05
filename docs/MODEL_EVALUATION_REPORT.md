# TURBINE-X 2.0 — Comprehensive Model Evaluation Report

## 1. Multi-Task Evaluation Protocol
All evaluation metrics are calculated on the immutable, locked 2015 evaluation test set ($N=7,384$ observations), with zero exposure to model fitting, transformer tuning, or threshold setting.

## 2. Advanced Regression Leaderboard Highlights
- **Turbine Energy Yield ($TEY$)**: Peak performance achieved by XGBoost ($R^2 = 0.9890$, $\text{MAE} = 1.4070\text{ MW}$) and HistGradientBoosting ($R^2 = 0.9881$, $\text{MAE} = 1.4000\text{ MW}$).
- **Carbon Monoxide ($CO$)**: Peak performance achieved by Extra Trees ($R^2 = 0.5769$, $\text{MAE} = 0.8855\text{ mg/m}^3$) and CatBoost ($R^2 = 0.5487$, $\text{MAE} = 0.9024\text{ mg/m}^3$).
- **Nitrogen Oxides ($NO_x$)**: Captured 73.4% of dynamic fluctuation variance ($\text{Explained Variance} = 0.7342$ via CatBoost) despite an industrial $-8.89\text{ mg/m}^3$ mean baseline drop between development and test data.

## 3. Supervised Classification Leaderboard Highlights
- **CO Risk Classification (3 Tiers)**: Peak accuracy achieved by Deep MLPClassifier (Accuracy = 58.64%, Macro F1 = 0.441, Balanced Accuracy = 0.485).
- **NOx Risk Classification (3 Tiers)**: Peak performance achieved by LogisticRegression (Accuracy = 45.40%, Macro F1 = 0.340) and KNNClassifier (Accuracy = 44.56%, Macro F1 = 0.390).

## 4. Unsupervised Anomaly Detection Summary
- Isolation Forest identifies 5.74% of test observations as thermodynamic outliers at 3% nominal contamination, primarily concentrating during extreme seasonal temperature transitions.

