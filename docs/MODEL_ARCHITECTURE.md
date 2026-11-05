# TURBINE-X: Model Architecture & Algorithmic Foundations

## 1. Algorithmic Families Benchmark Matrix

| Family | Model Key | Mathematical Formulation / Implementation | Optimization & Preprocessing | Strengths & Computational Characteristics |
|---|---|---|---|---|
| **Baselines** | `Dummy_mean` | $\hat{y} = \frac{1}{N}\sum y_i$ | Unscaled | Lower-bound baseline demonstrating naive mean performance ($R^2 \approx 0.0$). |
| **Baselines** | `LinearRegression` | $\min_w \|y - Xw\|_2^2$ | Standardized | Unregularized OLS solution; baseline for linear explanatory power. |
| **Linear** | `Ridge` | $\min_w \|y - Xw\|_2^2 + \alpha \|w\|_2^2$ | Standardized ($\alpha=1.0$) | Mitigates collinearity between thermodynamic variables ($CDP, GTEP, TIT$). |
| **Linear** | `Lasso` | $\min_w \|y - Xw\|_2^2 + \alpha \|w\|_1$ | Standardized ($\alpha=0.01$) | Enforces sparsity; performs automated feature selection. |
| **Linear** | `ElasticNet` | $\min_w \|y - Xw\|_2^2 + \alpha \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2} \|w\|_2^2$ | Standardized ($\alpha=0.01, \rho=0.5$) | Combines L1 feature grouping with L2 shrinkage. |
| **Trees** | `DecisionTree` | CART Regressor with variance reduction splitting | Unscaled ($\text{max\_depth}=12$) | Captures step-wise non-linear thresholding; prone to high variance. |
| **Ensembles** | `RandomForest` | Bagging ensemble of randomized decision trees | Unscaled ($\text{max\_features}=0.8$) | Suppresses variance through bootstrap aggregation; excellent out-of-the-box generalization. |
| **Ensembles** | `ExtraTrees` | Extremely Randomized Trees with random split cutpoints | Unscaled | Reduces variance further by drawing cutpoints uniformly at random. |
| **Boosting** | `HistGradientBoosting` | Histogram-binned gradient boosted decision trees | Unscaled (Integer binning) | $O(N)$ histogram binning; native multi-threading; highly efficient on 36K rows. |
| **Boosting** | `XGBoost` | Second-order Taylor gradient boosted trees | Unscaled ($\text{subsample}=0.8$) | Regularized loss objective with exact and histogram split finding. |
| **Boosting** | `CatBoost` | Symmetric oblivious decision trees | Unscaled (Permutation-driven) | Mitigates prediction shift and target leakage; outstanding convergence on tabular data. |
| **Distance** | `KNN` | $k$-Nearest Neighbors regression | Standardized ($k=7$) | Non-parametric local neighborhood estimation. |

---

## 2. Multi-Output Strategy
- **Single-Output Independent:** Models wrapped via `MultiOutputRegressor(estimator, n_jobs=-1)`. Fits independent estimators per target channel while parallelizing across CPU cores.
- **Native Multi-Output:** Random Forest and Extra Trees natively construct joint trees minimizing multi-target variance $\sum_{j=1}^M \text{Var}(y^{(j)})$.

