# TURBINE-X: Research Methodology & Evaluation Standards

## 1. Chronological Splitting vs. Random K-Fold Splitting
In continuous operational processes such as power plant turbines, random cross-validation randomly shuffles consecutive hourly records between train and test sets. Because weather patterns and plant degradation operate on continuous time scales, random splitting causes severe optimistic bias and temporal data leakage.

TURBINE-X enforces a strict chronological multi-year protocol:
1. **Development Years (2011–2013, $N = 22,191$):** Used for exploratory data analysis, feature selection, pipeline architecture, and hyperparameter optimization.
2. **Within-Development Temporal Folds:** Expanding-window chronological validation:
   - Fold 1: Train on 2011 ($N=7,411$) $\to$ Validate on 2012 ($N=7,628$).
   - Fold 2: Train on 2011–2012 ($N=15,039$) $\to$ Validate on 2013 ($N=7,152$).
3. **Calibration Year (2014, $N = 7,158$):** Calibrates split-conformal uncertainty intervals and tunes probability thresholds.
4. **Locked Evaluation Year (2015, $N = 7,384$):** Evaluated strictly once after model architectures and hyperparameters are locked.

---

## 2. Split-Conformal Uncertainty Calibration
Standard machine learning regression outputs point predictions $\hat{y}$ without finite-sample guarantees. TURBINE-X implements distribution-free **Split-Conformal Prediction**:
1. On Calibration data $(X_{\text{cal}}, Y_{\text{cal}})$ from 2014, compute non-conformity absolute residuals:
   $$R_i^{(j)} = |y_{i,\text{cal}}^{(j)} - \hat{y}_{i,\text{cal}}^{(j)}|$$
2. For nominal error rate $\alpha = 0.10$ (90% coverage), calculate the empirical conformal quantile:
   $$q_{1-\alpha}^{(j)} = \text{Quantile}\left(R^{(j)}, \frac{\lceil(n_{\text{cal}} + 1)(1 - \alpha)\rceil}{n_{\text{cal}}}\right)$$
3. Construct prediction intervals on unseen 2015 test data:
   $$C(x_{\text{test}})^{(j)} = [\hat{y}_{\text{test}}^{(j)} - q_{1-\alpha}^{(j)}, \; \hat{y}_{\text{test}}^{(j)} + q_{1-\alpha}^{(j)}]$$
This guarantees that for any test sample $(x_{n+1}, y_{n+1})$ drawn from the exchangeable operational distribution:
$$P(y_{n+1}^{(j)} \in C(x_{n+1})^{(j)}) \ge 1 - \alpha$$

