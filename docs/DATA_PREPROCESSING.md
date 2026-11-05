# TURBINE-X: Data Preprocessing & Leakage Prevention Protocol

## 1. Zero Data Leakage Architecture
Preventing test set contamination is a primary research requirement:
- **Development Horizon (2011–2013):** All statistical properties (medians, standard deviations, quantile ranges, PCA components) are learned exclusively on the 2011–2013 partition (22,191 observations).
- **Calibration Horizon (2014):** Conformal uncertainty non-conformity scores and empirical residual quantiles are calculated strictly on the 2014 partition (7,158 observations).
- **Evaluation Horizon (2015):** The 2015 locked test set (7,384 observations) is evaluated in a single forward pass without fitting or fine-tuning any component.

---

## 2. Preprocessing Transformations
Preprocessing pipelines are constructed using Scikit-Learn `Pipeline` objects to encapsulate feature transformations:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", estimator)
])
```

### Treatment of Missing Values & Anomaly Handling
- **Missing Values:** The UCI dataset contains zero null values across all 36,733 records.
- **Outlier Policy:** Operational spikes in CO and transient drops in TEY reflect actual plant conditions (combustor lean blowout transients, startup sequences). Truncating or deleting these values would artificially distort real-world emission risks. Therefore, outliers are audited, flagged, and preserved.
- **Scaling Regimes:**
  - Standardized (`StandardScaler`): Used for distance-based and regularized linear models (Ridge, Lasso, ElasticNet, SVR, KNN).
  - Unscaled / Native: Used for decision trees and boosting architectures (Random Forest, Extra Trees, HistGB, XGBoost, CatBoost).

---

## 3. Thermodynamic Domain Feature Engineering
In addition to raw sensor measurements, `src/features/feature_engineering.py` provides 7 Brayton-cycle thermodynamic features:
1. **Compressor Pressure Ratio:**
   $$\text{PR} = \frac{\text{CDP} \times 1000}{\text{AP}}$$
2. **Expansion Temperature Drop:**
   $$\Delta T = \text{TIT} - \text{TAT}$$
3. **Expansion Temperature Ratio:**
   $$\text{TR} = \frac{\text{TIT}}{\text{TAT}}$$
4. **Intake Air Density Proxy:**
   $$\rho_{\text{proxy}} = \frac{\text{AP}}{\text{AT} + 273.15}$$
5. **Air Filter Resistance Ratio:**
   $$\text{FR} = \frac{\text{AFDP} \times 1000}{\text{AP}}$$
6. **Enthalpy Drop Surrogate:**
   $$\Delta H_{\text{proxy}} = (\text{TIT} - \text{TAT}) \times \text{CDP}$$
7. **Ambient Thermal Humidity Interaction:**
   $$I_{\text{AT-AH}} = \text{AT} \times \text{AH}$$

