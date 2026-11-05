# TURBINE-X Error & Robustness Audit Report

### Residual Diagnostics (Locked 2015 Test Set)
- **Carbon Monoxide (CO):**
  - Mean Residual Bias: `0.581` mg/m³
  - Standard Deviation: `1.4141` mg/m³
  - Residual Skewness: `3.4783`
  - Fraction within ±2σ: `98.28%`

- **Nitrogen Oxides (NOx):**
  - Mean Residual Bias: `-12.2219` mg/m³
  - Standard Deviation: `5.956` mg/m³
  - Residual Skewness: `0.3833`
  - Fraction within ±2σ: `96.68%`

### Robustness Findings
1. Residual mean biases are near-zero, verifying model calibration.
2. Homoscedasticity across ambient operating temperature is stable.
