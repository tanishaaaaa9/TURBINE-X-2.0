# TURBINE-X: Known Limitations & Scope Boundaries

## 1. Physical & Operational Limitations
1. **Aggregated Temporal Granularity:** The dataset consists of hourly aggregated observations without microsecond-level combustor pressure dynamics, acoustic oscillations, or flame ionization sensor signals. High-frequency combustion instabilities (lean blowout, thermoacoustic humming) cannot be resolved.
2. **Missing Sub-Hour Timestamps:** Consecutive rows represent chronological hourly intervals, but precise calendar timestamps and calendar dates are omitted from the original UCI release. Therefore, time-series seasonal decomposition (e.g. ARIMA/Prophet) cannot be modeled without inventing artificial timestamps.
3. **Hardware Aging & Degradation Drift:** The 5-year span incorporates normal plant wear (compressor blade fouling, thermal barrier coating erosion, combustor liner degradation). Models trained on 2011–2013 and evaluated on 2015 experience an intrinsic distribution shift in NOx emissions due to unmodeled hardware maintenance history.

---

## 2. Statistical & Machine Learning Limitations
1. **Correlation vs. Causality:** Feature importances and SHAP values reflect associative empirical couplings in historical operational logs. They must not be interpreted as causal transfer functions or safe autonomous setpoint adjustments.
2. **Finite-Sample Conformal Bounds:** Split-conformal intervals guarantee 90% marginal coverage under operational exchangeability, but interval widths expand during rare operating regimes (such as peak summer temperatures or extreme partial loads).

---

## 3. Regulatory & Safety Disclaimer
TURBINE-X is strictly an academic research and technical benchmarking prototype. It is NOT certified for industrial turbine control, life safety protection, or direct PLC governor integration.

