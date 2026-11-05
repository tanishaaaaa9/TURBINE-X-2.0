# TURBINE-X 2.0 — Time-Series Forecasting Methodology

## 1. Sequence Continuity & Horizon Design
The UCI Gas Turbine dataset represents chronological hourly operating records. TURBINE-X 2.0 evaluates multi-step rolling forecasting:
- **$h = 1$ Hour**: Immediate next-hour operational setpoint tracking.
- **$h = 6$ Hours**: Mid-shift dispatch horizon.
- **$h = 24$ Hours**: Day-ahead operational emissions planning.

## 2. Baselines and Models
- **Persistence (Naive $t-1$)**: Benchmark persistence assuming static equilibrium.
- **Rolling Moving Average ($w=6$)**: Smooths high-frequency sensor noise.
- **Holt Linear Exponential Smoothing**: Accounts for local level and operational trend.
- **Autoregressive Lag Ridge Regression ($p=12$)**: Exploits multi-hour historical correlation.
- **Metrics**: MAE, RMSE, R², and MASE (Mean Absolute Scaled Error).
