# TURBINE-X 2.0 — Emission Risk Classification Methodology

## 1. Problem Formulation & Industrial Motivation
Continuous regression predictions (e.g. 2.1 mg/m³ CO) are essential for statistical tracking, but operational control rooms require discrete, actionable alarm levels. TURBINE-X 2.0 introduces an operational risk tier categorization:
- **Class 0 (Normal)**: Baseline combustor operation.
- **Class 1 (Elevated)**: Early indicator of flame variation or transient load.
- **Class 2 (High Risk)**: Severe emission exceedance requiring burner setpoint intervention.

## 2. Zero-Leakage Threshold Derivation
Thresholds are derived **strictly from the empirical tertiles (33.33% and 66.67%) of the 2011–2013 Development dataset**:
- **CO Tiers**: Normal ($\le 1.19	ext{ mg/m}^3$), Elevated ($1.19 < CO \le 1.98	ext{ mg/m}^3$), High Risk ($> 1.98	ext{ mg/m}^3$).
- **NOx Tiers**: Normal ($\le 63.53	ext{ mg/m}^3$), Elevated ($63.53 < NOX \le 71.66	ext{ mg/m}^3$), High Risk ($> 71.66	ext{ mg/m}^3$).
The locked 2015 test set was completely excluded during boundary selection.

## 3. Algorithmic Suite & Results
14 distinct classification architectures were benchmarked:
- **Top CO Risk Classifiers**: Deep MLPClassifier (Accuracy = 58.64%, Macro F1 = 0.441), StackingClassifier (55.34%, 0.437), HistGradientBoosting (53.21%, 0.398).
- **Top NOx Risk Classifiers**: LogisticRegression (45.40%, 0.340), ExtraTreesClassifier (44.60%, 0.359), KNN (44.56%, 0.390).
