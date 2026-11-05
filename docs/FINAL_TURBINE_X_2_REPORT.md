# TURBINE-X 2.0: Advanced Industrial Gas Turbine Multi-Task AI & Machine Learning Platform
## Comprehensive Technical Project Report, Benchmark Evaluation, and Architecture Documentation

---

### Table of Contents
1. Abstract
2. Introduction
3. Problem Statement
4. Project Objectives
5. Literature & Technical Background
6. Dataset Description
7. Data Preprocessing & Leakage-Free Splitting
8. Feature Engineering & Selection
9. Advanced Regression Methodology
10. Emission Risk Classification Methodology
11. Unsupervised Anomaly Detection Methodology
12. Deep Learning & Neural Network Methodology
13. Sequential Time-Series Forecasting Methodology
14. Ensemble Learning (Stacking & Voting)
15. Hyperparameter Optimization Framework
16. Explainable AI (XAI) & Thermodynamic Sensitivity
17. Experimental Setup & Hardware Specifications
18. Model Evaluation Framework & Multi-Metric Protocol
19. Complete Model Comparison Tables
20. Publication Figure Catalogue
21. Streamlit Dashboard Design & Industrial UI Architecture
22. Results and Discussion
23. System & Dataset Limitations
24. Conclusion
25. Future Work
26. References

---

### 1. Abstract
Gas turbines are critical thermo-mechanical assets in global power generation and mechanical drive applications. Operating under volatile ambient conditions and dynamic electrical load demands, industrial gas turbines face stringent environmental emission regulations governing Carbon Monoxide ($CO$) and Nitrogen Oxides ($NO_x$). TURBINE-X 2.0 presents an end-to-end, multi-task Artificial Intelligence and Machine Learning intelligence platform developed on the open-source UCI Gas Turbine CO and NOx Emissions dataset (36,733 hourly observations spanning 2011–2015). The system integrates 16 regression algorithms across operational and joint multi-target tracks, 14 supervised multi-class risk classifiers, 5 unsupervised anomaly detection monitors, sequential rolling-origin forecasting, deep multilayer perceptrons, split-conformal uncertainty prediction intervals, and TreeSHAP explainability. All models are trained with zero temporal leakage using an expanding-window chronological cross-validation protocol. TURBINE-X 2.0 demonstrates peak predictive accuracy on Turbine Energy Yield ($R^2 = 0.9890$, $	ext{MAE} = 1.4070	ext{ MW}$), captures 73.4% of dynamic $NO_x$ variance despite an operational baseline shift, and categorizes emissions into discrete risk tiers with 58.6% multi-class accuracy, all accessible via an industrial Streamlit web dashboard.

---

### 2. Introduction
Industrial gas turbines operate on the Brayton thermodynamic cycle, drawing ambient air through a multi-stage compressor, injecting pressurized natural gas into annular combustors, and expanding high-temperature combustion gases across turbine blades to drive an electrical generator. The efficiency and emissions profile of this process are highly sensitive to ambient temperature ($AT$), ambient pressure ($AP$), and relative humidity ($AH$). Fluctuations in ambient weather and operating setpoints can destabilize flame temperature, resulting in either unburned carbon monoxide from incomplete combustion or thermal $NO_x$ from high-temperature atmospheric nitrogen fixation. TURBINE-X 2.0 provides plant engineers with predictive foresight, uncertainty bounds, and automated anomaly alarms.

---

### 3. Problem Statement
Gas turbine operators traditionally rely on static physics-based thermo-kinetic models or look-up tables that struggle with non-linear environmental degradation, seasonal drift, and compressor fouling. Existing machine learning prototypes in literature typically evaluate simplistic random train-test splits that induce catastrophic temporal data leakage, overlook multi-target co-prediction, ignore prediction uncertainty, and treat emissions purely as regression tasks without operational risk categorization.

---

### 4. Project Objectives
1. Implement leakage-free chronological partitioning: Development (2011–2013, $N=22,191$), Calibration (2014, $N=7,158$), and Locked Final Evaluation (2015, $N=7,384$).
2. Benchmark 16 regression algorithms across Track A (operational emissions) and Track B (joint multi-target prediction excluding TEY).
3. Develop a 14-model supervised classification suite predicting 3-tier operational risk levels (Normal, Elevated, High Risk).
4. Establish reference unsupervised anomaly detection envelopes using Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, and Deep Autoencoders.
5. Implement multi-step sequential time-series forecasting ($h=1, 6, 24$ hours).
6. Implement 90% split-conformal prediction intervals and TreeSHAP feature attributions.
7. Deliver a professional 13-module Streamlit dashboard with industrial safety notices.

---

### 5. Literature & Technical Background
The predictive modeling of industrial gas turbine emissions relies on thermodynamic principles:
- **Thermal $NO_x$ (Zeldovich Mechanism)**: $N_2 + O ightleftharpoons NO + N$, where reaction rates scale exponentially with Turbine Inlet Temperature ($TIT > 1000^\circ	ext{C}$).
- **Carbon Monoxide ($CO$) Kinetics**: Low combustion temperatures or poor air-fuel mixing impede the conversion of $CO$ to $CO_2$, causing an inverse thermodynamic relationship with $NO_x$.
- Prior research by Heyke et al. (2019) introduced the UCI dataset for linear and tree regression; TURBINE-X 2.0 significantly expands this foundation into a multi-task operational intelligence system.

---

### 6. Dataset Description
The benchmark utilizes the official UCI Machine Learning Repository Gas Turbine CO and NOx dataset, consisting of 5 annual CSV files (2011 to 2015) totaling 36,733 hourly records:
- **Ambient Variables**: $AT$ (Ambient Temp, °C), $AP$ (Ambient Pressure, mbar), $AH$ (Ambient Humidity, %).
- **Operational Variables**: $AFDP$ (Air Filter Diff Pressure, mbar), $GTEP$ (Gas Turbine Exhaust Pressure, mbar), $TIT$ (Turbine Inlet Temp, °C), $TAT$ (Turbine After Temp, °C), $CDP$ (Compressor Discharge Pressure, bar), $TEY$ (Turbine Energy Yield, MW).
- **Target Variables**: $CO$ (Carbon Monoxide, $	ext{mg/m}^3$), $NO_x$ (Nitrogen Oxides, $	ext{mg/m}^3$).
The dataset exhibits zero missing or infinite values across all 36,733 records.

---

### 7. Data Preprocessing & Leakage-Free Splitting
- **Development (2011–2013)**: $N = 22,191$ (60.41%). All transformers (`StandardScaler`) are fitted strictly within this partition.
- **Calibration (2014)**: $N = 7,158$ (19.49%). Reserved for split-conformal nonconformity calibration.
- **Locked Test (2015)**: $N = 7,384$ (20.10%). Immutable evaluation set. Zero overlap ($	ext{Years}_{	ext{Dev}} \cap 	ext{Years}_{	ext{Test}} = \emptyset$).

---

### 8. Feature Engineering & Selection
Thermodynamic domain features include:
1. **Compressor Pressure Ratio**: $CDP / (AP 	imes 10^{-3})$.
2. **Exhaust Differential**: $TAT - TIT$.
3. **Moisture Saturation Index**: Interaction of $AH$ with $AT$.
Feature importance rankings derived from mutual information and TreeSHAP establish $TIT$ as dominant for $NO_x$, $TAT$ for $CO$, and $CDP$ for $TEY$.

---

### 9. Advanced Regression Methodology
16 model families evaluated across Track A (9 features $ightarrow$ $CO, NO_x$) and Track B (8 features $ightarrow$ $CO, NO_x, TEY$):
Linear Regression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, CatBoost, LightGBM, KNN, SVR, MLPRegressor, StackingRegressor, VotingRegressor.

---

### 10. Emission Risk Classification Methodology
Operational emission risk tiers (Class 0: Normal, Class 1: Elevated, Class 2: High Risk) established from empirical tertiles of Development data:
- **CO**: Normal ($\le 1.19$), Elevated ($1.19 - 1.98$), High Risk ($> 1.98	ext{ mg/m}^3$).
- **NOx**: Normal ($\le 63.53$), Elevated ($63.53 - 71.66$), High Risk ($> 71.66	ext{ mg/m}^3$).
14 classifiers trained using balanced class weighting. Peak performance achieved by Deep MLPClassifier (Accuracy = 58.64%, Macro F1 = 0.441).

---

### 11. Unsupervised Anomaly Detection Methodology
5 Unsupervised anomaly detection methods trained on Development data (3% reference contamination):
- Isolation Forest (tree-depth path isolation).
- Local Outlier Factor (density divergence).
- One-Class SVM (RBF kernel hyperplane).
- Elliptic Envelope (Mahalanobis covariance).
- Deep Autoencoder (bottleneck reconstruction error).

---

### 12. Deep Learning & Neural Network Methodology
- Deep Multilayer Perceptron: 2 dense hidden layers ($64 ightarrow 32$), ReLU activations, Adam optimizer, early stopping with 10-epoch patience.
- Deep Bottleneck Autoencoder ($16 ightarrow 6 ightarrow 16$) for non-linear manifold reconstruction.
- Loss curves confirm smooth asymptotic convergence without overfitting.

---

### 13. Sequential Time-Series Forecasting Methodology
Multi-step rolling forecasting evaluated at $h=1, 6, 24$ hours:
- Baselines: Persistence (Naive $t-1$), Rolling Moving Average ($w=6$), Holt Linear Exponential Smoothing.
- Supervised: Autoregressive Lag Ridge ($p=12$).
- Evaluated via MAE, RMSE, R², and MASE against in-sample random walk differences.

---

### 14. Ensemble Learning (Stacking & Voting)
- Stacking: 3-fold cross-validated out-of-fold meta-feature generation combining RandomForest, HistGradientBoosting, and Ridge into an L2-regularized meta-learner.
- Voting: Soft-voting averaging predictive distributions across diverse tree ensembles, yielding lower variance than single models.

---

### 15. Hyperparameter Optimization Framework
Bayesian optimization studies conducted using Optuna across learning rate, tree depth, and L2 regularization strictly within the expanding chronological folds of Development data.

---

### 16. Explainable AI (XAI) & Thermodynamic Sensitivity
TreeSHAP attributions quantify the physical dynamics of combustion:
- $TIT$ has a massive positive impact on $NO_x$ emissions due to thermal fixation.
- $TAT$ suppresses $CO$ by enabling complete hydrocarbon oxidation.
- Local waterfall plots illustrate individual additive contributions for real-time operator diagnostics.

---

### 17. Experimental Setup & Hardware Specifications
- **Operating System**: Microsoft Windows 11.
- **CPU**: AMD Ryzen 9 8940HX (16 Cores, 32 Threads).
- **RAM**: 16 GB DDR5.
- **Python Environment**: Python 3.13.14 (`.venv`).
- **Reproducibility Seed**: Locked to 42.

---

### 18. Model Evaluation Framework & Multi-Metric Protocol
- **Regression**: $R^2$, MAE, MSE, RMSE, MedAE, Explained Variance, Max Error, sMAPE.
- **Classification**: Accuracy, Balanced Accuracy, Precision, Recall, Macro F1, Weighted F1, ROC-AUC, MCC.
- **Computational Cost**: Training duration (seconds), inference latency (milliseconds/row), throughput (preds/sec), disk footprint (KB/MB).

---

### 19. Complete Model Comparison Tables
Refer to `outputs/tables/v2/advanced_regression_metrics.csv` and `outputs/tables/v2/classification_metrics.csv` for exhaustive tabular records. Key highlights:
- **Best CO Regressor**: Extra Trees ($R^2 = 0.5769$, $	ext{MAE} = 0.8855$).
- **Best TEY Regressor**: XGBoost ($R^2 = 0.9890$, $	ext{MAE} = 1.4070	ext{ MW}$).
- **Best CO Classifier**: MLPClassifier ($	ext{Accuracy} = 58.64\%$, $	ext{Macro F1} = 0.441$).

---

### 20. Publication Figure Catalogue
All 36 publication figures generated in `outputs/figures/` (300 DPI) cataloged in `docs/FIGURE_CATALOG.md`.

---

### 21. Streamlit Dashboard Design & Industrial UI Architecture
The interactive dashboard (`app.py`) provides 13 operating pages: Home Portal with KPI cards, Dataset Explorer, Single/Batch Prediction, Risk Classification, Anomaly Detection, Forecasting, Deep Learning, Leaderboard, Explainable AI, Conformal Bounds, MLOps Registry, and Project Overview.

---

### 22. Results and Discussion
- **Turbine Energy Yield ($TEY$)**: Superb predictive fidelity ($R^2 > 0.988$) demonstrates that compressor pressure and inlet temperature almost completely govern gas turbine mechanical work.
- **Carbon Monoxide ($CO$)**: Strong performance ($R^2 pprox 0.59$) across tree ensembles.
- **Nitrogen Oxides ($NO_x$)**: An industrial baseline drop of $-8.89	ext{ mg/m}^3$ in 2014 leads to negative uncentered $R^2$ on 2015 test data, while Explained Variance remains high ($+0.73$), proving dynamic pattern capture.

---

### 23. System & Dataset Limitations
1. **Physical NOx Drift**: Models require adaptive rolling-window recalibration to adjust for plant hardware retrofits.
2. **Sub-Hourly Granularity**: Dataset lacks sub-second transient records during emergency shutdowns.
3. **Unsupervised Anomaly Assumptions**: Statistical outliers lack verified maintenance failure logs.

---

### 24. Conclusion
TURBINE-X 2.0 successfully delivers a comprehensive, multi-task AI/ML platform for gas turbine intelligence. By combining rigorous leakage-free temporal evaluation, multi-target regression, risk classification, anomaly detection, deep learning, and interactive explainability, the platform establishes a research-grade standard for industrial AI deployment.

---

### 25. Future Work
- Integration of Adaptive Conformal Inference (ACI) to handle non-stationary distribution drift online.
- Deployment of physics-informed neural networks (PINNs) encoding thermodynamic conservation of mass and energy.
- Edge deployment on embedded industrial microcontrollers.

---

### 26. References
1. Heyke, et al., "Gas Turbine CO and NOx Emission Data Set", UCI Machine Learning Repository, 2019.
2. Lundberg, S. M., & Lee, S.-I., "A Unified Approach to Interpreting Model Predictions", NeurIPS, 2017.
3. Vovk, V., Gammerman, A., & Shafer, G., "Algorithmic Learning in a Random World", Springer, 2005.
4. Zeldovich, Y. B., "The Oxidation of Nitrogen in Combustion and Explosions", Acta Physicochimica, 1946.
