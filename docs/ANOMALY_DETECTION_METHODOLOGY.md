# TURBINE-X 2.0 — Unsupervised Anomaly Detection Methodology

## 1. Thermodynamic Normality Reference
In modern industrial gas turbines, anomalous operations include air filter clogs, compressor blade fouling, fuel valve jitter, and thermal hot-spots.
TURBINE-X 2.0 establishes an unsupervised reference envelope using exclusively the 2011–2013 Development data ($N=22,191$).

## 2. Algorithmic Portfolio
1. **Isolation Forest**: Multi-dimensional orthogonal splits detecting isolated points in low-density subspaces (3% contamination).
2. **Local Outlier Factor (LOF)**: Compares local density against k-nearest thermodynamic neighbors ($k=35$, novelty mode).
3. **One-Class SVM**: RBF kernel boundary enveloping normal operating points in high-dimensional Hilbert space.
4. **Elliptic Envelope**: Robust Mahalanobis covariance envelope assuming approximate elliptical clustering.
5. **Deep Bottleneck Autoencoder**: Neural network ($16 ightarrow 6 ightarrow 16$) trained to reconstruct normal operating profiles; anomalies flagged via elevated reconstruction MSE.
