"""TURBINE-X 2.0 Unsupervised Anomaly Detection Module.
Provides 6 unsupervised algorithms trained exclusively on Development reference data (2011-2013).
Produces continuous anomaly scores and binary anomaly flags (-1: anomaly, 1: normal).
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPRegressor


class DeepAutoencoderAnomaly(BaseEstimator, OutlierMixin):
    """Deep Neural Autoencoder Anomaly Detector using Reconstruction Error."""
    def __init__(self, hidden_layer_sizes=(16, 6, 16), max_iter=400, contamination=0.03, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.contamination = contamination
        self.random_state = random_state
        self.autoencoder_ = None
        self.threshold_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        X_scaled = self.scaler_.fit_transform(X)
        self.autoencoder_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            learning_rate_init=0.003,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=15
        )
        self.autoencoder_.fit(X_scaled, X_scaled)
        reconstructed = self.autoencoder_.predict(X_scaled)
        errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        self.threshold_ = float(np.percentile(errors, 100.0 * (1.0 - self.contamination)))
        return self

    def score_samples(self, X):
        """Reconstruction score: negative mean squared error (-MSE). Higher is more normal."""
        X_scaled = self.scaler_.transform(X)
        reconstructed = self.autoencoder_.predict(X_scaled)
        errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        return -errors

    def decision_function(self, X):
        """Scikit-learn standard decision function. Anomaly when < 0, Normal when >= 0."""
        scores = self.score_samples(X)
        return scores - (-self.threshold_)

    def predict(self, X):
        """Predict binary anomaly flags: -1 for anomaly, +1 for normal operational point."""
        return np.where(self.decision_function(X) < 0, -1, 1)


def get_anomaly_detectors(contamination: float = 0.03, random_state: int = 42) -> Dict[str, Any]:
    """Retrieve all 6 anomaly detection models configured for unsupervised reference baseline."""
    return {
        "IsolationForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=150,
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        "LocalOutlierFactor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LocalOutlierFactor(
                n_neighbors=35,
                contamination=contamination,
                novelty=True,
                n_jobs=-1
            ))
        ]),
        "OneClassSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=contamination
            ))
        ]),
        "EllipticEnvelope": Pipeline([
            ("scaler", StandardScaler()),
            ("model", EllipticEnvelope(
                contamination=contamination,
                random_state=random_state,
                support_fraction=0.85
            ))
        ]),
        "DeepAutoencoder": DeepAutoencoderAnomaly(
            hidden_layer_sizes=(16, 6, 16),
            contamination=contamination,
            random_state=random_state
        )
    }

