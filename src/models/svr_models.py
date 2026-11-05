"""TURBINE-X Distance & Kernel Based Models.
Support Vector Regression (SVR) and K-Nearest Neighbors (KNN).
Requires feature standardization.
"""
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from src.data.preprocessing import create_preprocessor
from sklearn.pipeline import Pipeline


def get_svr_model(C: float = 10.0, epsilon: float = 0.1, kernel: str = "rbf") -> Pipeline:
    """Returns Support Vector Regressor pipeline with standardization."""
    if kernel == "linear":
        base_svr = LinearSVR(C=C, epsilon=epsilon, max_iter=5000, random_state=42)
    else:
        base_svr = SVR(C=C, epsilon=epsilon, kernel=kernel)

    return Pipeline([
        ("preprocessor", create_preprocessor("standard")),
        ("model", MultiOutputRegressor(base_svr, n_jobs=-1))
    ])


def get_knn_model(n_neighbors: int = 7, weights: str = "distance") -> Pipeline:
    """Returns K-Nearest Neighbors regressor pipeline with standardization."""
    return Pipeline([
        ("preprocessor", create_preprocessor("standard")),
        ("model", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1))
    ])

