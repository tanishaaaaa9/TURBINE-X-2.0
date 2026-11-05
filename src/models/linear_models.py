"""TURBINE-X Regularized Linear Models.
Ridge (L2), Lasso (L1), and ElasticNet (L1+L2) regression pipelines.
"""
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from src.data.preprocessing import create_preprocessor
from sklearn.pipeline import Pipeline


def get_ridge_model(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """Returns L2-regularized Ridge Regression pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor("standard")),
        ("model", Ridge(alpha=alpha, random_state=random_state))
    ])


def get_lasso_model(alpha: float = 0.01, random_state: int = 42) -> Pipeline:
    """Returns L1-regularized Lasso Regression pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor("standard")),
        ("model", MultiOutputRegressor(Lasso(alpha=alpha, random_state=random_state, max_iter=10000)))
    ])


def get_elasticnet_model(alpha: float = 0.01, l1_ratio: float = 0.5, random_state: int = 42) -> Pipeline:
    """Returns ElasticNet regression pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor("standard")),
        ("model", MultiOutputRegressor(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000)))
    ])

