"""TURBINE-X Baseline Models.
Provides standard dummy/mean benchmarks and ordinary least squares regression.
"""
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from src.data.preprocessing import create_preprocessor
from sklearn.pipeline import Pipeline


def get_dummy_mean_model() -> Pipeline:
    """Returns a baseline model predicting target mean."""
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", MultiOutputRegressor(DummyRegressor(strategy="mean")))
    ])


def get_dummy_median_model() -> Pipeline:
    """Returns a baseline model predicting target median."""
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", MultiOutputRegressor(DummyRegressor(strategy="median")))
    ])


def get_linear_regression_model() -> Pipeline:
    """Returns Standardized Ordinary Least Squares Linear Regression."""
    return Pipeline([
        ("preprocessor", create_preprocessor("standard")),
        ("model", LinearRegression())
    ])

