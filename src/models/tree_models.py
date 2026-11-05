"""TURBINE-X Tree & Traditional Ensemble Models.
Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and HistGradientBoosting.
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputRegressor
from src.data.preprocessing import create_preprocessor
from sklearn.pipeline import Pipeline


def get_decision_tree_model(max_depth: int = 12, min_samples_leaf: int = 5, random_state: int = 42) -> Pipeline:
    """Returns single Decision Tree Regressor pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state))
    ])


def get_random_forest_model(n_estimators: int = 200, min_samples_leaf: int = 2, max_features: float = 0.8, n_jobs: int = -1, random_state: int = 42) -> Pipeline:
    """Returns Random Forest multi-output regressor pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state
        ))
    ])


def get_extra_trees_model(n_estimators: int = 200, min_samples_leaf: int = 2, max_features: float = 0.8, n_jobs: int = -1, random_state: int = 42) -> Pipeline:
    """Returns Extremely Randomized Trees multi-output regressor pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state
        ))
    ])


def get_gradient_boosting_model(n_estimators: int = 150, learning_rate: float = 0.08, max_depth: int = 5, random_state: int = 42) -> Pipeline:
    """Returns GradientBoostingRegressor wrapped for multi-output."""
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            ),
            n_jobs=-1
        ))
    ])


def get_hist_gradient_boosting_model(max_iter: int = 250, learning_rate: float = 0.06, l2_regularization: float = 1.0, random_state: int = 42) -> Pipeline:
    """Returns Fast HistGradientBoostingRegressor wrapped for multi-output."""
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                random_state=random_state
            ),
            n_jobs=-1
        ))
    ])

