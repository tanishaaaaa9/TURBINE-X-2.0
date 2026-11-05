"""TURBINE-X Model Factory.
Provides centralized instantiation of all supported regression models.
"""
from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline
from src.models.baselines import get_dummy_mean_model, get_dummy_median_model, get_linear_regression_model
from src.models.linear_models import get_ridge_model, get_lasso_model, get_elasticnet_model
from src.models.tree_models import (
    get_decision_tree_model,
    get_random_forest_model,
    get_extra_trees_model,
    get_gradient_boosting_model,
    get_hist_gradient_boosting_model
)
from src.models.boosting_models import get_xgboost_model, get_catboost_model
from src.models.svr_models import get_svr_model, get_knn_model


AVAILABLE_MODELS = [
    "Dummy_mean", "Dummy_median", "LinearRegression",
    "Ridge", "Lasso", "ElasticNet",
    "DecisionTree", "RandomForest", "ExtraTrees",
    "GradientBoosting", "HistGradientBoosting",
    "XGBoost", "CatBoost", "SVR", "KNN"
]


def create_model(model_name: str, quick: bool = False, random_state: int = 42, **kwargs) -> Pipeline:
    """Instantiate a model pipeline by name.
    
    Args:
        model_name: Canonical name of the model.
        quick: If True, uses reduced estimator count for fast prototyping.
        random_state: Seed for reproducibility.
        kwargs: Overrides passed to model constructors.
    """
    n_trees = 60 if quick else 250
    hgb_iters = 80 if quick else 250
    xgb_iters = 80 if quick else 200
    cb_iters = 100 if quick else 300

    if model_name == "Dummy_mean":
        return get_dummy_mean_model()
    elif model_name == "Dummy_median":
        return get_dummy_median_model()
    elif model_name == "LinearRegression":
        return get_linear_regression_model()
    elif model_name == "Ridge":
        return get_ridge_model(random_state=random_state, **kwargs)
    elif model_name == "Lasso":
        return get_lasso_model(random_state=random_state, **kwargs)
    elif model_name == "ElasticNet":
        return get_elasticnet_model(random_state=random_state, **kwargs)
    elif model_name == "DecisionTree":
        return get_decision_tree_model(random_state=random_state, **kwargs)
    elif model_name == "RandomForest":
        return get_random_forest_model(n_estimators=kwargs.get("n_estimators", n_trees), random_state=random_state)
    elif model_name == "ExtraTrees":
        return get_extra_trees_model(n_estimators=kwargs.get("n_estimators", n_trees), random_state=random_state)
    elif model_name == "GradientBoosting":
        return get_gradient_boosting_model(n_estimators=kwargs.get("n_estimators", 80 if quick else 180), random_state=random_state)
    elif model_name == "HistGradientBoosting":
        return get_hist_gradient_boosting_model(max_iter=kwargs.get("max_iter", hgb_iters), random_state=random_state)
    elif model_name == "XGBoost":
        return get_xgboost_model(n_estimators=kwargs.get("n_estimators", xgb_iters), random_state=random_state)
    elif model_name == "CatBoost":
        return get_catboost_model(iterations=kwargs.get("iterations", cb_iters), random_state=random_state)
    elif model_name == "SVR":
        return get_svr_model(kernel=kwargs.get("kernel", "linear" if quick else "rbf"))
    elif model_name == "KNN":
        return get_knn_model(n_neighbors=kwargs.get("n_neighbors", 7))
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {AVAILABLE_MODELS}")

