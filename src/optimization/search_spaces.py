"""TURBINE-X Hyperparameter Search Spaces.
Defines parameter distributions for Optuna and RandomizedSearchCV.
"""
from typing import Dict, Any


def get_optuna_search_space(model_name: str, trial) -> Dict[str, Any]:
    """Sample hyperparameters for a given model architecture using Optuna."""
    if model_name in ["Ridge", "ridge"]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)
        }
    elif model_name in ["RandomForest", "random_forest"]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0)
        }
    elif model_name in ["HistGradientBoosting", "hist_gradient_boosting"]:
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)
        }
    elif model_name in ["XGBoost", "xgboost"]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300, step=40),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
        }
    elif model_name in ["CatBoost", "catboost"]:
        return {
            "iterations": trial.suggest_int("iterations", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8)
        }
    else:
        raise ValueError(f"No Optuna search space defined for {model_name}")

