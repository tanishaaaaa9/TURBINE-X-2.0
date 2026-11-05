"""TURBINE-X Model Registry.
Exposes standard benchmarking registries for Quick and Full execution modes.
"""
from typing import Dict
from sklearn.pipeline import Pipeline
from src.models.model_factory import create_model


def get_benchmark_models(quick: bool = False, seed: int = 42) -> Dict[str, Pipeline]:
    """Retrieve full benchmark suite of models."""
    if quick:
        # Fast subset for rapid validation
        names = ["Dummy_mean", "LinearRegression", "Ridge", "ElasticNet", "HistGradientBoosting", "RandomForest", "XGBoost"]
    else:
        # Comprehensive suite
        names = [
            "Dummy_mean", "LinearRegression", "Ridge", "Lasso", "ElasticNet",
            "DecisionTree", "RandomForest", "ExtraTrees",
            "HistGradientBoosting", "XGBoost", "CatBoost", "KNN"
        ]

    return {name: create_model(name, quick=quick, random_state=seed) for name in names}


# Backward compatibility with existing src/pipeline/run_experiments.py
def model_registry(seed: int = 42, quick: bool = False) -> Dict[str, Pipeline]:
    return get_benchmark_models(quick=quick, seed=seed)

