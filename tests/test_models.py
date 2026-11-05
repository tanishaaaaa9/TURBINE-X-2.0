"""TURBINE-X Model Registry & Factory Unit Tests."""
import pytest
import numpy as np
from src.models.model_factory import create_model, AVAILABLE_MODELS
from src.models.model_registry import get_benchmark_models


def test_model_factory_creation():
    for name in ["LinearRegression", "Ridge", "ElasticNet", "RandomForest", "HistGradientBoosting", "XGBoost"]:
        model = create_model(name, quick=True, random_state=42)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_multioutput_prediction_shape():
    X = np.random.randn(20, 9)
    y = np.random.randn(20, 2)
    model = create_model("Ridge", quick=True)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5, 2)


def test_model_registry_quick_and_full():
    quick_models = get_benchmark_models(quick=True)
    assert len(quick_models) >= 5
    assert "Dummy_mean" in quick_models
    assert "HistGradientBoosting" in quick_models

