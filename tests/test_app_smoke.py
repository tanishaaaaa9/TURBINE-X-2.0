"""TURBINE-X Dashboard and Application Smoke Tests."""
import pytest
from pathlib import Path
import joblib
import pandas as pd
from src.data.loader import find_project_root, FEATURES_A, FEATURES_B


def test_model_artifacts_load_and_predict():
    root = find_project_root()
    model_files = list((root / "artifacts" / "models").glob("*/*.joblib"))
    if not model_files:
        model_files = list((root / "models").glob("*/*.joblib"))

    assert len(model_files) > 0, "No serialized model artifacts found!"

    # Load first artifact
    bundle = joblib.load(model_files[0])
    assert "estimator" in bundle
    assert "features" in bundle
    assert "targets" in bundle

    features = bundle["features"]
    targets = bundle["targets"]
    estimator = bundle["estimator"]

    # Predict on sample row
    sample_df = pd.DataFrame([{f: 10.0 for f in features}])
    pred = estimator.predict(sample_df)
    assert pred.shape == (1, len(targets))

