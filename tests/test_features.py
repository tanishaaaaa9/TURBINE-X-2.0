"""TURBINE-X Feature Engineering Unit Tests."""
import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import compute_domain_features, get_feature_names


def test_track_a_features_include_tey():
    features_a = get_feature_names(track="A", engineered=False)
    assert "TEY" in features_a
    assert "CO" not in features_a
    assert "NOX" not in features_a


def test_track_b_features_strictly_exclude_tey():
    features_b = get_feature_names(track="B", engineered=False)
    assert "TEY" not in features_b
    assert "CO" not in features_b
    assert "NOX" not in features_b


def test_compute_domain_features():
    # Synthetic test data
    dummy_data = {
        "AT": [15.0, 25.0],
        "AP": [1013.25, 1005.0],
        "AH": [60.0, 80.0],
        "AFDP": [4.0, 3.5],
        "GTEP": [25.0, 30.0],
        "TIT": [1080.0, 1100.0],
        "TAT": [550.0, 540.0],
        "TEY": [130.0, 145.0],
        "CDP": [12.0, 14.0],
        "CO": [1.5, 2.0],
        "NOX": [65.0, 70.0]
    }
    df = pd.DataFrame(dummy_data)
    
    # Track A
    df_feat_a = compute_domain_features(df, include_tey=True)
    assert "pressure_ratio" in df_feat_a.columns
    assert "temp_drop" in df_feat_a.columns
    assert "specific_power" in df_feat_a.columns

    # Track B
    df_feat_b = compute_domain_features(df, include_tey=False)
    assert "pressure_ratio" in df_feat_b.columns
    assert "temp_drop" in df_feat_b.columns
    assert "specific_power" not in df_feat_b.columns

