"""TURBINE-X Preprocessing & Splitting Tests."""
import pytest
import pandas as pd
import numpy as np
from src.data.splitting import temporal_split, get_temporal_cv_folds, verify_leakage_free
from src.data.preprocessing import create_preprocessor


def test_temporal_split_disjointness():
    # Synthetic multi-year data
    records = []
    for yr in range(2011, 2016):
        for i in range(10):
            records.append({"source_year": yr, "val": i, "target": i * 2})
    df = pd.DataFrame(records)

    dev, cal, test = temporal_split(df)
    assert set(dev["source_year"].unique()) == {2011, 2012, 2013}
    assert set(cal["source_year"].unique()) == {2014}
    assert set(test["source_year"].unique()) == {2015}

    leakage_dev_test = verify_leakage_free(dev, test)
    assert leakage_dev_test["leakage_detected"] is False


def test_preprocessor_pipeline_fit_transform():
    pipe = create_preprocessor("standard")
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_trans = pipe.fit_transform(X)
    assert X_trans.shape == X.shape
    # Standardized mean should be ~0, std ~1
    np.testing.assert_almost_equal(X_trans.mean(axis=0), [0.0, 0.0])

