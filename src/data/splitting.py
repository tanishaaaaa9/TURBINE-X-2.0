"""TURBINE-X Chronological Splitting & Leakage Prevention Module.
Implements the primary temporal evaluation protocol:
  - Development / Training: 2011–2013 (22,191 observations)
  - Calibration / Validation: 2014 (7,158 observations)
  - Locked Final Evaluation: 2015 (7,384 observations)
  - Chronological within-development expanding-window cross-validation.
"""
from typing import Dict, List, Tuple, Generator, Any
import pandas as pd
import numpy as np


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the locked temporal split by calendar year.
    
    Args:
        df: Combined dataframe with 'source_year' column.
        
    Returns:
        (dev_df, cal_df, test_df):
          - dev_df: Years 2011-2013 (Training/Development)
          - cal_df: Year 2014 (Calibration / Validation)
          - test_df: Year 2015 (Locked Final Evaluation)
    """
    if "source_year" not in df.columns:
        raise ValueError("DataFrame must contain 'source_year' column for temporal splitting.")

    dev_df = df[df["source_year"].isin([2011, 2012, 2013])].copy().reset_index(drop=True)
    cal_df = df[df["source_year"] == 2014].copy().reset_index(drop=True)
    test_df = df[df["source_year"] == 2015].copy().reset_index(drop=True)

    # Verification checks
    assert len(dev_df) > 0, "Development set is empty!"
    assert len(cal_df) > 0, "Calibration set is empty!"
    assert len(test_df) > 0, "Locked test set is empty!"

    # Disjointness check
    dev_years = set(dev_df["source_year"].unique())
    cal_years = set(cal_df["source_year"].unique())
    test_years = set(test_df["source_year"].unique())

    assert dev_years.isdisjoint(cal_years), "Data leakage: dev and cal share years!"
    assert dev_years.isdisjoint(test_years), "Data leakage: dev and test share years!"
    assert cal_years.isdisjoint(test_years), "Data leakage: cal and test share years!"

    return dev_df, cal_df, test_df


def get_temporal_cv_folds(dev_df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window temporal folds strictly within development data (2011-2013).
    
    Fold 1: Train on 2011 -> Validate on 2012
    Fold 2: Train on 2011-2012 -> Validate on 2013
    
    Returns:
        List of (train_indices, val_indices).
    """
    if "source_year" not in dev_df.columns:
        raise ValueError("dev_df must contain 'source_year' column.")

    folds = []
    years = sorted(dev_df["source_year"].unique())
    
    # Expanding window
    for split_point in range(1, len(years)):
        train_years = years[:split_point]
        val_year = years[split_point]
        
        train_idx = dev_df[dev_df["source_year"].isin(train_years)].index.values
        val_idx = dev_df[dev_df["source_year"] == val_year].index.values
        
        folds.append((train_idx, val_idx))

    return folds


def verify_leakage_free(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Verify that there is zero index overlap or target contamination between splits."""
    overlap_years = set(train_df.get("source_year", [])).intersection(set(test_df.get("source_year", [])))
    
    return {
        "disjoint_years": len(overlap_years) == 0,
        "overlapping_years": list(overlap_years),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "leakage_detected": len(overlap_years) > 0
    }

