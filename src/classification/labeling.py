"""TURBINE-X 2.0 Operational Emission Risk Classification & Labeling Module.
Defines data-driven operational risk thresholds strictly from development data (2011-2013).
Zero test leakage: Test observations are only categorized using pre-computed Dev thresholds.
"""
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd


def compute_operational_risk_thresholds(dev_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute empirical operational risk category boundaries strictly on Development data (2011-2013).
    
    Category definitions:
      0 = Normal Operating State (Low / Baseline emissions)
      1 = Elevated Emissions (Transient / Variable operating regime)
      2 = High Risk (Combustion instability / High emission regime)
      
    Derived from empirical 33rd and 67th tertiles of the 2011-2013 turbine records:
    """
    co_p33 = float(np.percentile(dev_df["CO"], 33.33))
    co_p67 = float(np.percentile(dev_df["CO"], 66.67))
    
    nox_p33 = float(np.percentile(dev_df["NOX"], 33.33))
    nox_p67 = float(np.percentile(dev_df["NOX"], 66.67))
    
    # Rounded to nearest engineering decimals for reproducibility
    thresholds = {
        "CO": {
            "tier_1_normal_max": round(co_p33, 2),     # ~1.40 mg/m3
            "tier_2_elevated_max": round(co_p67, 2),   # ~2.40 mg/m3
            "labels": ["Normal", "Elevated", "High Risk"]
        },
        "NOX": {
            "tier_1_normal_max": round(nox_p33, 2),    # ~63.0 mg/m3
            "tier_2_elevated_max": round(nox_p67, 2),  # ~73.0 mg/m3
            "labels": ["Normal", "Elevated", "High Risk"]
        }
    }
    return thresholds


def assign_risk_classes(series: pd.Series, t1_max: float, t2_max: float) -> np.ndarray:
    """Assign multi-class risk labels (0, 1, 2) based on pre-computed thresholds."""
    conditions = [
        series <= t1_max,
        (series > t1_max) & (series <= t2_max),
        series > t2_max
    ]
    choices = [0, 1, 2]
    return np.select(conditions, choices, default=1).astype(int)


def add_classification_targets(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
    """Add 'CO_risk' and 'NOX_risk' categorical columns to a dataframe."""
    out = df.copy()
    co_th = thresholds["CO"]
    nox_th = thresholds["NOX"]
    
    out["CO_risk"] = assign_risk_classes(out["CO"], co_th["tier_1_normal_max"], co_th["tier_2_elevated_max"])
    out["NOX_risk"] = assign_risk_classes(out["NOX"], nox_th["tier_1_normal_max"], nox_th["tier_2_elevated_max"])
    return out

