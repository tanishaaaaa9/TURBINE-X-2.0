"""TURBINE-X Feature Engineering Module.
Computes thermodynamic domain features, physical ratios, and interactions for Gas Turbine modeling.
Supports both Track A (with TEY in inputs) and Track B (strictly without TEY).
"""
from typing import List, Tuple
import numpy as np
import pandas as pd


def compute_domain_features(df: pd.DataFrame, include_tey: bool = True) -> pd.DataFrame:
    """Compute thermodynamic gas turbine domain features.
    
    Domain features derived from Brayton cycle principles:
      - pressure_ratio: Compressor Discharge Pressure / Ambient Pressure
      - temp_drop: Temperature drop across turbine expansion (TIT - TAT)
      - temp_ratio: TIT / TAT
      - air_density_proxy: AP / (AT + 273.15) (ambient air density proxy from ideal gas law)
      - filter_ratio: AFDP / AP (intake air filter resistance)
      - work_enthalpy_proxy: (TIT - TAT) * CDP (surrogate for turbine enthalpy drop)
      - heat_rate_proxy: TEY / (TIT * GTEP) if include_tey is True
    
    Args:
        df: Input DataFrame containing baseline gas turbine measurements.
        include_tey: Whether TEY can be used for feature engineering (Track A = True, Track B = False).
        
    Returns:
        DataFrame with original and new engineered features.
    """
    out = df.copy()

    # 1. Compressor Pressure Ratio (CDP is in bar, AP is in mbar: 1 bar = 1000 mbar)
    # CDP * 1000 / AP gives non-dimensional pressure ratio
    out["pressure_ratio"] = (out["CDP"] * 1000.0) / (out["AP"] + 1e-6)

    # 2. Turbine Temperature Drop (°C) across turbine stages
    out["temp_drop"] = out["TIT"] - out["TAT"]

    # 3. Expansion Temperature Ratio
    out["temp_ratio"] = out["TIT"] / (out["TAT"] + 1e-6)

    # 4. Ambient Air Density Proxy (P / T) in Kelvin
    out["air_density_proxy"] = out["AP"] / (out["AT"] + 273.15)

    # 5. Air Filter Differential Ratio
    out["filter_ratio"] = (out["AFDP"] * 1000.0) / (out["AP"] + 1e-6)

    # 6. Turbine Specific Enthalpy Extraction Proxy
    out["enthalpy_proxy"] = (out["TIT"] - out["TAT"]) * out["CDP"]

    # 7. Non-linear interaction between Ambient Temperature and Humidity
    out["ambient_temp_humidity"] = out["AT"] * out["AH"]

    # 8. Track A specific: energy yield ratios (NOT permitted in Track B)
    if include_tey and "TEY" in out.columns:
        out["specific_power"] = out["TEY"] / (out["CDP"] + 1e-6)
        out["thermal_load_ratio"] = out["TEY"] / (out["TIT"] + 1e-6)

    return out


def get_feature_names(track: str = "A", engineered: bool = False) -> List[str]:
    """Return feature column list for Track A or Track B, baseline or engineered."""
    base_a = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
    base_b = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"]

    if not engineered:
        return list(base_a) if track.upper() == "A" else list(base_b)

    eng_shared = [
        "pressure_ratio", "temp_drop", "temp_ratio",
        "air_density_proxy", "filter_ratio", "enthalpy_proxy",
        "ambient_temp_humidity"
    ]

    if track.upper() == "A":
        return base_a + eng_shared + ["specific_power", "thermal_load_ratio"]
    else:
        return base_b + eng_shared

