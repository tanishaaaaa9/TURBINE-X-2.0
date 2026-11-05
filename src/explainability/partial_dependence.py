"""TURBINE-X Partial Dependence Module.
Computes 1D and 2D partial dependence curves for gas turbine models.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence


def compute_feature_partial_dependence(
    fitted_model,
    X: pd.DataFrame,
    features: List[str],
    grid_resolution: int = 30
) -> Dict[str, Any]:
    """Calculate 1D partial dependence grids for specified features."""
    results = {}
    underlying = fitted_model
    if hasattr(fitted_model, "named_steps") and "model" in fitted_model.named_steps:
        underlying = fitted_model.named_steps["model"]

    for feat in features:
        if feat in X.columns:
            try:
                pdp = partial_dependence(
                    underlying,
                    X,
                    features=[feat],
                    grid_resolution=grid_resolution,
                    kind="average"
                )
                results[feat] = {
                    "grid_values": pdp["grid_values"][0].tolist(),
                    "average": pdp["average"][0].tolist()
                }
            except Exception as e:
                results[feat] = {"error": str(e)}

    return results

