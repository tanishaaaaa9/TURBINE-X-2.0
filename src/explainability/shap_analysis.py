"""TURBINE-X SHAP Explainability Engine.
Provides TreeExplainer and sampled background KernelExplainer for gas turbine models.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def compute_shap_explanations(
    model,
    X_sample: pd.DataFrame,
    background_size: int = 50,
    random_state: int = 42
) -> Dict[str, Any]:
    """Compute SHAP values for tree models or pipeline models using background sampling."""
    np.random.seed(random_state)
    bg = shap.sample(X_sample, background_size, random_state=random_state) if len(X_sample) > background_size else X_sample

    # Check if model has direct tree explainer support
    underlying = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        underlying = model.named_steps["model"]

    # For multi-output regressors, extract first or each sub-estimator
    explainers = []
    shap_vals_dict = {}

    if hasattr(underlying, "estimators_"):
        for j, sub_est in enumerate(underlying.estimators_):
            try:
                explainer = shap.TreeExplainer(sub_est)
                vals = explainer.shap_values(X_sample)
                shap_vals_dict[f"target_{j}"] = vals
            except Exception:
                try:
                    explainer = shap.Explainer(sub_est.predict, bg)
                    vals = explainer(X_sample).values
                    shap_vals_dict[f"target_{j}"] = vals
                except Exception:
                    pass
    elif hasattr(underlying, "predict"):
        try:
            explainer = shap.TreeExplainer(underlying)
            vals = explainer.shap_values(X_sample)
            shap_vals_dict["default"] = vals
        except Exception:
            explainer = shap.Explainer(underlying.predict, bg)
            vals = explainer(X_sample).values
            shap_vals_dict["default"] = vals

    return {
        "features": list(X_sample.columns),
        "shap_values": shap_vals_dict,
        "sample_size": len(X_sample)
    }

