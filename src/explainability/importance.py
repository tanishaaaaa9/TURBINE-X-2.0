"""TURBINE-X Feature Importance & Permutation Importance Module.
Extracts model-native importances, linear coefficients, and calculates permutation importance.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def extract_feature_importance(fitted_pipeline, feature_names: List[str], target_name: Optional[str] = None) -> pd.DataFrame:
    """Extract intrinsic feature importances or linear coefficients from fitted pipeline.
    
    Args:
        fitted_pipeline: Sklearn Pipeline.
        feature_names: List of input feature names.
        target_name: Optional target label.
    """
    model = fitted_pipeline.named_steps["model"]

    # 1. Tree-based multi-output or single-output
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": imps,
            "target": target_name or "All"
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # 2. MultiOutputRegressor wrapping tree models or linear models
    if hasattr(model, "estimators_"):
        rows = []
        for j, est in enumerate(model.estimators_):
            tgt = target_name or f"target_{j}"
            if hasattr(est, "feature_importances_"):
                for f, imp in zip(feature_names, est.feature_importances_):
                    rows.append({"feature": f, "importance": imp, "target": tgt})
            elif hasattr(est, "coef_"):
                for f, c in zip(feature_names, np.abs(est.coef_)):
                    rows.append({"feature": f, "importance": c, "target": tgt})
        if rows:
            return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

    # 3. Direct linear model with coef_
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        if coef.ndim == 1:
            return pd.DataFrame({
                "feature": feature_names,
                "importance": coef,
                "target": target_name or "Target"
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            rows = []
            for j in range(coef.shape[0]):
                for f, c in zip(feature_names, coef[j]):
                    rows.append({"feature": f, "importance": c, "target": f"target_{j}"})
            return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

    return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})


def compute_permutation_importance(
    fitted_pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """Calculate model-agnostic permutation importance on validation data."""
    res = permutation_importance(
        fitted_pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": res.importances_mean,
        "std_importance": res.importances_std
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    return df_imp

