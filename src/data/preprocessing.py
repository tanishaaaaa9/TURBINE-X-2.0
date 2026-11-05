"""TURBINE-X Preprocessing Pipelines Module.
Constructs scikit-learn preprocessing pipelines that strictly avoid data leakage.
Scalers and transformers are fit ONLY on development training data.
"""
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib


def create_preprocessor(scaler_type: str = "standard") -> Pipeline:
    """Create a leakage-free feature preprocessing pipeline.
    
    Args:
        scaler_type: 'standard', 'robust', 'minmax', or 'none'.
        
    Returns:
        sklearn.pipeline.Pipeline fitted only during pipeline execution.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    
    if scaler_type == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_type == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler_type == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_type == "none":
        pass
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    return Pipeline(steps)


def save_preprocessor(preprocessor: Pipeline, filepath: str) -> None:
    """Serialize fitted preprocessor using joblib."""
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str) -> Pipeline:
    """Load serialized preprocessor."""
    return joblib.load(filepath)

