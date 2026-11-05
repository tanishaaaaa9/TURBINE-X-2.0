"""TURBINE-X Advanced Boosting Models.
XGBoost and CatBoost multi-output regression pipelines.
"""
from sklearn.multioutput import MultiOutputRegressor
from src.data.preprocessing import create_preprocessor
from sklearn.pipeline import Pipeline
import xgboost as xgb
import catboost as cb


def get_xgboost_model(n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6, subsample: float = 0.8, colsample_bytree: float = 0.8, n_jobs: int = -1, random_state: int = 42) -> Pipeline:
    """Returns XGBoost multi-output regressor pipeline."""
    reg = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=0
    )
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", MultiOutputRegressor(reg, n_jobs=-1))
    ])


def get_catboost_model(iterations: int = 300, learning_rate: float = 0.06, depth: int = 6, random_state: int = 42) -> Pipeline:
    """Returns CatBoost multi-output regressor pipeline."""
    cb_reg = cb.CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False
    )
    return Pipeline([
        ("preprocessor", create_preprocessor("none")),
        ("model", MultiOutputRegressor(cb_reg, n_jobs=-1))
    ])
