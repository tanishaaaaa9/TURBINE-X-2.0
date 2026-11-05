"""TURBINE-X 2.0 Advanced Model Factory for Regression and Classification.
Provides 16 regression algorithms and 14 classification algorithms wrapped in clean Pipelines.
"""
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingRegressor, StackingRegressor,
    VotingClassifier, StackingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb


def create_regression_model_v2(name: str, random_state: int = 42) -> Pipeline:
    """Instantiate one of 16 regression models wrapped in a StandardScaler pipeline."""
    if name == "LinearRegression":
        model = LinearRegression()
    elif name == "Ridge":
        model = Ridge(alpha=1.0, random_state=random_state)
    elif name == "Lasso":
        model = MultiOutputRegressor(Lasso(alpha=0.1, random_state=random_state, max_iter=2000))
    elif name == "ElasticNet":
        model = MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=2000))
    elif name == "DecisionTree":
        model = DecisionTreeRegressor(max_depth=12, random_state=random_state)
    elif name == "RandomForest":
        model = RandomForestRegressor(n_estimators=150, max_depth=14, n_jobs=-1, random_state=random_state)
    elif name == "ExtraTrees":
        model = ExtraTreesRegressor(n_estimators=150, max_depth=14, n_jobs=-1, random_state=random_state)
    elif name == "HistGradientBoosting":
        model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=31, random_state=random_state))
    elif name == "XGBoost":
        model = MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.08, n_jobs=-1, random_state=random_state
        ))
    elif name == "CatBoost":
        model = MultiOutputRegressor(cb.CatBoostRegressor(
            iterations=150, depth=6, learning_rate=0.08, verbose=0, random_seed=random_state
        ))
    elif name == "LightGBM":
        model = MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.08, n_jobs=-1, random_state=random_state, verbose=-1
        ))
    elif name == "KNN":
        model = KNeighborsRegressor(n_neighbors=7, n_jobs=-1)
    elif name == "SVR":
        model = MultiOutputRegressor(SVR(C=10.0, epsilon=0.2, max_iter=2500))
    elif name == "MLPRegressor":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=500, learning_rate_init=0.003, early_stopping=True,
            n_iter_no_change=15, random_state=random_state
        )
    elif name == "StackingRegressor":
        base = [
            ("rf", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)),
            ("hgb", HistGradientBoostingRegressor(max_iter=80, random_state=random_state)),
            ("ridge", Ridge(alpha=1.0, random_state=random_state))
        ]
        model = MultiOutputRegressor(StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), cv=3))
    elif name == "VotingRegressor":
        estimators = [
            ("rf", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)),
            ("et", ExtraTreesRegressor(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)),
            ("hgb", HistGradientBoostingRegressor(max_iter=80, random_state=random_state))
        ]
        model = MultiOutputRegressor(VotingRegressor(estimators=estimators))
    else:
        raise ValueError(f"Unknown regression model: {name}")

    return Pipeline([("preprocessor", StandardScaler()), ("model", model)])


def create_classification_model_v2(name: str, random_state: int = 42) -> Pipeline:
    """Instantiate one of 14 multi-class classification models wrapped in a StandardScaler pipeline."""
    if name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    elif name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(max_depth=10, random_state=random_state, class_weight="balanced")
    elif name == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=120, max_depth=12, n_jobs=-1, random_state=random_state, class_weight="balanced")
    elif name == "ExtraTreesClassifier":
        model = ExtraTreesClassifier(n_estimators=120, max_depth=12, n_jobs=-1, random_state=random_state, class_weight="balanced")
    elif name == "HistGradientBoostingClassifier":
        model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31, random_state=random_state, class_weight="balanced")
    elif name == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    elif name == "XGBClassifier":
        model = xgb.XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, n_jobs=-1, random_state=random_state, eval_metric="mlogloss")
    elif name == "CatBoostClassifier":
        model = cb.CatBoostClassifier(iterations=120, depth=5, learning_rate=0.08, verbose=0, random_seed=random_state)
    elif name == "LGBMClassifier":
        model = lgb.LGBMClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, n_jobs=-1, random_state=random_state, verbose=-1, class_weight="balanced")
    elif name == "KNNClassifier":
        model = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
    elif name == "SVC":
        model = SVC(C=5.0, kernel="rbf", probability=True, random_state=random_state, max_iter=2500, class_weight="balanced")
    elif name == "MLPClassifier":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=500, learning_rate_init=0.003, early_stopping=True,
            n_iter_no_change=15, random_state=random_state
        )
    elif name == "StackingClassifier":
        base = [
            ("rf", RandomForestClassifier(n_estimators=40, max_depth=8, random_state=random_state, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=60, random_state=random_state)),
            ("lr", LogisticRegression(max_iter=500, random_state=random_state))
        ]
        model = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=500), cv=3)
    elif name == "VotingClassifier":
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=40, max_depth=8, random_state=random_state, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=40, max_depth=8, random_state=random_state, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=60, random_state=random_state))
        ]
        model = VotingClassifier(estimators=estimators, voting="soft")
    else:
        raise ValueError(f"Unknown classification model: {name}")

    return Pipeline([("preprocessor", StandardScaler()), ("model", model)])

