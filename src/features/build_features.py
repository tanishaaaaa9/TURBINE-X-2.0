from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def make_linear_pipeline(estimator):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", estimator)])

def make_tree_pipeline(estimator):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", estimator)])
