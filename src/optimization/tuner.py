"""TURBINE-X Hyperparameter Optimization Runner.
Optimizes models using strictly chronological cross-validation within the development set (2011-2013).
"""
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from src.data.splitting import get_temporal_cv_folds
from src.models.model_factory import create_model
from src.optimization.search_spaces import get_optuna_search_space

# Suppress Optuna chatty logs during runs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TemporalOptimizer:
    """Orchestrates hyperparameter optimization using strictly chronological folds."""

    def __init__(
        self,
        dev_df: pd.DataFrame,
        features: List[str],
        targets: List[str],
        n_trials: int = 15,
        random_state: int = 42
    ):
        self.dev_df = dev_df
        self.features = features
        self.targets = targets
        self.n_trials = n_trials
        self.random_state = random_state
        self.cv_folds = get_temporal_cv_folds(dev_df)

    def optimize_model(self, model_name: str) -> Dict[str, Any]:
        """Run an Optuna study optimizing average RMSE across targets."""
        start_time = time.time()

        def objective(trial):
            params = get_optuna_search_space(model_name, trial)
            fold_rmses = []

            for train_idx, val_idx in self.cv_folds:
                tr_data = self.dev_df.iloc[train_idx]
                va_data = self.dev_df.iloc[val_idx]

                X_tr, y_tr = tr_data[self.features], tr_data[self.targets].values
                X_va, y_va = va_data[self.features], va_data[self.targets].values

                # Instantiate model with candidate params
                candidate = create_model(model_name, quick=True, random_state=self.random_state, **params)
                candidate.fit(X_tr, y_tr)
                preds = candidate.predict(X_va)

                # Mean RMSE across targets for this fold
                target_rmses = [
                    mean_squared_error(y_va[:, j], preds[:, j]) ** 0.5
                    for j in range(len(self.targets))
                ]
                fold_rmses.append(np.mean(target_rmses))

            return float(np.mean(fold_rmses))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials)

        elapsed = time.time() - start_time
        trials_df = study.trials_dataframe()

        return {
            "model_name": model_name,
            "best_rmse": float(study.best_value),
            "best_params": study.best_params,
            "elapsed_seconds": elapsed,
            "total_trials": len(study.trials),
            "trials_df": trials_df
        }

