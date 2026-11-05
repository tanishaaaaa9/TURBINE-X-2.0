"""TURBINE-X Model Comparison Module.
Aggregates and formats comparative benchmark tables across architectures and prediction tracks.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def rank_models_by_metric(
    metrics_df: pd.DataFrame,
    metric: str = "RMSE",
    ascending: bool = True,
    group_by_target: bool = True
) -> pd.DataFrame:
    """Rank models across targets by a specific metric."""
    df = metrics_df.copy()
    if group_by_target and "target" in df.columns:
        return df.sort_values(["target", metric], ascending=[True, ascending]).reset_index(drop=True)
    return df.sort_values(metric, ascending=ascending).reset_index(drop=True)


def build_track_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compare common targets (CO and NOX) across Track A vs Track B."""
    common_targets = ["CO", "NOX"]
    subset = metrics_df[metrics_df["target"].isin(common_targets)].copy()
    
    pivot = subset.pivot_table(
        index=["model", "target"],
        columns="track",
        values=["RMSE", "MAE", "R2"]
    ).round(4)
    
    return pivot

