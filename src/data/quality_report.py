"""TURBINE-X Data Quality & Profiling Report Generator.
Generates comprehensive distribution statistics, annual trend analysis, and markdown reports.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import pandas as pd
from src.data.loader import EXPECTED_COLUMNS, FEATURES_A, TARGETS_A


def generate_data_quality_report(df: pd.DataFrame, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generate extensive statistical profile and data quality metrics for the gas turbine dataset."""
    feature_cols = [c for c in df.columns if c != "source_year"]
    
    # Statistical summary
    desc = df[feature_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
    
    # Skewness and Kurtosis
    skewness = df[feature_cols].skew().to_dict()
    kurtosis = df[feature_cols].kurtosis().to_dict()

    # Correlation matrix
    corr_matrix = df[feature_cols].corr().round(4).to_dict()

    # Per-year statistics
    yearly_stats = {}
    if "source_year" in df.columns:
        for yr, group in df.groupby("source_year"):
            yearly_stats[int(yr)] = {
                "row_count": len(group),
                "means": group[feature_cols].mean().round(3).to_dict(),
                "stds": group[feature_cols].std().round(3).to_dict(),
                "co_p95": float(group["CO"].quantile(0.95)),
                "nox_mean": float(group["NOX"].mean()),
                "tey_mean": float(group["TEY"].mean()) if "TEY" in group.columns else None
            }

    report = {
        "dataset_dimensions": {
            "total_observations": len(df),
            "total_features": len(feature_cols),
            "feature_names": feature_cols
        },
        "descriptive_statistics": desc,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "correlation_with_targets": {
            "CO": {k: v["CO"] for k, v in corr_matrix.items()},
            "NOX": {k: v["NOX"] for k, v in corr_matrix.items()}
        },
        "yearly_evolution": yearly_stats
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report

