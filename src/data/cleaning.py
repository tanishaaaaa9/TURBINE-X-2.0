"""TURBINE-X Data Cleaning & Anomaly Audit Module.
Audits anomalies, duplicates, and physical limits without silent data dropping.
Every cleaning decision is logged and traceable.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


class DataCleaner:
    """Audits data quality and cleans observations with full audit trail."""

    def __init__(self, drop_duplicates: bool = False, handle_outliers: str = "flag"):
        """
        Args:
            drop_duplicates: If True, duplicate rows are removed (logged). If False, only flagged.
            handle_outliers: 'flag' (default, non-destructive), 'clip' (winsorize to 1st/99th percentiles).
        """
        self.drop_duplicates = drop_duplicates
        self.handle_outliers = handle_outliers
        self.audit_log: List[Dict[str, Any]] = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Audit and clean dataset while preserving statistical integrity."""
        cleaned_df = df.copy()
        initial_rows = len(cleaned_df)
        operations = []

        # 1. Null handling check
        null_count = cleaned_df.isnull().sum().sum()
        if null_count > 0:
            cleaned_df = cleaned_df.dropna()
            dropped_nulls = initial_rows - len(cleaned_df)
            operations.append(f"Dropped {dropped_nulls} rows due to NaN values.")
        else:
            operations.append("Zero missing values verified across all columns.")

        # 2. Duplicate handling
        feature_cols = [c for c in cleaned_df.columns if c != "source_year"]
        num_dups = int(cleaned_df[feature_cols].duplicated().sum())
        if num_dups > 0:
            if self.drop_duplicates:
                cleaned_df = cleaned_df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
                operations.append(f"Removed {num_dups} duplicate records.")
            else:
                operations.append(f"Identified {num_dups} duplicate records (retained to preserve natural sampling frequency).")
        else:
            operations.append("Zero duplicate records found.")

        # 3. Outlier profiling (Tukey's IQR & 3-sigma bounds)
        outlier_summary = {}
        for col in feature_cols:
            series = cleaned_df[col]
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr
            iqr_outliers = int(((series < lower_fence) | (series > upper_fence)).sum())

            mean_val = series.mean()
            std_val = series.std()
            z_outliers = int((np.abs((series - mean_val) / (std_val + 1e-8)) > 3.0).sum())

            outlier_summary[col] = {
                "iqr_outliers_count": iqr_outliers,
                "iqr_outliers_pct": round(iqr_outliers / len(cleaned_df) * 100, 2),
                "z_score_3sigma_count": z_outliers,
                "z_score_3sigma_pct": round(z_outliers / len(cleaned_df) * 100, 2),
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(iqr),
                "min": float(series.min()),
                "max": float(series.max())
            }

            # Optional non-destructive clipping
            if self.handle_outliers == "clip":
                p01 = series.quantile(0.01)
                p99 = series.quantile(0.99)
                cleaned_df[col] = cleaned_df[col].clip(p01, p99)

        if self.handle_outliers == "clip":
            operations.append("Winsorized extreme values to 1st and 99th percentiles.")
        else:
            operations.append("Outliers flagged and retained without destructive clipping.")

        report = {
            "initial_rows": initial_rows,
            "final_rows": len(cleaned_df),
            "rows_removed": initial_rows - len(cleaned_df),
            "operations": operations,
            "outlier_summary": outlier_summary
        }

        self.audit_log.append(report)
        return cleaned_df, report

