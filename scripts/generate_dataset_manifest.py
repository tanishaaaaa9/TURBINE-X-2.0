"""TURBINE-X Dataset Manifest & Summary Table Generator.
Computes dataset statistics, cryptographic hashes, and exports dataset_summary.csv,
data_quality_report.csv, and dataset_manifest.json.
"""
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "outputs" / "tables"
META_DIR = ROOT / "outputs" / "metadata"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    years = range(2011, 2016)
    file_manifest = []
    frames = []

    for yr in years:
        p = RAW_DIR / f"gt_{yr}.csv"
        df_yr = pd.read_csv(p)
        df_yr["source_year"] = yr
        frames.append(df_yr)

        file_manifest.append({
            "year": yr,
            "filename": p.name,
            "sha256": compute_sha256(p),
            "rows": len(df_yr),
            "columns": len(df_yr.columns) - 1, # excluding source_year
            "file_size_bytes": p.stat().st_size
        })

    master_df = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in master_df.columns if c != "source_year"]

    # 1. Dataset summary table
    summary_df = master_df[feature_cols].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
    summary_df["skewness"] = master_df[feature_cols].skew()
    summary_df["kurtosis"] = master_df[feature_cols].kurtosis()
    summary_df["null_count"] = master_df[feature_cols].isnull().sum()
    summary_df = summary_df.round(4)
    summary_df.to_csv(TABLES_DIR / "dataset_summary.csv")

    # 2. Data quality report CSV
    quality_rows = []
    for col in feature_cols:
        series = master_df[col]
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        iqr = q75 - q25
        outliers_iqr = int(((series < (q25 - 1.5 * iqr)) | (series > (q75 + 1.5 * iqr))).sum())
        mean_val = series.mean()
        std_val = series.std()
        outliers_3sigma = int((np.abs(series - mean_val) > 3 * std_val).sum())

        quality_rows.append({
            "feature": col,
            "total_count": len(series),
            "missing_values": int(series.isnull().sum()),
            "infinite_values": int(np.isinf(series).sum()),
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(series.min(), 4),
            "max": round(series.max(), 4),
            "iqr_outliers": outliers_iqr,
            "iqr_outliers_pct": round(outliers_iqr / len(series) * 100, 2),
            "z_score_3sigma_outliers": outliers_3sigma,
            "z_score_3sigma_pct": round(outliers_3sigma / len(series) * 100, 2)
        })

    pd.DataFrame(quality_rows).to_csv(TABLES_DIR / "data_quality_report.csv", index=False)

    # 3. Complete dataset manifest JSON
    manifest = {
        "dataset_name": "UCI Gas Turbine CO and NOx Emission Data Set",
        "doi": "10.24432/C5WC95",
        "total_observations": len(master_df),
        "total_columns": len(feature_cols),
        "columns": feature_cols,
        "files": file_manifest,
        "chronological_partitions": {
            "development_years": [2011, 2012, 2013],
            "development_rows": int((master_df["source_year"] <= 2013).sum()),
            "calibration_year": [2014],
            "calibration_rows": int((master_df["source_year"] == 2014).sum()),
            "locked_test_year": [2015],
            "locked_test_rows": int((master_df["source_year"] == 2015).sum())
        }
    }

    with open(META_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved dataset_summary.csv, data_quality_report.csv, and dataset_manifest.json successfully.")


if __name__ == "__main__":
    main()

