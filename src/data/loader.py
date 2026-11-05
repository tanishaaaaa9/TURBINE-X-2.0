"""TURBINE-X Data Loading & Discovery Module.
Provides robust discovery and loading of the annual UCI Gas Turbine datasets (2011-2015).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Canonical feature and target definitions
FEATURES_A = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGETS_A = ["CO", "NOX"]

FEATURES_B = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"]
TARGETS_B = ["CO", "NOX", "TEY"]

EXPECTED_COLUMNS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"]
EXPECTED_ROWS_BY_YEAR = {2011: 7411, 2012: 7628, 2013: 7152, 2014: 7158, 2015: 7384}
TOTAL_EXPECTED_ROWS = sum(EXPECTED_ROWS_BY_YEAR.values())  # 36,733


def find_project_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """Dynamically locate the TURBINE-X project root directory."""
    if start_path is not None:
        p = Path(start_path).resolve()
        if (p / "data" / "raw" / "gt_2011.csv").exists():
            return p
    # Reliable anchor: 2 levels up from src/data/loader.py is the project root
    module_root = Path(__file__).resolve().parents[2]
    if (module_root / "data" / "raw" / "gt_2011.csv").exists():
        return module_root
    # Check current directory and subdirectories
    cwd = Path.cwd().resolve()
    if (cwd / "data" / "raw" / "gt_2011.csv").exists():
        return cwd
    if (cwd / "TURBINE_X_Full_Project" / "data" / "raw" / "gt_2011.csv").exists():
        return cwd / "TURBINE_X_Full_Project"
    return module_root


def load_raw_yearly_data(raw_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and concatenate all annual CSV files (2011-2015) with metadata.
    
    Args:
        raw_dir: Path to directory containing gt_2011.csv ... gt_2015.csv.
                 If None, automatically discovered from project root.
                 
    Returns:
        pd.DataFrame containing all 36,733 records with 'source_year' column added.
    """
    if raw_dir is None:
        raw_dir = find_project_root() / "data" / "raw"
    raw_path = Path(raw_dir)

    frames = []
    for year in range(2011, 2016):
        csv_file = raw_path / f"gt_{year}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {csv_file}. Please ensure gt_2011.csv through gt_2015.csv are present."
            )
        df_year = pd.read_csv(csv_file)
        
        # Validate columns
        if list(df_year.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Column mismatch in {csv_file.name}. Expected {EXPECTED_COLUMNS}, got {list(df_year.columns)}"
            )
        
        df_year["source_year"] = year
        frames.append(df_year)

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


def load_dataset_by_year(year: int, raw_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load a specific single-year dataset.
    
    Args:
        year: Year between 2011 and 2015.
        raw_dir: Raw data directory path.
    """
    if year not in EXPECTED_ROWS_BY_YEAR:
        raise ValueError(f"Year {year} is not in valid range (2011-2015).")
    if raw_dir is None:
        raw_dir = find_project_root() / "data" / "raw"
    csv_file = Path(raw_dir) / f"gt_{year}.csv"
    df = pd.read_csv(csv_file)
    df["source_year"] = year
    return df
