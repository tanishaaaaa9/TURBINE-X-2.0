"""TURBINE-X Dataset Validation Module.
Performs schema, type, missingness, duplicate, and range validation.
"""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from src.data.loader import EXPECTED_COLUMNS, EXPECTED_ROWS_BY_YEAR, TOTAL_EXPECTED_ROWS


def validate_schema(df: pd.DataFrame, year: Optional[int] = None) -> Dict[str, Any]:
    """Perform rigorous schema validation on a gas turbine dataframe.
    
    Args:
        df: Input DataFrame.
        year: Expected source year (optional).
        
    Returns:
        Dictionary summarizing validation status, errors, and statistics.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # 1. Column presence and order
    actual_cols = [c for c in df.columns if c != "source_year"]
    if actual_cols != EXPECTED_COLUMNS:
        errors.append(f"Column mismatch. Expected: {EXPECTED_COLUMNS}, Got: {actual_cols}")

    # 2. Row count check
    if year is not None and year in EXPECTED_ROWS_BY_YEAR:
        expected_rows = EXPECTED_ROWS_BY_YEAR[year]
        if len(df) != expected_rows:
            errors.append(f"Row count mismatch for year {year}. Expected {expected_rows}, got {len(df)}")
    elif len(df) == TOTAL_EXPECTED_ROWS:
        pass  # Combined dataset matches total rows
    
    # 3. Data type validation
    non_numeric_cols = df[actual_cols].select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        errors.append(f"Non-numeric columns found: {non_numeric_cols}")

    # 4. Missing value checks
    null_counts = df[actual_cols].isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        errors.append(f"Dataset contains {total_nulls} null values: {null_counts}")

    # 5. Infinite value checks
    num_df = df[actual_cols].select_dtypes(include=["number"])
    inf_counts = np.isinf(num_df.values).sum()
    if inf_counts > 0:
        errors.append(f"Dataset contains {inf_counts} infinite values.")

    # 6. Duplicate row checks (excluding source_year if present)
    dup_count = int(df[actual_cols].duplicated().sum())
    if dup_count > 0:
        warnings.append(f"Dataset contains {dup_count} exact duplicate rows.")

    # 7. Physical boundary checks for Gas Turbine variables
    physical_limits = {
        "AT": (-15.0, 50.0),      # Ambient temp (°C)
        "AP": (950.0, 1050.0),    # Ambient pressure (mbar)
        "AH": (10.0, 105.0),      # Ambient humidity (%)
        "AFDP": (0.0, 15.0),      # Air filter pressure drop (mbar)
        "GTEP": (10.0, 50.0),     # Exhaust pressure (mbar)
        "TIT": (950.0, 1200.0),   # Turbine inlet temp (°C)
        "TAT": (450.0, 650.0),    # Turbine after temp (°C)
        "TEY": (80.0, 200.0),     # Energy yield (MW)
        "CDP": (8.0, 20.0),       # Compressor discharge pressure (bar)
        "CO": (0.0, 100.0),       # CO emission (mg/m3)
        "NOX": (10.0, 200.0),     # NOx emission (mg/m3)
    }

    out_of_bounds = {}
    for col, (low, high) in physical_limits.items():
        if col in df.columns:
            outside = int(((df[col] < low) | (df[col] > high)).sum())
            if outside > 0:
                out_of_bounds[col] = {
                    "count": outside,
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "expected_range": [low, high]
                }
    if out_of_bounds:
        warnings.append(f"Features with values outside typical physical operating bounds: {list(out_of_bounds.keys())}")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "null_counts": null_counts,
        "duplicate_rows": dup_count,
        "out_of_bounds": out_of_bounds,
    }

