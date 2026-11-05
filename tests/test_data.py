"""TURBINE-X Data Loading & Discovery Unit Tests."""
import pytest
from pathlib import Path
import pandas as pd
from src.data.loader import (
    find_project_root,
    load_raw_yearly_data,
    load_dataset_by_year,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS_BY_YEAR,
    TOTAL_EXPECTED_ROWS
)
from src.data.validation import validate_schema


def test_project_root_discovery():
    root = find_project_root()
    assert root.exists()
    assert (root / "data" / "raw" / "gt_2011.csv").exists()


def test_yearly_row_counts():
    root = find_project_root()
    for year, expected_rows in EXPECTED_ROWS_BY_YEAR.items():
        df = load_dataset_by_year(year, root / "data" / "raw")
        assert len(df) == expected_rows, f"Year {year} expected {expected_rows} rows, got {len(df)}"
        assert list(df.columns) == EXPECTED_COLUMNS + ["source_year"]


def test_load_all_yearly_data():
    root = find_project_root()
    df_all = load_raw_yearly_data(root / "data" / "raw")
    assert len(df_all) == TOTAL_EXPECTED_ROWS
    assert df_all.isnull().sum().sum() == 0
    assert "source_year" in df_all.columns


def test_validate_schema_passing():
    root = find_project_root()
    df = load_dataset_by_year(2011, root / "data" / "raw")
    report = validate_schema(df, year=2011)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

