from pathlib import Path
import pandas as pd
from src.data.schema import validate_frame, EXPECTED_ROWS_BY_YEAR

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def test_each_year_schema_and_missingness():
    for year, rows in EXPECTED_ROWS_BY_YEAR.items():
        df = pd.read_csv(RAW / f"gt_{year}.csv")
        assert len(df) == rows
        assert validate_frame(df, year) == []
