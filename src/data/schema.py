EXPECTED_COLUMNS = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO', 'NOX']
EXPECTED_ROWS_BY_YEAR = {2011: 7411, 2012: 7628, 2013: 7152, 2014: 7158, 2015: 7384}

def validate_frame(df, year=None):
    errors = []
    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(f"Column mismatch: {list(df.columns)}")
    if df.isna().any().any():
        errors.append("Missing values detected")
    if not df.select_dtypes(include="number").shape[1] == len(EXPECTED_COLUMNS):
        errors.append("Unexpected non-numeric column")
    if year in EXPECTED_ROWS_BY_YEAR and len(df) != EXPECTED_ROWS_BY_YEAR[year]:
        errors.append(f"Unexpected row count for {year}: {len(df)}")
    return errors
