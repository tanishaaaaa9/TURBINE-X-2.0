from pathlib import Path
import pandas as pd

FEATURES_A = ["AT","AP","AH","AFDP","GTEP","TIT","TAT","TEY","CDP"]
TARGETS_A = ["CO","NOX"]
FEATURES_B = ["AT","AP","AH","AFDP","GTEP","TIT","TAT","CDP"]
TARGETS_B = ["CO","NOX","TEY"]
EXPECTED = FEATURES_A + ["CO","NOX"]

def load_yearly_data(raw_dir="data/raw"):
    raw = Path(raw_dir)
    frames=[]
    for year in range(2011, 2016):
        path=raw/f"gt_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Keep gt_2011.csv ... gt_2015.csv in data/raw/")
        df=pd.read_csv(path)
        if list(df.columns) != EXPECTED:
            raise ValueError(f"{path.name}: expected columns {EXPECTED}, got {list(df.columns)}")
        df["source_year"]=year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def split_by_year(df):
    dev=df[df.source_year<=2013].copy()
    calibration=df[df.source_year==2014].copy()
    test=df[df.source_year==2015].copy()
    return dev, calibration, test
