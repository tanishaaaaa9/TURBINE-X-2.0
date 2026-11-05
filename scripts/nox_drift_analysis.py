import pandas as pd
import numpy as np
from pathlib import Path

root = Path("D:/TURBINE_X_Full_Project/TURBINE_X_Full_Project")
for y in range(2011, 2016):
    df = pd.read_csv(root / "data" / "raw" / f"gt_{y}.csv")
    print(f"Year {y}: N={len(df)} | CO mean={df['CO'].mean():.2f}, std={df['CO'].std():.2f} | NOX mean={df['NOX'].mean():.2f}, std={df['NOX'].std():.2f} | TEY mean={df['TEY'].mean():.2f}, std={df['TEY'].std():.2f}")

# Check dev (2011-2013) vs test (2015) NOX distribution
dev_frames = [pd.read_csv(root / "data" / "raw" / f"gt_{y}.csv") for y in [2011, 2012, 2013]]
dev_df = pd.concat(dev_frames)
test_df = pd.read_csv(root / "data" / "raw" / "gt_2015.csv")

print("\n--- Distribution Comparison ---")
print(f"Dev (2011-2013) NOX: mean = {dev_df['NOX'].mean():.2f}, median = {dev_df['NOX'].median():.2f}, std = {dev_df['NOX'].std():.2f}, min = {dev_df['NOX'].min():.2f}, max = {dev_df['NOX'].max():.2f}")
print(f"Test (2015) NOX:     mean = {test_df['NOX'].mean():.2f}, median = {test_df['NOX'].median():.2f}, std = {test_df['NOX'].std():.2f}, min = {test_df['NOX'].min():.2f}, max = {test_df['NOX'].max():.2f}")

# Notice mean shift
diff_mean = test_df['NOX'].mean() - dev_df['NOX'].mean()
print(f"Mean Shift in NOX: {diff_mean:.2f} mg/m3")
print(f"Total variance in Test NOX: {test_df['NOX'].var():.2f}")
print(f"Mean squared error due to shift alone: {diff_mean**2:.2f}")

# Check Explained Variance vs R2:
# R2 = 1 - MSE / Var(y_true)
# If a model predicts with a systematic offset (e.g. diff_mean), MSE will exceed Var(y_true), causing negative R2!

