"""
Derive state-level face mask mandate start dates using the same idea as the
original face mask project.
"""

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"


df = pd.read_csv(RAW_DATA_DIR / "OxCGRT_AUS_latest.csv")

col_subsets = ["RegionName", "RegionCode", "Date", "H6M_Facial Coverings"]
df = df.loc[:, col_subsets].copy()
df.index = pd.to_datetime(df["Date"], format="%Y%m%d")

rolling_days = 14
df_rolling = (
    df.loc[:, ["RegionName", "H6M_Facial Coverings"]]
    .groupby("RegionName")
    .rolling(window=rolling_days)
    .mean()
)

mandate_limit = 3
df_mandates = (
    df_rolling[df_rolling["H6M_Facial Coverings"] >= mandate_limit]
    .groupby("RegionName")
    .head(1)
)

df_mandates.to_csv(DATA_DIR / "mandate_start_dates.csv")
