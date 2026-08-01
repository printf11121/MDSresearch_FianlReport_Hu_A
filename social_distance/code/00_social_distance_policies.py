"""
Create a simple state-level start-date file for social distancing related policies in Australia.

The policy signal is based on the OxCGRT containment indicators that are most directly tied to
social distancing:
    - C3: cancel public events
    - C4: restrictions on gatherings
    - C6: stay at home requirements
    - C7: restrictions on internal movement
"""

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"


df = pd.read_csv(RAW_DATA_DIR / "OxCGRT_AUS_latest.csv")

policy_cols = [
    "C3M_Cancel public events",
    "C4M_Restrictions on gatherings",
    "C6M_Stay at home requirements",
    "C7M_Restrictions on internal movement",
]

df = df.loc[:, ["RegionName", "RegionCode", "Date"] + policy_cols].copy()
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
df["policy_active"] = (df[policy_cols].fillna(0) > 0).any(axis=1).astype(int)

rolling_days = 14
df = df.sort_values(["RegionName", "Date"])
rolling = (
    df.groupby("RegionName")["policy_active"]
    .rolling(window=rolling_days, min_periods=rolling_days)
    .mean()
    .reset_index(level=0, drop=True)
)
df["policy_active_rolling"] = rolling

start_dates = (
    df.loc[df["policy_active_rolling"] >= 1, ["RegionName", "RegionCode", "Date"]]
    .groupby(["RegionName", "RegionCode"], as_index=False)
    .first()
)

start_dates.to_csv(
    DATA_DIR / "social_distance_policy_start_dates.csv",
    index=False,
)
