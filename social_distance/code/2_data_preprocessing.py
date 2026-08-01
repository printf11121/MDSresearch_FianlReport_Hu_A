"""
Preprocess the cleaned data for modelling and EDA.
"""

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def social_distance_policy_convert(row):
    endtime = pd.to_datetime(row["endtime"], format="%Y-%m-%d")
    state = row["state"]
    if state not in social_distance_states_date:
        return 0
    return int(social_distance_states_date[state][0] <= endtime)


def mandate_convert(row):
    endtime = pd.to_datetime(row["endtime"], format="%Y-%m-%d")
    state = row["state"]
    if state not in mandate_states_date:
        return 0
    return int(mandate_states_date[state][0] <= endtime)


cleaned_df = pd.read_csv(
    DATA_DIR / "cleaned_data.csv",
    keep_default_na=False,
)

policy_df = pd.read_csv(
    DATA_DIR / "social_distance_policy_start_dates.csv"
)
social_distance_states_date = {}
for state, date in zip(policy_df["RegionName"], policy_df["Date"]):
    social_distance_states_date.update({state: [date]})

for state, date_range in social_distance_states_date.items():
    social_distance_states_date[state] = [
        pd.to_datetime(date, format="%Y-%m-%d") for date in date_range
    ]

mandate_df = pd.read_csv(
    DATA_DIR / "mandate_start_dates.csv"
)
mandate_states_date = {}
for state, date in zip(mandate_df["RegionName"], mandate_df["Date"]):
    mandate_states_date.update({state: [date]})

for state, date_range in mandate_states_date.items():
    mandate_states_date[state] = [
        pd.to_datetime(date, format="%Y-%m-%d") for date in date_range
    ]

cleaned_df["within_social_distance_policy_period"] = cleaned_df.apply(
    social_distance_policy_convert, axis=1
)
cleaned_df["within_mandate_period"] = cleaned_df.apply(mandate_convert, axis=1)

convert_into_dummy_cols = [
    "state",
    "gender",
    "i9_health",
    "employment_status",
    "i11_health",
    "WCRex1",
    "WCRex2",
    "PHQ4_1",
    "PHQ4_2",
    "PHQ4_3",
    "PHQ4_4",
    "d1_comorbidities",
]

for col in convert_into_dummy_cols:
    dummy = pd.get_dummies(cleaned_df[col], prefix=col, drop_first=True)
    cleaned_df = pd.concat([cleaned_df, dummy], axis=1)
    cleaned_df = cleaned_df.drop(col, axis=1)

cleaned_df.to_csv(
    DATA_DIR / "cleaned_data_preprocessing.csv",
    index=False,
)
