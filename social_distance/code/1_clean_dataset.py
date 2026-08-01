"""
Clean the Australia survey data and construct a social distancing outcome.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"


def convert_datetime(dt):
    date = dt.split()[0]
    return datetime.strptime(date, "%d/%m/%Y")


def household_convert(size_str):
    for i in range(1, 8):
        if size_str == str(i):
            return i
        if size_str == "8 or more":
            return 8
    if size_str in ["Prefer not to say", "Don't know"]:
        return None
    return None


df = pd.read_csv(
    RAW_DATA_DIR / "australia.csv",
    na_values=[" ", "__NA__"],
    keep_default_na=True,
    low_memory=False,
)

df["endtime"] = df["endtime"].apply(convert_datetime)

thresh_value = 10781
missing_value_df = pd.read_csv(
    DATA_DIR / "missing_value_counts.csv"
)
columns_to_drop = missing_value_df.loc[
    missing_value_df["Missing Value Count"] > thresh_value, "Variable Name"
].tolist()
df.drop(columns=columns_to_drop, inplace=True)

sdate = "2021-02-10"
edate = "2021-10-18"
mask = (df["endtime"] <= edate) & (df["endtime"] >= sdate)

for i in range(1, 5):
    df.loc[mask, f"PHQ4_{i}"] = df.loc[mask, f"PHQ4_{i}"].fillna("N/A")
for i in range(1, 14):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")
for i in range(98, 100):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")

df.dropna(inplace=True)

for i in range(1, 3):
    df[f"r1_{i}"] = df[f"r1_{i}"].replace(
        {
            "7 - Agree": 7,
            "6": 6,
            "5": 5,
            "4": 4,
            "3": 3,
            "2": 2,
            "1 – Disagree": 1,
        }
    )

frequency_dict = {
    "Always": 5,
    "Frequently": 4,
    "Sometimes": 3,
    "Rarely": 2,
    "Not at all": 1,
}
for column in df.columns:
    if column.startswith("i12_health_"):
        df[column] = df[column].map(frequency_dict)

social_distancing_cols = [
    "i12_health_11",
    "i12_health_12",
    "i12_health_13",
    "i12_health_14",
    "i12_health_15",
    "i12_health_16",
    "i12_health_26",
]

available_social_distancing_cols = [
    col for col in social_distancing_cols if col in df.columns
]

df["social_distancing_scale"] = df[available_social_distancing_cols].median(axis=1)
df["social_distancing_binary"] = df["social_distancing_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No"
)

d1_cols = [col for col in df.columns if col.startswith("d1_")]
df["d1_comorbidities"] = "Yes"
df.loc[df["d1_health_99"] == "Yes", "d1_comorbidities"] = "No"
df.loc[df["d1_health_99"] == "N/A", "d1_comorbidities"] = "NA"
df.loc[df["d1_health_98"] == "Yes", "d1_comorbidities"] = "Prefer_not_to_say"
df = df.drop(d1_cols, axis=1)

start_date = df["endtime"].min()
df["week_number"] = ((df["endtime"] - start_date).dt.days // 14) + 1

df["household_size"] = df["household_size"].apply(household_convert)
df.dropna(inplace=True)

i12_cols = [col for col in df.columns if col.startswith("i12_health_")]

df = df.drop(["qweek", "weight"] + i12_cols, axis=1)
df.to_csv(DATA_DIR / "cleaned_data.csv", index=False)
