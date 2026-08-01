"""
This script reads the raw Australia survey dataset, counts missing values for each variable,
and saves the summary into data/missing_value_counts.csv.
"""

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"

df = pd.read_csv(
    RAW_DATA_DIR / "australia.csv",
    na_values=[" ", "__NA__"],
    keep_default_na=True,
    low_memory=False,
)

missing_value_counts = {}
for col in df.columns:
    missing_value_counts[col] = int(df[col].isna().sum())

missing_value_df = pd.DataFrame(
    list(missing_value_counts.items()),
    columns=["Variable Name", "Missing Value Count"],
).sort_values(by=["Missing Value Count", "Variable Name"])

missing_value_df.to_csv(
    DATA_DIR / "missing_value_counts.csv",
    index=False,
)
