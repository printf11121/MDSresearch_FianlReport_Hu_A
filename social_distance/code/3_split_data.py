"""
Split data into training and testing sets for social distancing prediction.

We investigate three related problems:
    1. Predicting social distancing overall
    2. Predicting social distancing before mandates
    3. Predicting social distancing during mandate periods
"""

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def stratified_split(frame, stratify_col, test_size=0.2, random_state=20240417):
    test_parts = []
    for _, group in frame.groupby(stratify_col):
        n_test = max(1, int(round(len(group) * test_size)))
        test_parts.append(group.sample(n=n_test, random_state=random_state))

    df_test = pd.concat(test_parts).sort_index()
    df_train = frame.drop(index=df_test.index)
    return df_train.copy(), df_test.copy()


def encode_yes_no(series):
    return series.map({"No": 0, "Yes": 1}).astype(int)


cleaned_df = pd.read_csv(
    DATA_DIR / "cleaned_data_preprocessing.csv",
    keep_default_na=False,
)

df_train, df_test = stratified_split(
    cleaned_df,
    stratify_col="within_mandate_period",
    test_size=0.2,
    random_state=20240417,
)

df_train.to_csv(DATA_DIR / "df_train.csv", index=False)
df_test.to_csv(DATA_DIR / "df_test.csv", index=False)


# %% Model 3: Predicting social distancing overall
response_col = ["social_distancing_binary"]

feature_cols = cleaned_df.columns.drop(
    [
        "RecordNo",
        "social_distancing_scale",
        "social_distancing_binary",
        "endtime",
    ]
)

X_train_model_3 = df_train[feature_cols]
X_test_model_3 = df_test[feature_cols]

y_train_model_3 = encode_yes_no(df_train[response_col].iloc[:, 0])
y_test_model_3 = encode_yes_no(df_test[response_col].iloc[:, 0])

X_train_model_3.to_csv(
    DATA_DIR / "X_train_model_3.csv",
    index=False,
)
X_test_model_3.to_csv(
    DATA_DIR / "X_test_model_3.csv",
    index=False,
)
pd.DataFrame({"y_train": y_train_model_3}).to_csv(
    DATA_DIR / "y_train_model_3.csv",
    index=False,
)
pd.DataFrame({"y_test": y_test_model_3}).to_csv(
    DATA_DIR / "y_test_model_3.csv",
    index=False,
)


# %% Model 3a: Predicting social distancing before mandates
response_col = ["social_distancing_binary"]

feature_cols = cleaned_df.columns.drop(
    [
        "RecordNo",
        "social_distancing_scale",
        "social_distancing_binary",
        "endtime",
        "within_social_distance_policy_period",
        "within_mandate_period",
    ]
)

mandate_starter = "2022-01-01"

logic_subsetter_train = (df_train["endtime"] < mandate_starter) & (
    df_train["within_mandate_period"] == 0
)
logic_subsetter_test = (df_test["endtime"] < mandate_starter) & (
    df_test["within_mandate_period"] == 0
)

X_train_model_3a = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_3a = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_3a = encode_yes_no(df_train.loc[logic_subsetter_train, response_col].iloc[:, 0])
y_test_model_3a = encode_yes_no(df_test.loc[logic_subsetter_test, response_col].iloc[:, 0])

X_train_model_3a.to_csv(
    DATA_DIR / "X_train_model_3a.csv",
    index=False,
)
X_test_model_3a.to_csv(
    DATA_DIR / "X_test_model_3a.csv",
    index=False,
)
pd.DataFrame({"y_train": y_train_model_3a}).to_csv(
    DATA_DIR / "y_train_model_3a.csv",
    index=False,
)
pd.DataFrame({"y_test": y_test_model_3a}).to_csv(
    DATA_DIR / "y_test_model_3a.csv",
    index=False,
)


# %% Model 3b: Predicting social distancing during mandate periods
response_col = ["social_distancing_binary"]

feature_cols = cleaned_df.columns.drop(
    [
        "RecordNo",
        "social_distancing_scale",
        "social_distancing_binary",
        "endtime",
        "within_social_distance_policy_period",
        "within_mandate_period",
    ]
)

logic_subsetter_train = df_train["within_mandate_period"] == 1
logic_subsetter_test = df_test["within_mandate_period"] == 1

X_train_model_3b = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_3b = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_3b = encode_yes_no(df_train.loc[logic_subsetter_train, response_col].iloc[:, 0])
y_test_model_3b = encode_yes_no(df_test.loc[logic_subsetter_test, response_col].iloc[:, 0])

X_train_model_3b.to_csv(
    DATA_DIR / "X_train_model_3b.csv",
    index=False,
)
X_test_model_3b.to_csv(
    DATA_DIR / "X_test_model_3b.csv",
    index=False,
)
pd.DataFrame({"y_train": y_train_model_3b}).to_csv(
    DATA_DIR / "y_train_model_3b.csv",
    index=False,
)
pd.DataFrame({"y_test": y_test_model_3b}).to_csv(
    DATA_DIR / "y_test_model_3b.csv",
    index=False,
)
