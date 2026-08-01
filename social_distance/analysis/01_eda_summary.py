"""
Generate basic EDA summaries and figures for the social distancing outcome.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

cleaned_df = pd.read_csv(
    DATA_DIR / "cleaned_data.csv",
    keep_default_na=False,
)
model_df = pd.read_csv(
    DATA_DIR / "cleaned_data_preprocessing.csv",
    keep_default_na=False,
)

numeric_cols = [
    "social_distancing_scale",
    "age",
    "household_size",
    "cantril_ladder",
    "r1_1",
    "r1_2",
    "week_number",
]

numeric_summary = cleaned_df[numeric_cols].describe().T
numeric_summary.to_csv(RESULTS_DIR / "eda_numeric_summary.csv")

categorical_cols = [
    "social_distancing_binary",
    "state",
    "gender",
    "employment_status",
    "i9_health",
    "i11_health",
    "d1_comorbidities",
]

rows = []
for col in categorical_cols:
    counts = cleaned_df[col].value_counts(dropna=False)
    proportions = cleaned_df[col].value_counts(normalize=True, dropna=False)
    for level in counts.index:
        rows.append(
            {
                "variable": col,
                "level": level,
                "count": int(counts[level]),
                "proportion": float(proportions[level]),
            }
        )
categorical_summary = pd.DataFrame(rows)
categorical_summary.to_csv(
    RESULTS_DIR / "eda_categorical_summary.csv",
    index=False,
)

policy_summary = (
    model_df["within_social_distance_policy_period"]
    .value_counts(normalize=True)
    .rename_axis("within_social_distance_policy_period")
    .reset_index(name="proportion")
)
policy_summary.to_csv(
    RESULTS_DIR / "eda_policy_period_summary.csv",
    index=False,
)

weekly_summary = (
    cleaned_df.groupby("week_number")["social_distancing_scale"]
    .median()
    .reset_index(name="median_social_distancing_scale")
)
weekly_summary.to_csv(
    RESULTS_DIR / "eda_weekly_social_distancing_summary.csv",
    index=False,
)

plt.figure(figsize=(8, 6))
plt.hist(
    cleaned_df["social_distancing_scale"],
    bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    color="#2C7FB8",
    edgecolor="white",
    linewidth=1.2,
)
plt.title("Distribution of Social Distancing Score")
plt.xlabel("Social distancing score (1 = weakest, 5 = strongest)")
plt.ylabel("Number of respondents")
plt.xlim(0.75, 5.25)
plt.xticks([1, 2, 3, 4, 5])
plt.grid(axis="y", alpha=0.35)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "social_distancing_scale_distribution.png",
    dpi=300,
)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(
    weekly_summary["week_number"],
    weekly_summary["median_social_distancing_scale"],
    marker="o",
)
plt.title("Median Social Distancing Scale Over Time")
plt.xlabel("Two-week period")
plt.ylabel("Median social distancing scale")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "social_distancing_scale_over_time.png",
    dpi=300,
)
plt.close()

plt.figure(figsize=(8, 6))
binary_counts = cleaned_df["social_distancing_binary"].value_counts()
plt.bar(binary_counts.index, binary_counts.values)
plt.title("Social Distancing Binary Outcome")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "social_distancing_binary_counts.png",
    dpi=300,
)
plt.close()
