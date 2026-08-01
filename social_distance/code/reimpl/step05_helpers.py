from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd

from .step01_config import CV_RANDOM_STATE


def load_training_task(task_name: str, task_data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(task_data_dir / f"X_train_{task_name}.csv", keep_default_na=False)
    y = pd.read_csv(task_data_dir / f"y_train_{task_name}.csv", keep_default_na=False).iloc[:, 0]
    return x, y.astype(int)


def load_test_task(task_name: str, task_data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(task_data_dir / f"X_test_{task_name}.csv", keep_default_na=False)
    y = pd.read_csv(task_data_dir / f"y_test_{task_name}.csv", keep_default_na=False).iloc[:, 0]
    return x, y.astype(int)


def upsample_if_needed(x: pd.DataFrame, y: pd.Series, enabled: bool) -> tuple[pd.DataFrame, pd.Series]:
    if not enabled:
        return x.reset_index(drop=True), y.reset_index(drop=True)

    sampled = x.copy()
    sampled["_target"] = y.to_numpy()
    max_size = int(sampled["_target"].value_counts().max())

    parts = []
    for _, group in sampled.groupby("_target"):
        parts.append(group.sample(n=max_size, replace=True, random_state=CV_RANDOM_STATE))

    balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=CV_RANDOM_STATE)
    y_balanced = balanced.pop("_target").astype(int)
    return balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)


def load_trials_frame(trials_path: Path) -> pd.DataFrame:
    rows = []
    with trials_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            params = entry.pop("params", {})
            user_attrs = entry.pop("user_attrs", {})
            entry.update(params)
            entry.update(user_attrs)
            rows.append(entry)
    return pd.DataFrame(rows)


def summarize_folds(folds_df: pd.DataFrame) -> pd.DataFrame:
    return (
        folds_df.groupby("model")
        .apply(
            lambda frame: pd.Series(
                {
                    "precision_mean": frame["test_precision"].mean(),
                    "precision_std": frame["test_precision"].std(ddof=1) / (len(frame) ** 0.5),
                    "recall_mean": frame["test_recall"].mean(),
                    "recall_std": frame["test_recall"].std(ddof=1) / (len(frame) ** 0.5),
                    "roc_auc_mean": frame["test_roc_auc"].mean(),
                    "roc_auc_std": frame["test_roc_auc"].std(ddof=1) / (len(frame) ** 0.5),
                    "accuracy_mean": frame["test_accuracy"].mean(),
                    "accuracy_std": frame["test_accuracy"].std(ddof=1) / (len(frame) ** 0.5),
                    "f1_mean": frame["test_f1"].mean(),
                    "f1_std": frame["test_f1"].std(ddof=1) / (len(frame) ** 0.5),
                }
            )
        )
        .reset_index()
    )


def iter_param_grid(param_grid: dict[str, list[object]]):
    keys = list(param_grid)
    for values in product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values))
