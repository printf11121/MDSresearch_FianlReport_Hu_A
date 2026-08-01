from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from .step01_config import CV_RANDOM_STATE, RESULTS_DIR, TASK_DATA_DIR
from .step05_helpers import load_training_task, summarize_folds, upsample_if_needed


@dataclass(frozen=True)
class LogisticTask:
    name: str
    upsample: bool


TASKS = [
    LogisticTask("model_3", upsample=False),
    LogisticTask("model_3a", upsample=True),
    LogisticTask("model_3b", upsample=True),
]


def run_single_task(task: LogisticTask) -> pd.DataFrame:
    x, y = load_training_task(task.name, TASK_DATA_DIR)
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=CV_RANDOM_STATE)
    model = LogisticRegression(max_iter=5000, random_state=CV_RANDOM_STATE)

    rows = []
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        x_valid = x.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        x_train, y_train = upsample_if_needed(x_train, y_train, task.upsample)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)

        clf = model.fit(x_train_scaled, y_train)
        preds = clf.predict(x_valid_scaled)
        probs = clf.predict_proba(x_valid_scaled)[:, 1]

        rows.append(
            {
                "model": task.name,
                "fold": fold_id,
                "test_precision": precision_score(y_valid, preds, zero_division=0),
                "test_recall": recall_score(y_valid, preds, zero_division=0),
                "test_roc_auc": roc_auc_score(y_valid, probs),
                "test_accuracy": accuracy_score(y_valid, preds),
                "test_f1": f1_score(y_valid, preds, zero_division=0),
            }
        )

    task_df = pd.DataFrame(rows)
    with open(RESULTS_DIR / f"{task.name}_logistic_reg.pkl", "wb") as handle:
        pickle.dump(task_df, handle)
    return task_df


def run_logistic_baseline() -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = [run_single_task(task) for task in TASKS]
    folds_df = pd.concat(all_frames, ignore_index=True)
    summary_df = summarize_folds(folds_df)

    folds_path = RESULTS_DIR / "logistic_regression_cv_folds.csv"
    summary_path = RESULTS_DIR / "logistic_regression_cv_summary.csv"
    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {"folds": folds_path, "summary": summary_path}
