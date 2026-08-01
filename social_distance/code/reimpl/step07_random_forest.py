from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError as exc:  # pragma: no cover - depends on local environment
    RandomForestClassifier = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None

from .step01_config import CV_RANDOM_STATE, RESULTS_DIR, TASK_DATA_DIR
from .step05_helpers import load_training_task, load_trials_frame, summarize_folds, upsample_if_needed


@dataclass(frozen=True)
class RFTask:
    name: str
    upsample: bool
    search_mode: str
    sort_order: tuple[str, ...]


TASKS = [
    RFTask(
        name="model_3",
        upsample=False,
        search_mode="all_params",
        sort_order=("max_depth", "min_samples_leaf", "max_features", "min_samples_split"),
    ),
    RFTask(
        name="model_3a",
        upsample=True,
        search_mode="subset_with_max_features",
        sort_order=("max_depth", "min_samples_leaf", "max_features"),
    ),
    RFTask(
        name="model_3b",
        upsample=True,
        search_mode="subset_with_max_features",
        sort_order=("max_depth", "min_samples_leaf", "max_features"),
    ),
]


def ensure_dependencies() -> None:
    if SKLEARN_IMPORT_ERROR is not None:  # pragma: no cover - depends on local environment
        raise ImportError("scikit-learn is required to run the Random Forest pipeline.") from SKLEARN_IMPORT_ERROR


def get_splitter() -> StratifiedShuffleSplit:
    ensure_dependencies()
    return StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=CV_RANDOM_STATE)


def suggest_params(trial: optuna.Trial, task: RFTask) -> dict[str, int | str | None]:
    if task.search_mode == "subset_with_max_features":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    return {
        "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 300),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def build_model(params: dict[str, int | str | None]) -> RandomForestClassifier:
    ensure_dependencies()
    return RandomForestClassifier(
        n_estimators=250,
        bootstrap=True,
        random_state=CV_RANDOM_STATE,
        n_jobs=1,
        **params,
    )


def evaluate_fold_metrics(task: RFTask, params: dict[str, int | str | None], x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    splitter = get_splitter()
    rows: list[dict[str, float | int | str]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        x_valid = x.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        x_fit, y_fit = upsample_if_needed(x_train, y_train, task.upsample)

        model = build_model(params=params)
        fitted = model.fit(x_fit, y_fit)

        preds = fitted.predict(x_valid)
        probs = fitted.predict_proba(x_valid)[:, 1]

        rows.append(
            {
                "fold": fold_id,
                "test_precision": precision_score(y_valid, preds, zero_division=0),
                "test_recall": recall_score(y_valid, preds, zero_division=0),
                "test_roc_auc": roc_auc_score(y_valid, probs),
                "test_accuracy": accuracy_score(y_valid, preds),
                "test_f1": f1_score(y_valid, preds, zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def objective_factory(task: RFTask, x: pd.DataFrame, y: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, task)
        fold_df = evaluate_fold_metrics(task=task, params=params, x=x, y=y)
        roc_auc_values = fold_df["test_roc_auc"].to_numpy()
        trial.set_user_attr("std_err", float(np.std(roc_auc_values, ddof=0) / np.sqrt(len(roc_auc_values))))
        return float(np.mean(roc_auc_values))

    return objective


def write_trials_jsonl(study: optuna.Study, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for trial in study.trials:
            payload = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            handle.write(json.dumps(payload) + "\n")


def write_best_trial_json(study: optuna.Study, output_path: Path) -> None:
    best_trial = study.best_trial
    payload = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def optimize_task(task: RFTask, n_trials: int) -> dict[str, Path]:
    x, y = load_training_task(task.name, TASK_DATA_DIR)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2013))
    study.optimize(objective_factory(task, x, y), n_trials=n_trials, n_jobs=1)

    trials_path = RESULTS_DIR / f"{task.name}_rf_trials.jsonl"
    best_path = RESULTS_DIR / f"{task.name}_rf_trial_best.json"
    write_trials_jsonl(study, trials_path)
    write_best_trial_json(study, best_path)

    return {"trials": trials_path, "best": best_path}


def select_best_within_one_std(task: RFTask, trials_path: Path, best_path: Path) -> Path:
    trials_df = load_trials_frame(trials_path)
    best_payload = json.loads(best_path.read_text(encoding="utf-8"))
    threshold = best_payload["value"] - best_payload["user_attrs"]["std_err"]

    shortlisted = trials_df.loc[trials_df["value"] >= threshold].copy()
    if shortlisted.empty:
        shortlisted = trials_df.sort_values("value", ascending=False).head(1).copy()
    shortlisted = shortlisted.sort_values(list(task.sort_order)).reset_index(drop=True)
    selected = shortlisted.iloc[0].to_dict()

    output_path = RESULTS_DIR / f"{task.name}_rf_best_within_one.json"
    output_path.write_text(json.dumps(selected), encoding="utf-8")
    return output_path


def load_selected_params(selection_path: Path) -> dict[str, int | str | None]:
    params = json.loads(selection_path.read_text(encoding="utf-8"))
    params.pop("number", None)
    params.pop("value", None)
    params.pop("std_err", None)

    for key, value in list(params.items()):
        if pd.isna(value):
            params[key] = None

    if "max_depth" in params and params["max_depth"] is not None:
        params["max_depth"] = int(params["max_depth"])
    if "min_samples_split" in params and params["min_samples_split"] is not None:
        params["min_samples_split"] = int(params["min_samples_split"])
    if "min_samples_leaf" in params and params["min_samples_leaf"] is not None:
        params["min_samples_leaf"] = int(params["min_samples_leaf"])

    return params


def cross_validate_task(task: RFTask, selection_path: Path) -> pd.DataFrame:
    x, y = load_training_task(task.name, TASK_DATA_DIR)
    params = load_selected_params(selection_path)
    fold_df = evaluate_fold_metrics(task=task, params=params, x=x, y=y)
    fold_df.insert(0, "model", task.name)

    with open(RESULTS_DIR / f"{task.name}_rf.pkl", "wb") as handle:
        pickle.dump(fold_df, handle)

    return fold_df


def run_random_forest_pipeline(n_trials: int = 100) -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_folds = []
    for task in TASKS:
        paths = optimize_task(task, n_trials=n_trials)
        selection_path = select_best_within_one_std(task, paths["trials"], paths["best"])
        all_folds.append(cross_validate_task(task, selection_path))

    folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = summarize_folds(folds_df)

    folds_path = RESULTS_DIR / "random_forest_cv_folds.csv"
    summary_path = RESULTS_DIR / "random_forest_cv_summary.csv"
    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {"folds": folds_path, "summary": summary_path}
