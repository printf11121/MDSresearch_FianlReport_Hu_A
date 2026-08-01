from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - depends on local environment
    xgb = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError as exc:  # pragma: no cover - depends on local environment
    StratifiedShuffleSplit = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None

from .step01_config import CV_RANDOM_STATE, RESULTS_DIR, TASK_DATA_DIR
from .step05_helpers import load_training_task, load_trials_frame, summarize_folds, upsample_if_needed


@dataclass(frozen=True)
class XGBTask:
    name: str
    upsample: bool
    search_mode: str
    sort_order: tuple[str, ...]


TASKS = [
    XGBTask(
        name="model_3",
        upsample=False,
        search_mode="full_task",
        sort_order=("learning_rate", "subsample", "max_depth", "colsample_bytree"),
    ),
    XGBTask(
        name="model_3a",
        upsample=True,
        search_mode="learning_subsample",
        sort_order=("learning_rate", "subsample"),
    ),
    XGBTask(
        name="model_3b",
        upsample=True,
        search_mode="mandate_like",
        sort_order=("learning_rate", "subsample", "max_depth", "min_child_weight"),
    ),
]


def ensure_dependencies() -> None:
    if XGBOOST_IMPORT_ERROR is not None:  # pragma: no cover - depends on local environment
        raise ImportError("xgboost is required to run the XGBoost pipeline.") from XGBOOST_IMPORT_ERROR
    if SKLEARN_IMPORT_ERROR is not None:  # pragma: no cover - depends on local environment
        raise ImportError("scikit-learn is required to run the XGBoost pipeline.") from SKLEARN_IMPORT_ERROR


def get_splitter() -> StratifiedShuffleSplit:
    ensure_dependencies()
    return StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=CV_RANDOM_STATE)


def suggest_params(trial: optuna.Trial, task: XGBTask) -> dict[str, float | int]:
    if task.search_mode == "learning_subsample":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        }

    if task.search_mode == "mandate_like":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        }

    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
    }


def build_model(params: dict[str, float | int], y: pd.Series) -> xgb.XGBClassifier:
    ensure_dependencies()
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0

    model_params = {
        "n_estimators": 250,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": CV_RANDOM_STATE,
        "n_jobs": 1,
        "scale_pos_weight": scale_pos_weight,
        **params,
    }
    return xgb.XGBClassifier(**model_params)


def evaluate_fold_metrics(task: XGBTask, params: dict[str, float | int], x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    splitter = get_splitter()
    rows: list[dict[str, float | int | str]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        x_valid = x.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        x_fit, y_fit = upsample_if_needed(x_train, y_train, task.upsample)

        model = build_model(params=params, y=y_fit)
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


def objective_factory(task: XGBTask, x: pd.DataFrame, y: pd.Series):
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


def optimize_task(task: XGBTask, n_trials: int) -> dict[str, Path]:
    x, y = load_training_task(task.name, TASK_DATA_DIR)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2020))
    study.optimize(objective_factory(task, x, y), n_trials=n_trials, n_jobs=1)

    trials_path = RESULTS_DIR / f"{task.name}_xgboost_trials.jsonl"
    best_path = RESULTS_DIR / f"{task.name}_xgboost_trial_best.json"
    write_trials_jsonl(study, trials_path)
    write_best_trial_json(study, best_path)

    return {"trials": trials_path, "best": best_path}


def select_best_within_one_std(task: XGBTask, trials_path: Path, best_path: Path) -> Path:
    trials_df = load_trials_frame(trials_path)
    best_payload = json.loads(best_path.read_text(encoding="utf-8"))
    threshold = best_payload["value"] - best_payload["user_attrs"]["std_err"]

    shortlisted = trials_df.loc[trials_df["value"] >= threshold].copy()
    if shortlisted.empty:
        shortlisted = trials_df.sort_values("value", ascending=False).head(1).copy()
    shortlisted = shortlisted.sort_values(list(task.sort_order)).reset_index(drop=True)
    selected = shortlisted.iloc[0].to_dict()

    output_path = RESULTS_DIR / f"{task.name}_xgboost_best_within_one.json"
    output_path.write_text(json.dumps(selected), encoding="utf-8")
    return output_path


def load_selected_params(selection_path: Path) -> dict[str, float | int]:
    params = json.loads(selection_path.read_text(encoding="utf-8"))
    params.pop("number", None)
    params.pop("value", None)
    params.pop("std_err", None)

    if "max_depth" in params and params["max_depth"] is not None:
        params["max_depth"] = int(params["max_depth"])
    if "min_child_weight" in params and params["min_child_weight"] is not None:
        params["min_child_weight"] = int(params["min_child_weight"])

    return params


def cross_validate_task(task: XGBTask, selection_path: Path) -> pd.DataFrame:
    x, y = load_training_task(task.name, TASK_DATA_DIR)
    params = load_selected_params(selection_path)
    fold_df = evaluate_fold_metrics(task=task, params=params, x=x, y=y)
    fold_df.insert(0, "model", task.name)

    with open(RESULTS_DIR / f"{task.name}_xgboost.pkl", "wb") as handle:
        pickle.dump(fold_df, handle)

    return fold_df


def run_xgboost_pipeline(n_trials: int = 100) -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_folds = []
    for task in TASKS:
        paths = optimize_task(task, n_trials=n_trials)
        selection_path = select_best_within_one_std(task, paths["trials"], paths["best"])
        all_folds.append(cross_validate_task(task, selection_path))

    folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = summarize_folds(folds_df)

    folds_path = RESULTS_DIR / "xgboost_cv_folds.csv"
    summary_path = RESULTS_DIR / "xgboost_cv_summary.csv"
    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {"folds": folds_path, "summary": summary_path}
