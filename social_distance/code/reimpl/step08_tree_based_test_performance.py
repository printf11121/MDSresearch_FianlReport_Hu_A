from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - depends on local environment
    xgb = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

from .step01_config import RESULTS_DIR, TASK_DATA_DIR
from .step05_helpers import load_test_task, load_training_task, upsample_if_needed


TASKS = [
    {"name": "model_3", "upsample": False},
    {"name": "model_3a", "upsample": True},
    {"name": "model_3b", "upsample": True},
]


def load_selected_params(selection_path: Path) -> dict:
    params = json.loads(selection_path.read_text(encoding="utf-8"))
    for key in ["number", "value", "std_err"]:
        params.pop(key, None)

    for key, value in list(params.items()):
        if pd.isna(value):
            params[key] = None

    for key in ["max_depth", "min_child_weight", "min_samples_split", "min_samples_leaf"]:
        if key in params and params[key] is not None:
            params[key] = int(params[key])

    return params


def build_rf_model(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        bootstrap=True,
        random_state=20240627,
        n_jobs=1,
        **params,
    )


def build_xgb_model(params: dict, y: pd.Series):
    if XGBOOST_IMPORT_ERROR is not None:  # pragma: no cover - depends on local environment
        raise ImportError("xgboost is required to evaluate XGBoost test-set performance.") from XGBOOST_IMPORT_ERROR

    positives = float(y.sum())
    negatives = float(len(y) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0
    return xgb.XGBClassifier(
        n_estimators=250,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=20240627,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        **params,
    )


def evaluate_predictions(y_true: pd.Series, preds, probs) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0),
    }


def evaluate_random_forest_task(task: dict) -> dict[str, float | str]:
    x_train, y_train = load_training_task(task["name"], TASK_DATA_DIR)
    x_test, y_test = load_test_task(task["name"], TASK_DATA_DIR)
    x_fit, y_fit = upsample_if_needed(x_train, y_train, task["upsample"])

    params_path = RESULTS_DIR / f"{task['name']}_rf_best_within_one.json"
    params = load_selected_params(params_path)
    model = build_rf_model(params).fit(x_fit, y_fit)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_predictions(y_test, preds, probs)
    return {"task": task["name"], "model": "random forest", **metrics}


def evaluate_xgboost_task(task: dict) -> dict[str, float | str]:
    x_train, y_train = load_training_task(task["name"], TASK_DATA_DIR)
    x_test, y_test = load_test_task(task["name"], TASK_DATA_DIR)
    x_fit, y_fit = upsample_if_needed(x_train, y_train, task["upsample"])

    params_path = RESULTS_DIR / f"{task['name']}_xgboost_best_within_one.json"
    params = load_selected_params(params_path)
    model = build_xgb_model(params=params, y=y_fit).fit(x_fit, y_fit)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_predictions(y_test, preds, probs)
    return {"task": task["name"], "model": "XGBoost", **metrics}


def run_tree_based_test_performance() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for task in TASKS:
        rows.append(evaluate_random_forest_task(task))
    for task in TASKS:
        rows.append(evaluate_xgboost_task(task))

    output_path = RESULTS_DIR / "tree_based_test_performance.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
