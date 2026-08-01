from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

from reimpl.step01_config import RESULTS_DIR, TASK_DATA_DIR
from reimpl.step05_helpers import load_training_task


TARGET_TASKS = ["model_3", "model_3a", "model_3b"]
MODEL_TYPES = ["rf", "xgboost"]
NUM_PERMUTATIONS = 80
TOP_N = 10


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_params(task_name: str, model_type: str) -> dict:
    params = load_json(RESULTS_DIR / f"{task_name}_{model_type}_best_within_one.json")
    for key in ["number", "value", "std_err"]:
        params.pop(key, None)

    for key, value in list(params.items()):
        if pd.isna(value):
            params[key] = None

    if model_type == "rf":
        for key in ["max_depth", "min_samples_split", "min_samples_leaf"]:
            if key in params and params[key] is not None:
                params[key] = int(params[key])
    else:
        for key in ["max_depth", "min_child_weight"]:
            if key in params and params[key] is not None:
                params[key] = int(params[key])

    return params


def random_upsample(x: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    sampled = x.copy()
    sampled["_target"] = y.to_numpy()
    max_size = int(sampled["_target"].value_counts().max())

    parts = []
    for _, group in sampled.groupby("_target"):
        parts.append(group.sample(n=max_size, replace=True, random_state=random_state))

    balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=random_state)
    y_balanced = balanced.pop("_target").astype(int)
    return balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)


def build_rf_model(params: dict, random_state: int) -> RandomForestClassifier:
    model_params = {
        "n_estimators": 250,
        "bootstrap": True,
        "random_state": random_state,
        "n_jobs": 1,
        **params,
    }
    return RandomForestClassifier(**model_params)


def build_xgb_model(params: dict, y: pd.Series, random_state: int) -> xgb.XGBClassifier:
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0

    model_params = {
        "n_estimators": 250,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": 1,
        "scale_pos_weight": scale_pos_weight,
        **params,
    }
    return xgb.XGBClassifier(**model_params)


def fit_with_resample(
    x: pd.DataFrame,
    y: pd.Series,
    task_name: str,
    model_type: str,
    params: dict,
    repetition_id: int,
) -> pd.Series:
    use_upsample = task_name in {"model_3a", "model_3b"}
    random_state = 20240626 + repetition_id
    x_fit, y_fit = (random_upsample(x, y, random_state) if use_upsample else (x, y))

    if model_type == "rf":
        model = build_rf_model(params, random_state=random_state)
    else:
        model = build_xgb_model(params, y_fit, random_state=random_state)

    fitted = model.fit(x_fit, y_fit)
    return pd.Series(fitted.feature_importances_, index=x.columns)


def export_feature_importance(task_name: str, model_type: str, num_permutations: int = NUM_PERMUTATIONS) -> tuple[Path, Path, Path]:
    x, y = load_training_task(task_name, TASK_DATA_DIR)
    params = load_params(task_name, model_type)
    rows = []

    for repetition_id in range(num_permutations):
        rows.append(
            fit_with_resample(
                x=x,
                y=y,
                task_name=task_name,
                model_type=model_type,
                params=params,
                repetition_id=repetition_id,
            )
        )

    out = pd.DataFrame(rows)
    raw_output_path = RESULTS_DIR / f"{task_name}_{model_type}_feature_importance.csv"
    out.to_csv(raw_output_path, index=False)

    summary = pd.DataFrame(
        {
            "feature": out.columns,
            "importance_mean": out.mean(axis=0).to_numpy(),
            "importance_std": out.std(axis=0, ddof=1).fillna(0).to_numpy(),
        }
    ).sort_values("importance_mean", ascending=False)

    summary_output_path = RESULTS_DIR / f"{task_name}_{model_type}_top_features.csv"
    summary.to_csv(summary_output_path, index=False)

    top10_output_path = RESULTS_DIR / f"{task_name}_{model_type}_top10_features.csv"
    summary.head(TOP_N).to_csv(top10_output_path, index=False)

    return raw_output_path, summary_output_path, top10_output_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for task_name in TARGET_TASKS:
        for model_type in MODEL_TYPES:
            outputs.extend(export_feature_importance(task_name, model_type))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
