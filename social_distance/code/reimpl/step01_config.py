from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
RESULTS_DIR = PROJECT_ROOT / "results"

TASK_DATA_DIR = DATA_DIR

TASK_RANDOM_STATE = 20240417
CV_RANDOM_STATE = 20240627

LABEL_MAP = {"No": 0, "Yes": 1}
