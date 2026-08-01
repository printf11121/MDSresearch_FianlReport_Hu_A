from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reimpl.step06_xgboost_pipeline import run_xgboost_pipeline


if __name__ == "__main__":
    outputs = run_xgboost_pipeline()
    print(outputs)
