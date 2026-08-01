from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reimpl.step07_random_forest import run_random_forest_pipeline


if __name__ == "__main__":
    outputs = run_random_forest_pipeline()
    print(outputs)
