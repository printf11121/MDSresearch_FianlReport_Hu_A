from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reimpl.step04_logistic_baseline import run_logistic_baseline


if __name__ == "__main__":
    outputs = run_logistic_baseline()
    print(outputs)
