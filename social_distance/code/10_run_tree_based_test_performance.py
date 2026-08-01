from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reimpl.step08_tree_based_test_performance import run_tree_based_test_performance


if __name__ == "__main__":
    output_path = run_tree_based_test_performance()
    print(output_path)
