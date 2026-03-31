"""CLI script to run evaluation.

Usage:
    python scripts/evaluate.py                        # Full evaluation
    python scripts/evaluate.py --category concept     # Only concept questions
    python scripts/evaluate.py --output report.json   # Custom output path
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run CiteSage evaluation")
    parser.add_argument("--category", type=str, default=None,
                        choices=["concept", "factual", "discovery", "comparison", "literature_review"],
                        help="Only evaluate questions of this category")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("CiteSage — Evaluation Pipeline")
    print("=" * 60)

    run_evaluation(category_filter=args.category, output_path=args.output)


if __name__ == "__main__":
    main()
