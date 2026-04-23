from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import train_and_persist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train credit-risk model and persist artifacts."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to CSV dataset. If missing, demo data is generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory to save model.joblib and metadata.json.",
    )
    parser.add_argument(
        "--demo-rows",
        type=int,
        default=5000,
        help="Rows to generate for demo dataset when dataset path does not exist.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "xgboost"],
        help="Candidate models to evaluate. Supported: logistic, xgboost.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else None
    output_dir = Path(args.output_dir)

    metadata = train_and_persist(
        dataset_path=dataset_path,
        output_dir=output_dir,
        demo_rows=args.demo_rows,
        candidate_models=args.models,
    )

    print("Training complete.")
    print(f"Selected model: {metadata['model_name']}")
    print("Metrics:")
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
