from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.data import load_or_create_dataset
from src.modeling import save_artifacts, train_and_select_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "demo_loan_data.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def train_and_persist(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    demo_rows: int = 5000,
    candidate_models: Iterable[str] = ("logistic", "xgboost"),
    random_state: int = 42,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACTS_DIR

    data = load_or_create_dataset(
        dataset_path=dataset_path,
        demo_rows=demo_rows,
        random_state=random_state,
    )

    result = train_and_select_model(
        raw_df=data,
        candidate_models=candidate_models,
        random_state=random_state,
    )

    metadata: dict[str, Any] = {
        "model_name": result["model_name"],
        "metrics": result["metrics"],
        "feature_columns": result["feature_columns"],
        "rows_used": result["rows_used"],
        "dataset_path": str(dataset_path),
        "candidate_models": list(candidate_models),
        "training_failures": result["failures"],
    }
    save_artifacts(result["pipeline"], metadata, output_dir=output_dir)
    return metadata
