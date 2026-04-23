from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.modeling import load_artifacts
from src.scoring import score_portfolio
from src.training import train_and_persist

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DEMO_DATA_PATH = ROOT_DIR / "data" / "demo_loan_data.csv"

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Scores loan records and returns default probabilities plus risk segments.",
    version="1.0.0",
)


class ScoreRequest(BaseModel):
    records: list[dict[str, Any]]


@lru_cache(maxsize=1)
def get_model_bundle() -> tuple[object, dict[str, Any]]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        train_and_persist(dataset_path=DEMO_DATA_PATH, output_dir=ARTIFACTS_DIR)
    return load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    _, metadata = get_model_bundle()
    return {
        "status": "ok",
        "model": metadata.get("model_name"),
        "feature_count": len(metadata.get("feature_columns", [])),
    }


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    if not request.records:
        raise HTTPException(status_code=400, detail="records must contain at least one loan row.")

    model, metadata = get_model_bundle()
    frame = pd.DataFrame(request.records)
    scored = score_portfolio(frame, pipeline=model, feature_columns=metadata["feature_columns"])

    response_rows = scored[
        ["default_probability", "risk_segment", "predicted_default", "expected_loss"]
    ].to_dict(orient="records")
    return {
        "rows_scored": len(response_rows),
        "average_default_probability": float(scored["default_probability"].mean()),
        "predictions": response_rows,
    }
