from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import prepare_features, prepare_portfolio_frame


def _build_estimator(model_name: str, random_state: int = 42) -> tuple[str, Any]:
    normalized = model_name.strip().lower()
    if normalized == "xgboost":
        try:
            from xgboost import XGBClassifier

            return (
                "xgboost",
                XGBClassifier(
                    n_estimators=280,
                    max_depth=5,
                    learning_rate=0.06,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=random_state,
                ),
            )
        except Exception:
            pass

    return (
        "logistic",
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )


def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X_train.columns if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No usable features were detected in the dataset.")

    return ColumnTransformer(transformers=transformers)


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_single_model(
    frame: pd.DataFrame, model_name: str, random_state: int = 42
) -> dict[str, Any]:
    if "target" not in frame.columns:
        raise ValueError("Target column is required for training.")

    training_frame = frame.dropna(subset=["target"]).copy()
    y = training_frame["target"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training requires at least two target classes.")

    X, feature_columns = prepare_features(training_frame, feature_columns=None)
    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = _build_preprocessor(X_train)
    resolved_name, estimator = _build_estimator(model_name=model_name, random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)
    metrics = _compute_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "model_name": resolved_name,
        "feature_columns": feature_columns,
        "rows_used": int(len(training_frame)),
    }


def train_and_select_model(
    raw_df: pd.DataFrame,
    candidate_models: Iterable[str] = ("logistic", "xgboost"),
    random_state: int = 42,
) -> dict[str, Any]:
    prepared = prepare_portfolio_frame(raw_df, random_state=random_state)
    if "target" not in prepared.columns:
        raise ValueError(
            "Could not derive a target column. Provide one of: target/default/loan_status."
        )

    best_result: dict[str, Any] | None = None
    failures: dict[str, str] = {}
    for candidate in candidate_models:
        try:
            result = train_single_model(prepared, model_name=candidate, random_state=random_state)
            if best_result is None or result["metrics"]["roc_auc"] > best_result["metrics"]["roc_auc"]:
                best_result = result
        except Exception as exc:
            failures[candidate] = str(exc)

    if best_result is None:
        raise RuntimeError(f"All candidate models failed: {failures}")

    best_result["prepared_frame"] = prepared
    best_result["failures"] = failures
    return best_result


def save_artifacts(pipeline: Pipeline, metadata: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_artifacts(model_path: Path, metadata_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return pipeline, metadata
