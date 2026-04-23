from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import prepare_features, prepare_portfolio_frame

RISK_BINS = [-0.001, 0.10, 0.25, 0.40, 1.00]
RISK_LABELS = ["Low", "Moderate", "High", "Very High"]


def score_portfolio(
    raw_df: pd.DataFrame, pipeline: Pipeline, feature_columns: list[str]
) -> pd.DataFrame:
    portfolio = prepare_portfolio_frame(raw_df)
    X, _ = prepare_features(portfolio, feature_columns=feature_columns)

    probabilities = pipeline.predict_proba(X)[:, 1]
    scored = portfolio.copy()
    scored["default_probability"] = np.round(probabilities, 4)
    scored["predicted_default"] = (scored["default_probability"] >= 0.50).astype(int)
    scored["risk_segment"] = pd.cut(
        scored["default_probability"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True,
    ).astype(str)

    loan_amount = pd.to_numeric(scored.get("loan_amnt", 0), errors="coerce").fillna(0.0)
    scored["expected_loss"] = np.round(scored["default_probability"] * loan_amount, 2)
    return scored


def build_delinquency_trend(scored: pd.DataFrame) -> pd.DataFrame:
    trend_source = scored.copy()
    trend_source["issue_date"] = pd.to_datetime(trend_source["issue_date"], errors="coerce")
    trend_source = trend_source.dropna(subset=["issue_date"]).copy()
    trend_source["month"] = trend_source["issue_date"].dt.to_period("M").dt.to_timestamp()

    aggregations = {
        "default_probability": "mean",
        "days_past_due": "mean",
        "loan_amnt": "sum",
        "predicted_default": "mean",
    }
    if "target" in trend_source.columns:
        aggregations["target"] = "mean"

    trend = trend_source.groupby("month", as_index=False).agg(aggregations)
    trend = trend.rename(
        columns={
            "default_probability": "avg_default_probability",
            "days_past_due": "avg_days_past_due",
            "loan_amnt": "portfolio_exposure",
            "predicted_default": "predicted_default_rate",
            "target": "observed_default_rate",
        }
    )
    return trend.sort_values("month")


def build_risk_segmentation(scored: pd.DataFrame) -> pd.DataFrame:
    loan_amount = pd.to_numeric(scored.get("loan_amnt", 0), errors="coerce").fillna(0.0)
    segment_source = scored.copy()
    segment_source["loan_amount_clean"] = loan_amount

    segmentation = (
        segment_source.groupby("risk_segment", as_index=False)
        .agg(
            loan_count=("risk_segment", "size"),
            exposure=("loan_amount_clean", "sum"),
            avg_pd=("default_probability", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .sort_values("avg_pd")
    )
    return segmentation
