from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

TARGET_CANDIDATES: Sequence[str] = (
    "target",
    "default",
    "loan_default",
    "is_default",
    "bad_loan",
)

STATUS_COLUMN_CANDIDATES: Sequence[str] = (
    "loan_status",
    "status",
)

DATE_CANDIDATES: Sequence[str] = (
    "issue_date",
    "issue_d",
    "application_date",
    "date",
    "month",
)

ALIAS_MAP: dict[str, list[str]] = {
    "loan_amnt": ["loan_amount", "amt_credit", "principal"],
    "annual_inc": ["annual_income", "income", "amt_income_total"],
    "int_rate": ["interest_rate"],
    "fico_score": ["fico", "credit_score"],
    "dti": ["debt_to_income", "dti_ratio"],
}

NON_FEATURE_COLUMNS: set[str] = {
    "target",
    "loan_status",
    "status",
    "issue_date",
    "id",
    "member_id",
    "predicted_default",
    "default_probability",
    "risk_segment",
    "expected_loss",
}

DEFAULT_KEYWORDS: tuple[str, ...] = (
    "charged off",
    "default",
    "late (31-120",
    "late (16-30",
    "does not meet the credit policy. status:charged off",
)

POSITIVE_BINARY_VALUES: set[str] = {"1", "true", "yes", "y", "default", "bad"}
NEGATIVE_BINARY_VALUES: set[str] = {"0", "false", "no", "n", "paid", "good"}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in frame.columns
    ]
    return frame


def apply_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    for canonical, aliases in ALIAS_MAP.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            if alias in frame.columns:
                frame[canonical] = frame[alias]
                break
    return frame


def _as_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce") > 0).astype("Int64")

    normalized = series.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.nan, index=series.index, dtype="float64")
    mapped[normalized.isin(POSITIVE_BINARY_VALUES)] = 1
    mapped[normalized.isin(NEGATIVE_BINARY_VALUES)] = 0
    return mapped.astype("Int64")


def derive_target(frame: pd.DataFrame) -> pd.Series | None:
    for column in TARGET_CANDIDATES:
        if column in frame.columns:
            target = _as_binary(frame[column])
            if target.notna().any():
                return target.fillna(0).astype(int)

    for status_column in STATUS_COLUMN_CANDIDATES:
        if status_column in frame.columns:
            status = frame[status_column].astype(str).str.strip().str.lower()
            default_mask = status.apply(
                lambda value: any(keyword in value for keyword in DEFAULT_KEYWORDS)
            )
            return default_mask.astype(int)

    if "days_past_due" in frame.columns:
        dpd = pd.to_numeric(frame["days_past_due"], errors="coerce").fillna(0)
        return (dpd > 30).astype(int)
    return None


def _ensure_issue_date(frame: pd.DataFrame, random_state: int = 42) -> pd.Series:
    for column in DATE_CANDIDATES:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                return parsed.fillna(parsed.median())

    rng = np.random.default_rng(random_state)
    date_pool = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
    sampled = rng.choice(date_pool, size=len(frame), replace=True)
    return pd.to_datetime(sampled)


def _ensure_days_past_due(
    frame: pd.DataFrame, target: pd.Series | None, random_state: int = 42
) -> pd.Series:
    if "days_past_due" in frame.columns:
        numeric = pd.to_numeric(frame["days_past_due"], errors="coerce")
        return numeric.fillna(0).clip(lower=0)

    rng = np.random.default_rng(random_state)
    if target is None:
        synthetic = rng.gamma(shape=2.0, scale=5.0, size=len(frame))
        return pd.Series(np.round(synthetic), index=frame.index).astype(int)

    default_component = rng.gamma(shape=4.5, scale=8.0, size=len(frame))
    normal_component = rng.gamma(shape=1.4, scale=2.4, size=len(frame))
    simulated = np.where(target.to_numpy() == 1, default_component, normal_component)
    return pd.Series(np.round(simulated), index=frame.index).astype(int)


def _try_parse_numeric_objects(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_object_dtype(output[column]):
            stripped = (
                output[column]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            parsed = pd.to_numeric(stripped, errors="coerce")
            if parsed.notna().mean() >= 0.80:
                output[column] = parsed
    return output


def prepare_portfolio_frame(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    frame = standardize_columns(df)
    frame = apply_aliases(frame)
    frame = _try_parse_numeric_objects(frame)

    target = derive_target(frame)
    if target is not None:
        frame["target"] = target

    frame["issue_date"] = _ensure_issue_date(frame, random_state=random_state)
    frame["days_past_due"] = _ensure_days_past_due(
        frame, target=target, random_state=random_state
    )

    if "loan_amnt" not in frame.columns:
        frame["loan_amnt"] = np.nan
    return frame


def prepare_features(
    frame: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    prepared = frame.copy()
    if feature_columns is None:
        feature_columns = [
            column for column in prepared.columns if column not in NON_FEATURE_COLUMNS
        ]

    X = prepared.reindex(columns=feature_columns).copy()
    for column in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            X[column] = X[column].astype("int64") // 10**9
        elif pd.api.types.is_bool_dtype(X[column]):
            X[column] = X[column].astype(int)
    return X, feature_columns


def create_demo_dataset(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    loan_amnt = rng.integers(2_000, 45_000, size=n_rows)
    term_months = rng.choice([36, 60], size=n_rows, p=[0.70, 0.30])
    int_rate = np.round(rng.normal(loc=12.2, scale=4.0, size=n_rows).clip(5.0, 30.0), 2)
    annual_inc = np.round(
        rng.lognormal(mean=np.log(65_000), sigma=0.55, size=n_rows).clip(18_000, 320_000),
        2,
    )
    dti = np.round(rng.normal(loc=18.0, scale=8.0, size=n_rows).clip(0.0, 45.0), 2)
    fico_score = np.round(rng.normal(loc=690, scale=55, size=n_rows).clip(540, 830))
    revol_util = np.round(rng.normal(loc=52.0, scale=20.0, size=n_rows).clip(0.0, 100.0), 2)
    delinq_2yrs = rng.poisson(lam=0.35, size=n_rows)
    open_acc = rng.integers(2, 25, size=n_rows)
    total_acc = open_acc + rng.integers(3, 35, size=n_rows)
    emp_length = rng.integers(0, 11, size=n_rows)
    issue_date = rng.choice(
        pd.date_range("2021-01-01", "2025-12-01", freq="MS"), size=n_rows, replace=True
    )

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"], size=n_rows, p=[0.43, 0.44, 0.11, 0.02]
    )
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business", "car"],
        size=n_rows,
        p=[0.40, 0.28, 0.15, 0.09, 0.08],
    )
    verification_status = rng.choice(
        ["Verified", "Source Verified", "Not Verified"], size=n_rows, p=[0.34, 0.31, 0.35]
    )
    grade = rng.choice(list("ABCDEFG"), size=n_rows, p=[0.17, 0.20, 0.20, 0.18, 0.13, 0.08, 0.04])

    risk_signal = (
        -4.2
        + 0.07 * (int_rate - 12)
        + 0.06 * (dti - 18)
        + 0.85 * (term_months == 60)
        + 0.35 * delinq_2yrs
        + 0.75 * (fico_score < 640)
        + 0.45 * (revol_util > 80)
        - 0.000006 * (annual_inc - 65_000)
        - 0.03 * emp_length
    )
    default_probability = 1 / (1 + np.exp(-risk_signal))
    target = rng.binomial(1, default_probability)

    default_statuses = np.array(["Charged Off", "Default", "Late (31-120 days)"])
    normal_statuses = np.array(["Fully Paid", "Current"])
    loan_status = np.where(
        target == 1,
        rng.choice(default_statuses, size=n_rows, p=[0.60, 0.24, 0.16]),
        rng.choice(normal_statuses, size=n_rows, p=[0.68, 0.32]),
    )

    default_dpd = rng.gamma(shape=4.8, scale=9.0, size=n_rows)
    normal_dpd = rng.gamma(shape=1.3, scale=2.6, size=n_rows)
    days_past_due = np.round(np.where(target == 1, default_dpd, normal_dpd)).astype(int)

    current_balance = np.round(loan_amnt * rng.uniform(0.2, 1.0, size=n_rows), 2)

    return pd.DataFrame(
        {
            "loan_amnt": loan_amnt,
            "term_months": term_months,
            "int_rate": int_rate,
            "annual_inc": annual_inc,
            "dti": dti,
            "fico_score": fico_score,
            "revol_util": revol_util,
            "delinq_2yrs": delinq_2yrs,
            "open_acc": open_acc,
            "total_acc": total_acc,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "purpose": purpose,
            "verification_status": verification_status,
            "grade": grade,
            "issue_date": pd.to_datetime(issue_date),
            "days_past_due": days_past_due,
            "loan_status": loan_status,
            "target": target,
            "current_balance": current_balance,
        }
    )


def load_or_create_dataset(
    dataset_path: Path, demo_rows: int = 5000, random_state: int = 42
) -> pd.DataFrame:
    if dataset_path.exists():
        return pd.read_csv(dataset_path)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    generated = create_demo_dataset(n_rows=demo_rows, random_state=random_state)
    generated.to_csv(dataset_path, index=False)
    return generated
