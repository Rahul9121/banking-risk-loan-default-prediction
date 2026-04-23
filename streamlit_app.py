from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import create_demo_dataset
from src.modeling import load_artifacts
from src.scoring import build_delinquency_trend, build_risk_segmentation, score_portfolio
from src.training import train_and_persist

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DEMO_DATA_PATH = ROOT_DIR / "data" / "demo_loan_data.csv"


@st.cache_data(show_spinner=False)
def load_demo_portfolio(rows: int = 3500) -> pd.DataFrame:
    if DEMO_DATA_PATH.exists():
        return pd.read_csv(DEMO_DATA_PATH)

    DEMO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo_df = create_demo_dataset(n_rows=rows)
    demo_df.to_csv(DEMO_DATA_PATH, index=False)
    return demo_df


def ensure_artifacts() -> None:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        train_and_persist(dataset_path=DEMO_DATA_PATH, output_dir=ARTIFACTS_DIR)


def load_model_bundle() -> tuple[object, dict]:
    ensure_artifacts()
    return load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH)


st.set_page_config(page_title="Banking Risk & Loan Default Dashboard", layout="wide")
st.title("Banking Risk & Loan Default Prediction System")
st.caption(
    "Predicts loan default probability, visualizes portfolio risk distribution, and tracks delinquency trends."
)

with st.sidebar:
    st.header("Controls")
    data_mode = st.radio("Portfolio source", ["Demo portfolio", "Upload scoring CSV"], index=0)
    uploaded_scoring_file = None
    if data_mode == "Upload scoring CSV":
        uploaded_scoring_file = st.file_uploader("Upload portfolio CSV", type=["csv"])

    st.divider()
    st.subheader("Model")
    if st.button("Retrain model on demo data"):
        with st.spinner("Training model..."):
            train_and_persist(dataset_path=DEMO_DATA_PATH, output_dir=ARTIFACTS_DIR)
        st.success("Model retrained successfully.")

    uploaded_training_file = st.file_uploader(
        "Optional: upload labeled CSV to retrain", type=["csv"], key="training_csv"
    )
    if uploaded_training_file is not None and st.button("Train model from uploaded CSV"):
        training_df = pd.read_csv(uploaded_training_file)
        training_path = ROOT_DIR / "data" / "uploaded_training_data.csv"
        training_path.parent.mkdir(parents=True, exist_ok=True)
        training_df.to_csv(training_path, index=False)
        with st.spinner("Training model on uploaded dataset..."):
            train_and_persist(dataset_path=training_path, output_dir=ARTIFACTS_DIR)
        st.success("Training completed from uploaded dataset.")

if data_mode == "Upload scoring CSV":
    if uploaded_scoring_file is None:
        st.info("Upload a CSV from the sidebar to score your portfolio.")
        st.stop()
    portfolio_df = pd.read_csv(uploaded_scoring_file)
else:
    portfolio_df = load_demo_portfolio()

pipeline, metadata = load_model_bundle()
feature_columns = metadata.get("feature_columns", [])
scored_df = score_portfolio(portfolio_df, pipeline=pipeline, feature_columns=feature_columns)
trend_df = build_delinquency_trend(scored_df)
segment_df = build_risk_segmentation(scored_df)

total_loans = len(scored_df)
avg_pd = float(scored_df["default_probability"].mean())
high_risk_share = float(
    scored_df["risk_segment"].isin(["High", "Very High"]).mean() * 100
)
expected_loss = float(scored_df["expected_loss"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Loans", f"{total_loans:,}")
col2.metric("Average Default Probability", f"{avg_pd:.2%}")
col3.metric("High + Very High Risk", f"{high_risk_share:.1f}%")
col4.metric("Expected Loss Proxy", f"${expected_loss:,.0f}")

if "target" in scored_df.columns:
    observed_default = float(pd.to_numeric(scored_df["target"], errors="coerce").fillna(0).mean())
    st.caption(f"Observed default rate in dataset: **{observed_default:.2%}**")

hist_col, seg_col = st.columns(2)
with hist_col:
    fig_distribution = px.histogram(
        scored_df,
        x="default_probability",
        nbins=35,
        title="Portfolio Risk Distribution",
    )
    fig_distribution.update_layout(xaxis_title="Default Probability", yaxis_title="Loan Count")
    st.plotly_chart(fig_distribution, use_container_width=True)

with seg_col:
    fig_segments = px.bar(
        segment_df,
        x="risk_segment",
        y="loan_count",
        color="risk_segment",
        title="Risk Segmentation",
    )
    fig_segments.update_layout(showlegend=False, xaxis_title="Risk Segment", yaxis_title="Loans")
    st.plotly_chart(fig_segments, use_container_width=True)

trend_left, trend_right = st.columns(2)
with trend_left:
    fig_trend_pd = px.line(
        trend_df,
        x="month",
        y="avg_default_probability",
        markers=True,
        title="Delinquency Trend: Avg Default Probability",
    )
    fig_trend_pd.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig_trend_pd, use_container_width=True)

with trend_right:
    fig_trend_dpd = px.line(
        trend_df,
        x="month",
        y="avg_days_past_due",
        markers=True,
        title="Delinquency Trend: Avg Days Past Due",
    )
    st.plotly_chart(fig_trend_dpd, use_container_width=True)

st.subheader("Top Risky Loans")
top_risky = scored_df.sort_values("default_probability", ascending=False).head(30)
display_columns = [
    column
    for column in [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "fico_score",
        "days_past_due",
        "default_probability",
        "risk_segment",
        "expected_loss",
    ]
    if column in top_risky.columns
]
st.dataframe(top_risky[display_columns], use_container_width=True)

csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Scored Portfolio CSV",
    data=csv_bytes,
    file_name="scored_portfolio.csv",
    mime="text/csv",
)

with st.expander("Model details"):
    st.json(
        {
            "selected_model": metadata.get("model_name"),
            "metrics": metadata.get("metrics", {}),
            "feature_count": len(metadata.get("feature_columns", [])),
            "training_rows": metadata.get("rows_used"),
            "training_dataset": metadata.get("dataset_path"),
            "candidate_models": metadata.get("candidate_models"),
            "training_failures": metadata.get("training_failures"),
        }
    )
