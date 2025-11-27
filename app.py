"""Streamlit app for behavior shift detection and explainable anomaly detection.

The app walks through:
1. Data aggregation from transaction and login logs into daily behavior vectors.
2. Feature engineering with rolling windows, ratios, volatility, and device/IP stats.
3. Baseline modeling to learn user behavioral fingerprints (mean + covariance).
4. Behavior shift detection via Mahalanobis Distance.
5. Anomaly detection via Isolation Forest.
6. Risk scoring that blends shift and anomaly signals.
7. Explainability with SHAP values.
8. Visualization in an interactive dashboard.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Behavior Shift Detection", layout="wide")

# -----------------------------
# Data schema
# -----------------------------


@dataclass
class TransactionRecord:
    user_id: str
    event_time: dt.datetime
    amount: float
    device_id: str
    ip: str
    channel: str


@dataclass
class LoginRecord:
    user_id: str
    event_time: dt.datetime
    success: bool
    device_id: str
    ip: str


# -----------------------------
# Synthetic data for demo
# -----------------------------


def _random_choice(series: List[str], size: int) -> List[str]:
    rng = np.random.default_rng(42)
    return rng.choice(series, size=size).tolist()


def generate_synthetic_logs(n_users: int = 8, days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible synthetic transaction and login logs."""
    rng = np.random.default_rng(0)
    start = dt.datetime.now() - dt.timedelta(days=days)
    user_ids = [f"user_{i+1}" for i in range(n_users)]

    transactions: List[TransactionRecord] = []
    logins: List[LoginRecord] = []

    for user in user_ids:
        base_amount = rng.uniform(30, 200)
        for day in range(days):
            date = start + dt.timedelta(days=day)
            # Normal behavior
            txn_count = rng.poisson(3)
            login_count = rng.poisson(2) + 1
            for _ in range(txn_count):
                amount = np.maximum(rng.normal(base_amount, base_amount * 0.3), 1)
                transactions.append(
                    TransactionRecord(
                        user,
                        date + dt.timedelta(minutes=int(rng.integers(0, 24 * 60))),
                        float(amount),
                        device_id=f"device_{rng.integers(1, 4)}",
                        ip=f"10.0.{rng.integers(0, 5)}.{rng.integers(1, 255)}",
                        channel=rng.choice(["web", "mobile"]),
                    )
                )
            for _ in range(login_count):
                logins.append(
                    LoginRecord(
                        user,
                        date + dt.timedelta(minutes=int(rng.integers(0, 24 * 60))),
                        bool(rng.choice([True] * 9 + [False])),
                        device_id=f"device_{rng.integers(1, 5)}",
                        ip=f"10.0.{rng.integers(0, 5)}.{rng.integers(1, 255)}",
                    )
                )

        # Inject behavior shift in the last 10 days for half the users
        if int(user.split("_")[-1]) % 2 == 0:
            for day in range(days - 10, days):
                date = start + dt.timedelta(days=day)
                for _ in range(rng.poisson(6)):
                    amount = np.maximum(rng.normal(base_amount * 2.5, base_amount), 5)
                    transactions.append(
                        TransactionRecord(
                            user,
                            date + dt.timedelta(minutes=int(rng.integers(0, 24 * 60))),
                            float(amount),
                            device_id=f"new_device_{rng.integers(5, 8)}",
                            ip=f"172.16.{rng.integers(0, 5)}.{rng.integers(1, 255)}",
                            channel=rng.choice(["web", "mobile"]),
                        )
                    )
                for _ in range(rng.poisson(3)):
                    logins.append(
                        LoginRecord(
                            user,
                            date + dt.timedelta(minutes=int(rng.integers(0, 24 * 60))),
                            bool(rng.choice([True] * 6 + [False] * 4)),
                            device_id=f"new_device_{rng.integers(5, 8)}",
                            ip=f"172.16.{rng.integers(0, 5)}.{rng.integers(1, 255)}",
                        )
                    )

    txn_df = pd.DataFrame([t.__dict__ for t in transactions])
    login_df = pd.DataFrame([l.__dict__ for l in logins])
    return txn_df, login_df


# -----------------------------
# Feature engineering
# -----------------------------


def aggregate_daily(txn_df: pd.DataFrame, login_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw logs into daily behavioral vectors per user."""
    txn_df = txn_df.copy()
    login_df = login_df.copy()
    txn_df["event_date"] = pd.to_datetime(txn_df["event_time"]).dt.date
    login_df["event_date"] = pd.to_datetime(login_df["event_time"]).dt.date

    txn_daily = (
        txn_df.groupby(["user_id", "event_date"]).agg(
            txn_count=("amount", "count"),
            txn_total=("amount", "sum"),
            txn_avg=("amount", "mean"),
            txn_std=("amount", "std"),
            txn_max=("amount", "max"),
            device_unique=("device_id", "nunique"),
            ip_unique=("ip", "nunique"),
        )
    ).reset_index()

    login_daily = (
        login_df.groupby(["user_id", "event_date"]).agg(
            login_count=("success", "count"),
            login_success=("success", "sum"),
            login_fail=("success", lambda s: (~s).sum()),
            login_device_unique=("device_id", "nunique"),
            login_ip_unique=("ip", "nunique"),
        )
    ).reset_index()

    daily = pd.merge(txn_daily, login_daily, on=["user_id", "event_date"], how="outer").fillna(0)
    daily["event_date"] = pd.to_datetime(daily["event_date"])
    daily = daily.sort_values(["user_id", "event_date"])
    daily["txn_std"] = daily["txn_std"].fillna(0)
    daily["txn_avg"] = daily["txn_avg"].fillna(0)
    return daily


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling windows, ratios, and volatility features."""
    feats = daily.copy()
    feats = feats.set_index("event_date")
    engineered: List[pd.DataFrame] = []

    for user, group in feats.groupby("user_id"):
        g = group.sort_index().copy()
        # Rolling statistics
        for window in [7, 30]:
            g[f"txn_total_roll_{window}"] = g["txn_total"].rolling(window, min_periods=3).mean()
            g[f"txn_avg_roll_{window}"] = g["txn_avg"].rolling(window, min_periods=3).mean()
            g[f"login_count_roll_{window}"] = g["login_count"].rolling(window, min_periods=3).mean()

        # Ratios and volatility
        g["success_rate"] = g["login_success"].div(g["login_count"].replace({0: np.nan}))
        g["success_rate"] = g["success_rate"].fillna(1.0)
        g["fail_ratio"] = g["login_fail"].div(g["login_count"].replace({0: np.nan})).fillna(0)
        g["txn_volatility"] = g["txn_std"].fillna(0) / (g["txn_avg"].replace({0: np.nan}))
        g["txn_volatility"] = g["txn_volatility"].replace([np.inf, -np.inf], 0).fillna(0)

        g = g.reset_index()
        engineered.append(g)

    features = pd.concat(engineered).sort_values(["user_id", "event_date"])
    features = features.fillna(0)
    return features


# -----------------------------
# Modeling
# -----------------------------


def baseline_fingerprint(features: pd.DataFrame, baseline_days: int = 30) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute mean and covariance fingerprint per user using the earliest baseline_days."""
    fingerprints: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    for user, group in features.groupby("user_id"):
        baseline = group.sort_values("event_date").head(baseline_days)
        matrix = baseline[numeric_cols].to_numpy()
        mean = matrix.mean(axis=0)
        cov = np.cov(matrix, rowvar=False)
        # Regularize covariance to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-3
        fingerprints[user] = (mean, cov)
    return fingerprints


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    return float(np.sqrt(diff.T @ inv_cov @ diff))


def compute_shift_scores(features: pd.DataFrame, fingerprints: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.Series:
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    scores: List[float] = []
    for _, row in features.iterrows():
        mean, cov = fingerprints[row.user_id]
        x = row[numeric_cols].to_numpy()
        scores.append(mahalanobis_distance(x, mean, cov))
    return pd.Series(scores, index=features.index, name="shift_score")


@dataclass
class DetectionResult:
    features: pd.DataFrame
    shift_scores: pd.Series
    anomaly_scores: pd.Series
    risk: pd.Series
    model: IsolationForest
    shap_values: np.ndarray
    explainer: shap.TreeExplainer


@lru_cache(maxsize=8)
def run_pipeline(txn_df: pd.DataFrame, login_df: pd.DataFrame) -> DetectionResult:
    daily = aggregate_daily(txn_df, login_df)
    features = build_features(daily)
    numeric_cols = features.select_dtypes(include=[np.number]).columns

    fingerprints = baseline_fingerprint(features)
    shift_scores = compute_shift_scores(features, fingerprints)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features[numeric_cols])

    iso = IsolationForest(random_state=42, contamination=0.05)
    iso.fit(scaled)
    # Higher score => more normal; invert to get anomaly magnitude
    raw_anomaly = -iso.decision_function(scaled)
    anomaly_scores = pd.Series(raw_anomaly, index=features.index, name="anomaly_score")

    # Blend shift and anomaly into a risk percentile
    blended = zscore(shift_scores.fillna(0)) + zscore(anomaly_scores.fillna(0))
    risk = pd.Series(MinMaxScaler().fit_transform(blended.to_frame())[:, 0], index=features.index, name="risk_score")

    explainer = shap.TreeExplainer(iso)
    shap_values = explainer.shap_values(scaled)

    enriched = features.copy()
    enriched["shift_score"] = shift_scores
    enriched["anomaly_score"] = anomaly_scores
    enriched["risk_score"] = risk

    return DetectionResult(enriched, shift_scores, anomaly_scores, risk, iso, shap_values, explainer)


# -----------------------------
# Streamlit layout
# -----------------------------


def sidebar_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Data Input")
    st.sidebar.write("Upload CSVs or use generated demo data.")

    txn_file = st.sidebar.file_uploader("Transaction log CSV", type=["csv"])
    login_file = st.sidebar.file_uploader("Login log CSV", type=["csv"])
    use_demo = st.sidebar.checkbox("Use synthetic demo data", value=True)

    if use_demo or not (txn_file and login_file):
        st.sidebar.info("Using synthetic logs (toggle off to upload your own).")
        txn_df, login_df = generate_synthetic_logs()
    else:
        txn_df = pd.read_csv(txn_file, parse_dates=["event_time"])
        login_df = pd.read_csv(login_file, parse_dates=["event_time"])
    return txn_df, login_df


def render_overview(result: DetectionResult):
    st.subheader("Pipeline Outputs")
    st.caption("Rolling features → fingerprints → Mahalanobis shift → Isolation Forest → Risk score")

    latest = result.features.sort_values("event_date").groupby("user_id").tail(1)
    st.metric("Users analyzed", len(latest))

    cols = st.columns(2)
    with cols[0]:
        st.write("Top risk users (latest day)")
        st.dataframe(
            latest.sort_values("risk_score", ascending=False)[
                ["user_id", "event_date", "risk_score", "shift_score", "anomaly_score", "txn_total", "login_count"]
            ].style.format({"risk_score": "{:.2f}", "shift_score": "{:.2f}", "anomaly_score": "{:.2f}"})
        )
    with cols[1]:
        st.write("Daily risk timeline")
        st.line_chart(result.features.set_index("event_date")["risk_score"])


def render_user_detail(result: DetectionResult):
    st.subheader("User Drilldown")
    users = result.features["user_id"].unique().tolist()
    user_id = st.selectbox("Choose user", users)
    user_df = result.features[result.features["user_id"] == user_id].set_index("event_date")

    st.write("Behavioral trajectory")
    st.line_chart(user_df[["txn_total", "login_count", "risk_score", "shift_score"]])

    st.write("Explainability (SHAP) on Isolation Forest")
    shap_fig = shap.force_plot(
        result.explainer.expected_value,
        result.shap_values[user_df.index],
        user_df[result.model.feature_names_in_],
        matplotlib=True,
        show=False,
    )
    st.pyplot(shap_fig, clear_figure=True)

    st.write("Feature distribution (recent 14 days)")
    recent = user_df.tail(14)
    st.bar_chart(recent[["txn_total", "txn_volatility", "fail_ratio", "device_unique", "ip_unique"]])


# -----------------------------
# Main app
# -----------------------------


def main():
    st.title("Behavior Shift Detection + Explainable AI for Online Banking")
    st.write(
        "This dashboard aggregates transaction/login logs, engineers behavioral features,"
        " builds a fingerprint baseline, detects shifts with Mahalanobis distance, and"
        " flags anomalies with Isolation Forest. Risk blends both signals and SHAP explains why."
    )

    txn_df, login_df = sidebar_inputs()
    st.write("Transaction sample", txn_df.head())
    st.write("Login sample", login_df.head())

    with st.spinner("Running detection pipeline"):
        result = run_pipeline(txn_df, login_df)

    render_overview(result)
    st.divider()
    render_user_detail(result)


if __name__ == "__main__":
    main()
