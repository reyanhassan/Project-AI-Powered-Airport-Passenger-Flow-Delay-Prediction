"""Streamlit dashboard for analytics, delay prediction, and simulation."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DELAY_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "airline_delay_sample.csv"
CLEAN_DELAY_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "airline_delay_cleaned.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_delay_model.joblib"


st.set_page_config(
    page_title="Airport Passenger Flow System",
    page_icon="airport",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with caching for faster dashboard refreshes."""

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def render_home_page() -> None:
    """Render the dashboard home page."""

    st.title("AI-Powered Airport Passenger Flow & Delay Prediction System")
    st.write(
        "A Python semester project combining SimPy passenger flow simulation, "
        "machine learning delay prediction, and interactive analytics."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Delay Records", len(load_csv(RAW_DELAY_DATA_PATH)))
    col2.metric("Trained Models", len(load_csv(MODEL_METRICS_PATH)))
    col3.metric("Best Model Saved", "Yes" if BEST_MODEL_PATH.exists() else "No")

    st.subheader("Project Modules")
    st.write(
        "- Simulation: passenger arrivals, check-in, security, gate, boarding, "
        "waiting times, and congestion statistics."
    )
    st.write(
        "- Machine Learning: preprocessing, categorical encoding, train/test "
        "split, model comparison, and best-model saving."
    )
    st.write(
        "- Dashboard: simulation analytics, delay prediction, and model "
        "performance views."
    )


def main() -> None:
    """Render the selected dashboard page."""

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Go to",
        ["Home"],
    )

    if selected_page == "Home":
        render_home_page()


if __name__ == "__main__":
    main()
