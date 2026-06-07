"""Streamlit dashboard for analytics, delay prediction, and simulation."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation import (
    SimulationConfig,
    calculate_statistics,
    passengers_to_dataframe,
    run_simulation,
)

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


@st.cache_data
def run_cached_simulation(
    passengers: int,
    arrival_interval: float,
    check_in_counters: int,
    security_lanes: int,
    boarding_gates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the SimPy simulation and return table-ready results."""

    config = SimulationConfig(
        num_passengers=passengers,
        average_arrival_interval=arrival_interval,
        check_in_counters=check_in_counters,
        security_lanes=security_lanes,
        boarding_gates=boarding_gates,
        random_seed=random_seed,
    )
    simulated_passengers = run_simulation(config=config)
    passenger_data = passengers_to_dataframe(simulated_passengers)
    statistics = calculate_statistics(simulated_passengers)

    return passenger_data, statistics


def render_simulation_analytics_page() -> None:
    """Render passenger flow simulation analytics."""

    st.title("Simulation Analytics")

    control_col1, control_col2, control_col3 = st.columns(3)
    passengers = control_col1.slider("Passengers", 10, 300, 80, 10)
    arrival_interval = control_col1.slider("Arrival Interval", 0.5, 8.0, 2.0, 0.5)
    random_seed = control_col1.number_input("Random Seed", min_value=1, value=42)

    check_in_counters = control_col2.slider("Check-in Counters", 1, 10, 3)
    security_lanes = control_col2.slider("Security Lanes", 1, 10, 2)
    boarding_gates = control_col3.slider("Boarding Gates", 1, 8, 1)

    passenger_data, statistics = run_cached_simulation(
        passengers,
        arrival_interval,
        check_in_counters,
        security_lanes,
        boarding_gates,
        int(random_seed),
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total Passengers", statistics["total_passengers"])
    metric_col2.metric("Average Total Wait", f"{statistics['average_total_wait']} min")
    metric_col3.metric("Busiest Stage", statistics["busiest_stage"])
    metric_col4.metric("Congestion", statistics["congestion_level"])

    wait_chart_data = pd.DataFrame(
        {
            "stage": ["Check-in", "Security", "Boarding"],
            "average_wait_minutes": [
                statistics["average_check_in_wait"],
                statistics["average_security_wait"],
                statistics["average_boarding_wait"],
            ],
        }
    ).set_index("stage")

    chart_col1, chart_col2 = st.columns(2)
    chart_col1.subheader("Average Wait by Stage")
    chart_col1.bar_chart(wait_chart_data)

    chart_col2.subheader("Journey Time by Passenger")
    chart_col2.line_chart(passenger_data.set_index("passenger_id")["journey_time"])

    st.subheader("Passenger Simulation Records")
    st.dataframe(passenger_data, use_container_width=True)


def main() -> None:
    """Render the selected dashboard page."""

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Go to",
        ["Home", "Simulation Analytics"],
    )

    if selected_page == "Home":
        render_home_page()
    elif selected_page == "Simulation Analytics":
        render_simulation_analytics_page()


if __name__ == "__main__":
    main()
