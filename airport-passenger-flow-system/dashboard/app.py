"""Streamlit dashboard for analytics, delay prediction, and simulation."""

import html
from io import BytesIO
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from advanced_airport import render_advanced_animated_airport_page
from professional_control_center import render_professional_airport_control_center_page

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation import (
    SimulationConfig,
    build_live_simulation_data,
    calculate_statistics,
    passengers_to_dataframe,
    run_simulation,
)
from ml.preprocessing import clean_delay_data, load_delay_dataset

RAW_DELAY_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "airline_delay_realistic.csv"
CLEAN_DELAY_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "airline_delay_cleaned.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_delay_model.joblib"
MODEL_CHART_PNG_PATH = PROJECT_ROOT / "reports" / "model_metrics_comparison.png"
MODEL_CHART_HTML_PATH = PROJECT_ROOT / "reports" / "model_metrics_comparison.html"


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


@st.cache_data
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame."""

    return pd.read_csv(BytesIO(file_bytes))


@st.cache_resource
def load_best_model():
    """Load the saved delay prediction model once per dashboard session."""

    if BEST_MODEL_PATH.exists():
        return joblib.load(BEST_MODEL_PATH)

    return None


@st.cache_data
def load_clean_delay_data() -> pd.DataFrame:
    """Load cleaned delay data for dashboard filters and select boxes."""

    if CLEAN_DELAY_DATA_PATH.exists():
        return pd.read_csv(CLEAN_DELAY_DATA_PATH)

    return clean_delay_data(load_delay_dataset(RAW_DELAY_DATA_PATH))


@st.cache_data
def load_live_airport_data(
    passengers: int,
    arrival_interval: float,
    random_seed: int,
) -> dict[str, object]:
    """Run a dashboard-sized live simulation and cache the results."""

    config = SimulationConfig(
        num_passengers=passengers,
        average_arrival_interval=arrival_interval,
        check_in_counters=3,
        security_lanes=2,
        boarding_gates=2,
        random_seed=random_seed,
    )
    return build_live_simulation_data(config=config, frame_count=28)


def render_airport_dashboard_css() -> None:
    """Apply airport-control-screen styling to the live simulation page."""

    st.markdown(
        """
        <style>
        .airport-screen {
            background: #071018;
            border: 1px solid #1d3b53;
            border-radius: 8px;
            padding: 18px;
            color: #eef7ff;
            box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.15);
        }
        .airport-title {
            color: #8fd3ff;
            font-size: 34px;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 6px;
        }
        .airport-subtitle {
            color: #93a4b7;
            font-size: 15px;
            margin-bottom: 18px;
        }
        .airport-zone {
            background: #0d1b26;
            border: 1px solid #21415c;
            border-radius: 8px;
            padding: 14px;
            min-height: 110px;
        }
        .airport-zone h4 {
            color: #cbeaff;
            font-size: 17px;
            margin: 0 0 10px 0;
            letter-spacing: 0;
        }
        .airport-zone-value {
            color: #ffffff;
            font-size: 30px;
            font-weight: 800;
            line-height: 1.15;
        }
        .airport-zone-label {
            color: #9eb1c5;
            font-size: 13px;
            margin-top: 4px;
        }
        .airport-card {
            background: #0b1822;
            border: 1px solid #23455f;
            border-radius: 8px;
            padding: 14px;
            min-height: 156px;
        }
        .airport-card-title {
            color: #d8f1ff;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .airport-card-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            color: #b9c7d6;
            font-size: 13px;
            margin: 7px 0;
        }
        .airport-card-value {
            color: #ffffff;
            font-weight: 700;
            text-align: right;
        }
        .status-badge {
            display: inline-block;
            border-radius: 6px;
            padding: 4px 9px;
            color: #06101a;
            font-size: 12px;
            font-weight: 800;
            margin-top: 8px;
        }
        .status-open {
            background: #47d16c;
        }
        .status-busy {
            background: #f5b84b;
        }
        .flight-board {
            width: 100%;
            border-collapse: collapse;
            background: #071018;
            color: #edf8ff;
            border: 1px solid #244861;
            border-radius: 8px;
            overflow: hidden;
            font-size: 14px;
        }
        .flight-board th {
            background: #10283a;
            color: #8fd3ff;
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #244861;
        }
        .flight-board td {
            padding: 10px;
            border-bottom: 1px solid #152c3d;
        }
        .flight-board tr:last-child td {
            border-bottom: none;
        }
        .flight-status {
            display: inline-block;
            border-radius: 6px;
            padding: 4px 8px;
            font-weight: 800;
            min-width: 78px;
            text-align: center;
        }
        .flight-on-time {
            background: #1f9d55;
            color: #ffffff;
        }
        .flight-delayed {
            background: #d64545;
            color: #ffffff;
        }
        .flight-boarding {
            background: #2f80ed;
            color: #ffffff;
        }
        .flight-gate-closed {
            background: #f5b84b;
            color: #06101a;
        }
        .flight-departed {
            background: #73808c;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_dashboard_delay_data(uploaded_file) -> tuple[pd.DataFrame, str, bool, str]:
    """Return the active dataset for analytics and prediction pages.

    If the uploaded file follows the expected airline-delay columns, it is
    cleaned and used by the delay pages. Otherwise, the dashboard still shows a
    generic preview of the uploaded CSV.
    """

    if uploaded_file is None:
        return load_clean_delay_data(), "Sample airline delay dataset", True, ""

    try:
        uploaded_data = read_uploaded_csv(uploaded_file.getvalue())
    except Exception as error:
        return pd.DataFrame(), uploaded_file.name, False, f"Could not read CSV: {error}"

    try:
        cleaned_data = clean_delay_data(uploaded_data)
        return cleaned_data, uploaded_file.name, True, ""
    except ValueError as error:
        return uploaded_data, uploaded_file.name, False, str(error)


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
    st.write(
        "- Uploads: use the sidebar CSV uploader to inspect your own dataset. "
        "Files with the airline-delay columns also power the analytics and "
        "prediction pages."
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


def render_generic_dataset_summary(
    dataset: pd.DataFrame,
    dataset_name: str,
    schema_message: str,
) -> None:
    """Render a generic summary when an uploaded CSV is not a delay dataset."""

    st.info(
        "This uploaded file does not match the airline-delay training schema, "
        "so generic dataset analytics are shown."
    )
    if schema_message:
        st.caption(schema_message)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Dataset", dataset_name)
    metric_col2.metric("Rows", len(dataset))
    metric_col3.metric("Columns", len(dataset.columns))

    missing_values = dataset.isna().sum().sort_values(ascending=False)
    numeric_columns = dataset.select_dtypes(include="number").columns.tolist()

    st.subheader("Missing Values by Column")
    st.bar_chart(missing_values)

    if numeric_columns:
        st.subheader("Numeric Column Preview")
        selected_column = st.selectbox("Numeric Column", numeric_columns)
        st.line_chart(dataset[selected_column].reset_index(drop=True))

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(dataset, use_container_width=True)


def render_dataset_analytics_page(uploaded_file=None) -> None:
    """Render charts from the airline delay dataset."""

    st.title("Dataset Analytics")

    delay_data, dataset_name, is_delay_dataset, schema_message = get_dashboard_delay_data(
        uploaded_file
    )
    if delay_data.empty:
        st.warning("Delay dataset is not available.")
        return

    st.caption(f"Active dataset: {dataset_name}")

    if not is_delay_dataset:
        render_generic_dataset_summary(delay_data, dataset_name, schema_message)
        return

    delay_rate = delay_data["delayed"].mean()
    peak_hour = int(delay_data["scheduled_hour"].value_counts().idxmax())
    busiest_airline = delay_data["airline"].value_counts().idxmax()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Flights", len(delay_data))
    metric_col2.metric("Delay Rate", f"{delay_rate:.1%}")
    metric_col3.metric("Peak Hour", f"{peak_hour}:00")
    metric_col4.metric("Busiest Airline", busiest_airline)

    chart_col1, chart_col2 = st.columns(2)

    delay_distribution = (
        delay_data["delayed"]
        .replace({0: "On Time", 1: "Delayed"})
        .value_counts()
        .rename_axis("status")
        .reset_index(name="flights")
        .set_index("status")
    )
    chart_col1.subheader("Delay Distribution")
    chart_col1.bar_chart(delay_distribution)

    traffic_by_hour = (
        delay_data.groupby("scheduled_hour")["passenger_count"]
        .sum()
        .rename("passengers")
        .sort_index()
    )
    chart_col2.subheader("Peak Traffic by Hour")
    chart_col2.line_chart(traffic_by_hour)

    st.subheader("Weather and Delay")
    weather_delay = (
        delay_data.groupby("weather_condition")["delayed"]
        .mean()
        .sort_values(ascending=False)
        .rename("delay_rate")
    )
    st.bar_chart(weather_delay)

    st.subheader("Cleaned Delay Dataset")
    st.dataframe(delay_data, use_container_width=True)


def _select_options(data: pd.DataFrame, column: str) -> list[str]:
    """Return sorted text options for a dashboard select box."""

    return sorted(data[column].dropna().astype(str).unique().tolist())


def render_prediction_page(uploaded_file=None) -> None:
    """Render the ML delay prediction page."""

    st.title("ML Prediction")

    model = load_best_model()
    delay_data, dataset_name, is_delay_dataset, schema_message = get_dashboard_delay_data(
        uploaded_file
    )

    if model is None:
        st.warning("Best model file is not available. Run ml/train.py first.")
        return

    st.caption(f"Active dataset for select boxes: {dataset_name}")

    if not is_delay_dataset:
        st.warning(
            "The uploaded CSV can be previewed in Dataset Analytics, but it "
            "cannot be used for delay prediction because required columns are "
            "missing."
        )
        if schema_message:
            st.caption(schema_message)
        delay_data = load_clean_delay_data()

    with st.form("delay_prediction_form"):
        col1, col2, col3 = st.columns(3)

        airline = col1.selectbox("Airline", _select_options(delay_data, "airline"))
        origin_airport = col1.selectbox(
            "Origin Airport",
            _select_options(delay_data, "origin_airport"),
        )
        destination_airport = col1.selectbox(
            "Destination Airport",
            _select_options(delay_data, "destination_airport"),
        )

        flight_day = col2.selectbox("Flight Day", _select_options(delay_data, "flight_day"))
        weather_condition = col2.selectbox(
            "Weather",
            _select_options(delay_data, "weather_condition"),
        )
        scheduled_hour = col2.slider("Scheduled Hour", 0, 23, 14)

        passenger_count = col3.number_input("Passenger Count", 50, 300, 165)
        previous_delay_minutes = col3.number_input("Previous Delay Minutes", 0, 180, 10)
        security_wait_minutes = col3.number_input("Security Wait Minutes", 0, 120, 20)
        gate_changes = col3.number_input("Gate Changes", 0, 5, 0)

        submitted = st.form_submit_button("Predict Delay")

    if submitted:
        prediction_data = pd.DataFrame(
            [
                {
                    "airline": airline,
                    "origin_airport": origin_airport,
                    "destination_airport": destination_airport,
                    "flight_day": flight_day,
                    "scheduled_hour": scheduled_hour,
                    "weather_condition": weather_condition,
                    "passenger_count": passenger_count,
                    "previous_delay_minutes": previous_delay_minutes,
                    "security_wait_minutes": security_wait_minutes,
                    "gate_changes": gate_changes,
                }
            ]
        )

        prediction = int(model.predict(prediction_data)[0])
        result_label = "Delayed" if prediction == 1 else "On Time"

        result_col1, result_col2 = st.columns(2)
        result_col1.metric("Prediction", result_label)

        if hasattr(model, "predict_proba"):
            delay_probability = model.predict_proba(prediction_data)[0][1]
            result_col2.metric("Delay Probability", f"{delay_probability:.1%}")

        st.dataframe(prediction_data, use_container_width=True)


def render_model_performance_page() -> None:
    """Render model comparison metrics and saved charts."""

    st.title("Model Performance")

    metrics_data = load_csv(MODEL_METRICS_PATH)
    if metrics_data.empty:
        st.warning("Model metrics are not available. Run ml/train.py first.")
        return

    best_row = metrics_data.sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    ).iloc[0]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Best Model", best_row["model"])
    metric_col2.metric("Accuracy", f"{best_row['accuracy']:.3f}")
    metric_col3.metric("Precision", f"{best_row['precision']:.3f}")
    metric_col4.metric("F1 Score", f"{best_row['f1_score']:.3f}")

    st.subheader("Metric Comparison")
    metric_columns = ["accuracy", "precision", "recall", "f1_score"]
    st.bar_chart(metrics_data.set_index("model")[metric_columns])

    chart_col1, chart_col2 = st.columns(2)
    if MODEL_CHART_PNG_PATH.exists():
        chart_col1.subheader("Report Chart")
        chart_col1.image(str(MODEL_CHART_PNG_PATH))

    if MODEL_CHART_HTML_PATH.exists():
        chart_col2.subheader("Interactive Chart")
        components.html(
            MODEL_CHART_HTML_PATH.read_text(encoding="utf-8"),
            height=460,
            scrolling=True,
        )

    st.subheader("Model Metrics Table")
    st.dataframe(metrics_data, use_container_width=True)


def _split_count(total: int, slots: int) -> list[int]:
    """Split a queue length across multiple station cards."""

    base_count = total // slots
    remainder = total % slots
    return [base_count + (1 if index < remainder else 0) for index in range(slots)]


def _active_passengers(
    event_log: pd.DataFrame,
    start_event: str,
    end_event: str,
    current_time: float,
    limit: int,
) -> list[str]:
    """Find passenger IDs being served at a station at the current frame."""

    active_passengers = []
    if event_log.empty:
        return active_passengers

    for passenger_id, passenger_events in event_log.groupby("passenger_id"):
        event_times = dict(zip(passenger_events["event"], passenger_events["time"]))
        start_time = event_times.get(start_event)
        end_time = event_times.get(end_event)

        if start_time is None or end_time is None:
            continue
        if start_time <= current_time < end_time:
            active_passengers.append(f"P{int(passenger_id):03d}")
        if len(active_passengers) == limit:
            break

    return active_passengers


def render_station_card(
    title: str,
    queue_length: int,
    active_passenger: str,
    average_wait: float,
    completion_time: str | None = None,
) -> None:
    """Render one airport operations card."""

    status = "Busy" if active_passenger != "None" else "Open"
    status_class = "status-busy" if status == "Busy" else "status-open"
    completion_row = ""

    if completion_time is not None:
        completion_row = (
            '<div class="airport-card-row">'
            "<span>Completion</span>"
            f'<span class="airport-card-value">{html.escape(completion_time)}</span>'
            "</div>"
        )

    st.markdown(
        (
            '<div class="airport-card">'
            f'<div class="airport-card-title">{html.escape(title)}</div>'
            '<div class="airport-card-row">'
            "<span>Queue length</span>"
            f'<span class="airport-card-value">{queue_length}</span>'
            "</div>"
            '<div class="airport-card-row">'
            "<span>Serving</span>"
            f'<span class="airport-card-value">{html.escape(active_passenger)}</span>'
            "</div>"
            '<div class="airport-card-row">'
            "<span>Average wait</span>"
            f'<span class="airport-card-value">{average_wait:.2f} min</span>'
            "</div>"
            f"{completion_row}"
            f'<span class="status-badge {status_class}">{status}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_airport_station_cards(
    live_data: dict[str, object],
    frame_index: int,
) -> None:
    """Render check-in, security, and boarding station cards."""

    event_log = live_data["event_log"]
    queue_timeline = live_data["queue_timeline"]
    statistics = live_data["statistics"]
    current_row = queue_timeline.iloc[frame_index]
    current_time = float(current_row["time"])

    check_in_active = _active_passengers(
        event_log,
        "started_check_in",
        "finished_check_in",
        current_time,
        3,
    )
    security_active = _active_passengers(
        event_log,
        "started_security",
        "finished_security",
        current_time,
        2,
    )
    boarding_active = _active_passengers(
        event_log,
        "started_boarding",
        "boarded_flight",
        current_time,
        2,
    )

    st.markdown("### Check-in Counters")
    check_in_queues = _split_count(int(current_row["check_in_queue"]), 3)
    check_in_cols = st.columns(3)
    for index, column in enumerate(check_in_cols):
        with column:
            render_station_card(
                f"Counter {index + 1}",
                check_in_queues[index],
                check_in_active[index] if index < len(check_in_active) else "None",
                float(statistics["average_check_in_wait"]),
            )

    st.markdown("### Security Checkpoints")
    security_queues = _split_count(int(current_row["security_queue"]), 2)
    security_cols = st.columns(2)
    for index, column in enumerate(security_cols):
        with column:
            render_station_card(
                f"Lane {index + 1}",
                security_queues[index],
                security_active[index] if index < len(security_active) else "None",
                float(statistics["average_security_wait"]),
            )

    st.markdown("### Boarding Gates")
    boarding_queues = _split_count(int(current_row["boarding_queue"]), 2)
    gate_cols = st.columns(2)
    for index, column in enumerate(gate_cols):
        estimated_minutes = current_time + boarding_queues[index] * 1.7
        with column:
            render_station_card(
                f"Gate A{index + 1}",
                boarding_queues[index],
                boarding_active[index] if index < len(boarding_active) else "None",
                float(statistics["average_boarding_wait"]),
                completion_time=f"{estimated_minutes:.1f} min",
            )


def _flight_status_class(status: str) -> str:
    """Return the CSS class for a flight board status."""

    status_key = status.lower().replace(" ", "-")
    return f"flight-{status_key}"


def render_flight_information_screen(flight_delay_data: pd.DataFrame) -> None:
    """Render a real-airport style flight information screen."""

    st.markdown("### Flight Status Screen")

    rows = []
    for _, flight in flight_delay_data.iterrows():
        status = str(flight["status"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(flight['flight_number']))}</td>"
            f"<td>{html.escape(str(flight['destination']))}</td>"
            f"<td>{html.escape(str(flight['scheduled_time']))}</td>"
            f"<td>{html.escape(str(flight['estimated_time']))}</td>"
            "<td>"
            f'<span class="flight-status {_flight_status_class(status)}">'
            f"{html.escape(status)}</span>"
            "</td>"
            "</tr>"
        )

    st.markdown(
        (
            '<table class="flight-board">'
            "<thead>"
            "<tr>"
            "<th>Flight Number</th>"
            "<th>Destination</th>"
            "<th>Scheduled Time</th>"
            "<th>Estimated Time</th>"
            "<th>Status</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        ),
        unsafe_allow_html=True,
    )


def flight_screen_for_frame(
    flight_delay_data: pd.DataFrame,
    frame_index: int,
    frame_count: int,
) -> pd.DataFrame:
    """Return flight statuses adjusted for the current animation frame."""

    animated_flights = flight_delay_data.copy()
    progress = frame_index / max(frame_count - 1, 1)

    if progress < 0.25:
        animated_flights.loc[
            animated_flights["status"].isin(["Boarding", "Gate Closed", "Departed"]),
            "status",
        ] = "On Time"
    elif progress < 0.55:
        animated_flights.loc[animated_flights.index[:2], "status"] = "Boarding"
    elif progress < 0.80:
        animated_flights.loc[animated_flights.index[:2], "status"] = "Gate Closed"
    else:
        animated_flights.loc[animated_flights.index[:2], "status"] = "Departed"

    delayed_mask = animated_flights["delay_minutes"] >= 25
    animated_flights.loc[delayed_mask, "status"] = "Delayed"

    return animated_flights


def render_live_frame_header(live_data: dict[str, object], frame_index: int) -> None:
    """Render compact frame-level metrics above the animated cards."""

    queue_timeline = live_data["queue_timeline"]
    current_row = queue_timeline.iloc[frame_index]

    frame_col1, frame_col2, frame_col3, frame_col4 = st.columns(4)
    frame_col1.metric("Simulation Time", f"{current_row['time']:.1f} min")
    frame_col2.metric("Check-in Queue", int(current_row["check_in_queue"]))
    frame_col3.metric("Security Queue", int(current_row["security_queue"]))
    frame_col4.metric("Boarding Queue", int(current_row["boarding_queue"]))


def render_airport_delay_charts(live_data: dict[str, object]) -> None:
    """Render Plotly charts for airport delay and queue behavior."""

    st.markdown("### Delay and Queue Analytics")

    flight_delay_data = live_data["flight_delay_data"].copy()
    queue_timeline = live_data["queue_timeline"].copy()
    stage_wait_times = live_data["stage_wait_times"].copy()

    status_colors = {
        "On Time": "#47d16c",
        "Delayed": "#d64545",
        "Boarding": "#2f80ed",
        "Gate Closed": "#f5b84b",
        "Departed": "#73808c",
    }

    delay_chart = px.bar(
        flight_delay_data,
        x="flight_number",
        y="delay_minutes",
        color="status",
        color_discrete_map=status_colors,
        labels={
            "flight_number": "Flight",
            "delay_minutes": "Delay Minutes",
            "status": "Status",
        },
        title="Delay Minutes by Flight",
        template="plotly_dark",
    )

    queue_chart = px.line(
        queue_timeline,
        x="time",
        y=["check_in_queue", "security_queue", "boarding_queue", "total_queue"],
        labels={"time": "Simulation Time", "value": "Queue Length", "variable": "Queue"},
        title="Passenger Queue Length Over Time",
        template="plotly_dark",
    )

    wait_chart = px.bar(
        stage_wait_times,
        x="stage",
        y="average_wait_minutes",
        color="stage",
        labels={"stage": "Airport Stage", "average_wait_minutes": "Average Wait"},
        title="Average Wait Time by Airport Stage",
        template="plotly_dark",
    )
    wait_chart.update_layout(showlegend=False)

    flight_delay_data["delay_group"] = flight_delay_data["delay_minutes"].apply(
        lambda minutes: "Delayed" if minutes >= 15 else "On Time"
    )
    pie_chart = px.pie(
        flight_delay_data,
        names="delay_group",
        color="delay_group",
        color_discrete_map={"Delayed": "#d64545", "On Time": "#47d16c"},
        title="Delayed vs On-Time Flights",
        template="plotly_dark",
    )

    chart_col1, chart_col2 = st.columns(2)
    chart_col1.plotly_chart(delay_chart, use_container_width=True)
    chart_col2.plotly_chart(queue_chart, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    chart_col3.plotly_chart(wait_chart, use_container_width=True)
    chart_col4.plotly_chart(pie_chart, use_container_width=True)


def render_airport_live_simulation_page() -> None:
    """Render a visual airport operations dashboard."""

    render_airport_dashboard_css()
    st.markdown(
        """
        <div class="airport-screen">
            <div class="airport-title">Airport Live Simulation</div>
            <div class="airport-subtitle">Passenger flow, gate activity, and flight delay operations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col1, control_col2, control_col3 = st.columns(3)
    passengers = control_col1.slider("Live Passengers", 30, 180, 90, 10)
    arrival_interval = control_col2.slider("Arrival Gap", 0.8, 5.0, 2.0, 0.2)
    random_seed = control_col3.number_input("Live Seed", min_value=1, value=42)

    live_data = load_live_airport_data(passengers, arrival_interval, int(random_seed))
    statistics = live_data["statistics"]
    queue_timeline = live_data["queue_timeline"]
    flight_delay_data = live_data["flight_delay_data"]

    latest_queue = queue_timeline.iloc[-1]
    delayed_flights = int((flight_delay_data["status"] == "Delayed").sum())

    st.markdown("### Operations Overview")
    overview_cols = st.columns(5)
    overview_values = [
        ("Passenger Arrival", statistics["total_passengers"], "simulated passengers"),
        ("Check-in Counters", "3", "active counters"),
        ("Security Checkpoints", "2", "active lanes"),
        ("Boarding Gates", "A1 / A2", "boarding positions"),
        ("Flight Status Screen", delayed_flights, "delayed flights"),
    ]

    for column, (title, value, label) in zip(overview_cols, overview_values):
        column.markdown(
            f"""
            <div class="airport-zone">
                <h4>{title}</h4>
                <div class="airport-zone-value">{value}</div>
                <div class="airport-zone-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"Latest total queue: {int(latest_queue['total_queue'])} passengers | "
        f"Average total wait: {statistics['average_total_wait']} minutes"
    )

    default_frame = min(int(len(queue_timeline) * 0.6), len(queue_timeline) - 1)
    start_simulation = st.button("Start Simulation", type="primary")
    live_placeholder = st.empty()

    if start_simulation:
        for frame_index in range(len(queue_timeline)):
            live_placeholder.empty()
            with live_placeholder.container():
                render_live_frame_header(live_data, frame_index)
                render_airport_station_cards(live_data, frame_index)
                render_flight_information_screen(
                    flight_screen_for_frame(
                        flight_delay_data,
                        frame_index,
                        len(queue_timeline),
                    )
                )
            time.sleep(0.18)

        final_frame = len(queue_timeline) - 1
        live_placeholder.empty()
        with live_placeholder.container():
            render_live_frame_header(live_data, final_frame)
            render_airport_station_cards(live_data, final_frame)
            render_flight_information_screen(
                flight_screen_for_frame(
                    flight_delay_data,
                    final_frame,
                    len(queue_timeline),
                )
            )
    else:
        with live_placeholder.container():
            render_live_frame_header(live_data, default_frame)
            render_airport_station_cards(live_data, default_frame)
            render_flight_information_screen(
                flight_screen_for_frame(
                    flight_delay_data,
                    default_frame,
                    len(queue_timeline),
                )
            )

    render_airport_delay_charts(live_data)


def main() -> None:
    """Render the selected dashboard page."""

    st.sidebar.title("Navigation")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        help=(
            "Upload any CSV for preview. Use the sample airline-delay columns "
            "to enable full delay analytics and prediction."
        ),
    )

    selected_page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Dataset Analytics",
            "Simulation Analytics",
            "Airport Live Simulation",
            "Advanced Animated Airport",
            "Professional Airport Control Center",
            "ML Prediction",
            "Model Performance",
        ],
    )

    if selected_page == "Home":
        render_home_page()
    elif selected_page == "Dataset Analytics":
        render_dataset_analytics_page(uploaded_file)
    elif selected_page == "Simulation Analytics":
        render_simulation_analytics_page()
    elif selected_page == "Airport Live Simulation":
        render_airport_live_simulation_page()
    elif selected_page == "Advanced Animated Airport":
        render_advanced_animated_airport_page()
    elif selected_page == "Professional Airport Control Center":
        render_professional_airport_control_center_page()
    elif selected_page == "ML Prediction":
        render_prediction_page(uploaded_file)
    elif selected_page == "Model Performance":
        render_model_performance_page()


if __name__ == "__main__":
    main()
