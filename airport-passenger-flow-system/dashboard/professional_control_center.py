"""Professional Airport Control Center page for the Streamlit dashboard."""

from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    clean_delay_data,
    load_delay_dataset,
)

BEST_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_delay_model.joblib"
RAW_DELAY_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "airline_delay_realistic.csv"
CLEAN_DELAY_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "airline_delay_cleaned.csv"
MODEL_INPUT_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


@st.cache_resource
def load_professional_delay_model():
    """Load the saved delay model used by the control center."""

    if BEST_MODEL_PATH.exists():
        return joblib.load(BEST_MODEL_PATH)

    return None


@st.cache_data
def load_professional_delay_data() -> pd.DataFrame:
    """Load cleaned delay data for select boxes and realistic defaults."""

    if CLEAN_DELAY_DATA_PATH.exists():
        return pd.read_csv(CLEAN_DELAY_DATA_PATH)

    return clean_delay_data(load_delay_dataset(RAW_DELAY_DATA_PATH))


def render_professional_streamlit_css() -> None:
    """Style Streamlit sections so they match the control room page."""

    st.markdown(
        """
        <style>
        .pcc-panel {
            background: #071624;
            border: 1px solid rgba(59, 215, 255, 0.32);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .pcc-panel-title {
            color: #ecf8ff;
            font-size: 24px;
            font-weight: 900;
            margin-bottom: 4px;
        }
        .pcc-panel-subtitle {
            color: #8ea9bb;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sorted_options(data: pd.DataFrame, column: str) -> list[str]:
    """Return clean select-box options for one dataset column."""

    return sorted(data[column].dropna().astype(str).unique().tolist())


def _risk_level(delay_probability: float) -> str:
    """Convert a model probability into a readable operations risk level."""

    if delay_probability >= 0.75:
        return "Critical"
    if delay_probability >= 0.55:
        return "High"
    if delay_probability >= 0.35:
        return "Medium"
    return "Low"


def estimate_delay_minutes(
    delay_probability: float,
    previous_delay_minutes: float,
    security_wait_minutes: float,
    gate_changes: int,
) -> int:
    """Estimate delay minutes from the classifier probability and operations data."""

    estimate = (
        delay_probability * 55
        + previous_delay_minutes * 0.22
        + security_wait_minutes * 0.18
        + gate_changes * 5
        - 18
    )
    return max(0, int(round(estimate)))


def predict_delay_summary(model, model_input: pd.DataFrame) -> dict[str, object]:
    """Run the saved ML model and return display-ready delay values."""

    if model is None or model_input.empty:
        return {
            "prediction": "Model Missing",
            "delay_probability": 0.0,
            "estimated_delay_minutes": 0,
            "risk_level": "Unknown",
        }

    prediction = int(model.predict(model_input)[0])
    if hasattr(model, "predict_proba"):
        delay_probability = float(model.predict_proba(model_input)[0][1])
    else:
        delay_probability = float(prediction)

    row = model_input.iloc[0]
    estimated_delay_minutes = estimate_delay_minutes(
        delay_probability,
        float(row["previous_delay_minutes"]),
        float(row["security_wait_minutes"]),
        int(row["gate_changes"]),
    )

    return {
        "prediction": "Delayed" if prediction == 1 else "On Time",
        "delay_probability": delay_probability,
        "estimated_delay_minutes": estimated_delay_minutes,
        "risk_level": _risk_level(delay_probability),
    }


def render_delay_prediction_section(
    delay_data: pd.DataFrame,
    model,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Render the ML delay prediction controls and summary metrics."""

    st.markdown(
        """
        <div class="pcc-panel">
            <div class="pcc-panel-title">Delay Prediction</div>
            <div class="pcc-panel-subtitle">Saved ML model connected to the control center flight operations view</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if delay_data.empty:
        st.warning("Delay data is not available for control-center predictions.")
        empty_input = pd.DataFrame(columns=MODEL_INPUT_COLUMNS)
        return empty_input, predict_delay_summary(model, empty_input)

    defaults = delay_data.iloc[0]
    input_col1, input_col2, input_col3 = st.columns(3)

    airline = input_col1.selectbox(
        "Airline",
        _sorted_options(delay_data, "airline"),
        key="pcc_airline",
    )
    origin_airport = input_col1.selectbox(
        "Origin Airport",
        _sorted_options(delay_data, "origin_airport"),
        key="pcc_origin_airport",
    )
    destination_airport = input_col1.selectbox(
        "Destination Airport",
        _sorted_options(delay_data, "destination_airport"),
        key="pcc_destination_airport",
    )

    flight_day = input_col2.selectbox(
        "Flight Day",
        _sorted_options(delay_data, "flight_day"),
        key="pcc_flight_day",
    )
    weather_condition = input_col2.selectbox(
        "Weather",
        _sorted_options(delay_data, "weather_condition"),
        key="pcc_weather_condition",
    )
    scheduled_hour = input_col2.slider(
        "Scheduled Hour",
        0,
        23,
        int(defaults["scheduled_hour"]),
        key="pcc_scheduled_hour",
    )

    passenger_count = input_col3.number_input(
        "Passenger Count",
        50,
        320,
        int(defaults["passenger_count"]),
        key="pcc_passenger_count",
    )
    previous_delay_minutes = input_col3.number_input(
        "Previous Delay Minutes",
        0,
        180,
        int(defaults["previous_delay_minutes"]),
        key="pcc_previous_delay_minutes",
    )
    security_wait_minutes = input_col3.number_input(
        "Security Wait Minutes",
        0,
        120,
        int(defaults["security_wait_minutes"]),
        key="pcc_security_wait_minutes",
    )
    gate_changes = input_col3.number_input(
        "Gate Changes",
        0,
        5,
        int(defaults["gate_changes"]),
        key="pcc_gate_changes",
    )

    model_input = pd.DataFrame(
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
    summary = predict_delay_summary(model, model_input)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Delay Prediction", str(summary["prediction"]))
    metric_col2.metric("Delay Probability", f"{summary['delay_probability']:.1%}")
    metric_col3.metric("Estimated Delay Minutes", summary["estimated_delay_minutes"])
    metric_col4.metric("Risk Level", str(summary["risk_level"]))

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(summary["delay_probability"]) * 100,
            number={"suffix": "%"},
            title={"text": "Delay Probability Gauge"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3bd7ff"},
                "steps": [
                    {"range": [0, 35], "color": "#12382d"},
                    {"range": [35, 55], "color": "#3a3212"},
                    {"range": [55, 100], "color": "#3d1519"},
                ],
            },
        )
    )
    gauge.update_layout(
        template="plotly_dark",
        height=260,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    st.plotly_chart(gauge, use_container_width=True)

    return model_input, summary


def build_professional_control_center_html() -> str:
    """Build the HTML/CSS/JavaScript airport control center component."""

    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  :root {
    --bg: #030914;
    --panel: #071624;
    --panel-soft: #0b2031;
    --line: #1f78a8;
    --neon: #3bd7ff;
    --text: #ecf8ff;
    --muted: #8ea9bb;
    --yellow: #ffd166;
    --green: #34d399;
    --blue: #3b82f6;
    --red: #f87171;
    --busy: #fbbf24;
    --closed: #64748b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, Segoe UI, Arial, sans-serif;
  }
  .control-shell {
    min-height: 960px;
    padding: 18px;
    background:
      radial-gradient(circle at 18% 0%, rgba(59, 215, 255, 0.18), transparent 26%),
      linear-gradient(rgba(59, 215, 255, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(59, 215, 255, 0.04) 1px, transparent 1px),
      #030914;
    background-size: auto, 34px 34px, 34px 34px;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: center;
    margin-bottom: 14px;
  }
  .title {
    font-size: 34px;
    font-weight: 900;
    letter-spacing: 0;
    color: #f3fbff;
  }
  .subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px;
  }
  .system-chip {
    padding: 10px 14px;
    border: 1px solid rgba(59, 215, 255, 0.42);
    border-radius: 8px;
    color: var(--neon);
    background: rgba(7, 22, 36, 0.9);
    font-weight: 900;
    box-shadow: 0 0 24px rgba(59, 215, 255, 0.12);
  }
  .airport-map {
    position: relative;
    height: 760px;
    border: 1px solid rgba(59, 215, 255, 0.34);
    border-radius: 14px;
    background:
      linear-gradient(135deg, rgba(59, 215, 255, 0.08), transparent 44%),
      rgba(7, 22, 36, 0.94);
    overflow: hidden;
    box-shadow: inset 0 0 60px rgba(59, 215, 255, 0.07), 0 18px 48px rgba(0, 0, 0, 0.32);
  }
  .zone {
    position: absolute;
    border: 1px solid rgba(59, 215, 255, 0.44);
    border-radius: 10px;
    background: rgba(11, 32, 49, 0.88);
    padding: 10px;
    min-height: 78px;
  }
  .zone-title {
    color: var(--neon);
    font-size: 12px;
    font-weight: 900;
    text-transform: uppercase;
  }
  .zone-note {
    color: var(--muted);
    font-size: 11px;
    margin-top: 4px;
  }
  .entrance { left: 24px; top: 314px; width: 126px; height: 112px; }
  .checkin-queue { left: 188px; top: 250px; width: 142px; height: 240px; }
  .checkin-counters { left: 370px; top: 210px; width: 150px; height: 320px; }
  .security-queue { left: 560px; top: 250px; width: 132px; height: 240px; }
  .security-lanes { left: 730px; top: 210px; width: 145px; height: 320px; }
  .immigration-queue { left: 916px; top: 250px; width: 142px; height: 240px; }
  .immigration-counters { left: 1098px; top: 210px; width: 160px; height: 320px; }
  .lounge { left: 260px; top: 588px; width: 280px; height: 122px; }
  .boarding-queue { left: 598px; top: 588px; width: 190px; height: 122px; }
  .boarding-gates { left: 842px; top: 588px; width: 190px; height: 122px; }
  .aircraft { left: 1092px; top: 600px; width: 170px; height: 104px; }
  .desk {
    position: relative;
    border: 1px solid rgba(59, 215, 255, 0.42);
    background: rgba(59, 215, 255, 0.11);
    border-radius: 7px;
    padding: 7px;
    margin-top: 8px;
    color: #e8f8ff;
    font-size: 11px;
    font-weight: 800;
    overflow: hidden;
  }
  .desk .progress {
    position: absolute;
    left: 0;
    bottom: 0;
    height: 3px;
    width: 0%;
    background: linear-gradient(90deg, var(--green), var(--neon));
    transition: width 0.28s ease;
  }
  .route {
    position: absolute;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent);
    opacity: 0.78;
    transform-origin: left center;
  }
  .route.main { left: 142px; top: 370px; width: 1110px; }
  .route.down { left: 460px; top: 514px; width: 520px; transform: rotate(22deg); }
  .aircraft-body {
    position: absolute;
    right: 18px;
    bottom: 22px;
    width: 112px;
    height: 40px;
    border-radius: 52px 12px 12px 52px;
    background: linear-gradient(90deg, #e8f8ff, #75bdf1);
  }
  .aircraft-body::before,
  .aircraft-body::after {
    content: "";
    position: absolute;
    left: 43px;
    width: 48px;
    height: 10px;
    background: #75bdf1;
  }
  .aircraft-body::before { top: -10px; transform: rotate(-23deg); }
  .aircraft-body::after { bottom: -10px; transform: rotate(23deg); }
  .legend {
    display: flex;
    gap: 10px;
    margin-top: 12px;
    color: var(--muted);
    font-size: 12px;
  }
  .legend span {
    border: 1px solid rgba(59, 215, 255, 0.28);
    background: rgba(7, 22, 36, 0.88);
    border-radius: 999px;
    padding: 6px 10px;
  }
  .passenger {
    position: absolute;
    left: 0;
    top: 0;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--yellow);
    border: 2px solid #fff0b8;
    box-shadow: 0 0 15px rgba(255, 209, 102, 0.64);
    transform: translate(-40px, -40px);
    transition: opacity 0.2s ease;
    z-index: 20;
  }
  .passenger::after {
    content: attr(data-id);
    position: absolute;
    top: 21px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    font-weight: 900;
    color: #eefaff;
    text-shadow: 0 1px 4px #000;
    white-space: nowrap;
  }
  .passenger.boarded {
    background: var(--green);
    border-color: #d9ffec;
    box-shadow: 0 0 15px rgba(52, 211, 153, 0.68);
  }
  .queue-dot {
    position: absolute;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(59, 215, 255, 0.72);
    box-shadow: 0 0 11px rgba(59, 215, 255, 0.7);
    z-index: 8;
  }
  .operations-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 16px;
    margin-top: 16px;
  }
  .control-panel {
    border: 1px solid rgba(59, 215, 255, 0.32);
    border-radius: 14px;
    background: rgba(7, 22, 36, 0.92);
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
    padding: 14px;
  }
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }
  .panel-title {
    color: #f4fbff;
    font-size: 20px;
    font-weight: 900;
  }
  .panel-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
  }
  .resource-groups {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
  }
  .resource-group {
    border: 1px solid rgba(59, 215, 255, 0.22);
    border-radius: 10px;
    background: rgba(3, 9, 20, 0.56);
    padding: 10px;
  }
  .resource-group-title {
    color: var(--neon);
    font-size: 13px;
    font-weight: 900;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .resource-stack {
    display: grid;
    gap: 8px;
  }
  .resource-card {
    border: 1px solid rgba(52, 211, 153, 0.42);
    border-left: 5px solid var(--green);
    border-radius: 8px;
    background: rgba(11, 32, 49, 0.88);
    padding: 9px;
    min-height: 145px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
  }
  .resource-card.busy {
    border-color: rgba(251, 191, 36, 0.62);
    border-left-color: var(--busy);
    box-shadow: 0 0 18px rgba(251, 191, 36, 0.12);
  }
  .resource-card.overloaded {
    border-color: rgba(248, 113, 113, 0.72);
    border-left-color: var(--red);
    box-shadow: 0 0 22px rgba(248, 113, 113, 0.2);
    animation: overloadPulse 1s infinite alternate;
  }
  .resource-card.closed {
    border-color: rgba(100, 116, 139, 0.52);
    border-left-color: var(--closed);
    opacity: 0.72;
  }
  .resource-name {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: #eefaff;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 7px;
  }
  .status-pill {
    border-radius: 999px;
    padding: 3px 7px;
    background: rgba(52, 211, 153, 0.18);
    color: var(--green);
    font-size: 10px;
    font-weight: 900;
  }
  .status-pill.busy {
    background: rgba(251, 191, 36, 0.18);
    color: var(--busy);
  }
  .status-pill.overloaded {
    background: rgba(248, 113, 113, 0.18);
    color: var(--red);
  }
  .status-pill.closed {
    background: rgba(100, 116, 139, 0.2);
    color: #cbd5e1;
  }
  .bottleneck-alert {
    border: 1px solid rgba(52, 211, 153, 0.44);
    border-radius: 8px;
    background: rgba(52, 211, 153, 0.1);
    color: var(--green);
    padding: 9px 11px;
    min-width: 260px;
    text-align: right;
    font-size: 12px;
    font-weight: 900;
  }
  .bottleneck-alert.busy {
    border-color: rgba(251, 191, 36, 0.58);
    background: rgba(251, 191, 36, 0.12);
    color: var(--busy);
  }
  .bottleneck-alert.overloaded {
    border-color: rgba(248, 113, 113, 0.7);
    background: rgba(248, 113, 113, 0.13);
    color: var(--red);
    animation: overloadPulse 0.75s infinite alternate;
  }
  .bottleneck-alert.closed {
    border-color: rgba(100, 116, 139, 0.54);
    background: rgba(100, 116, 139, 0.12);
    color: #cbd5e1;
  }
  .resource-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: var(--muted);
    font-size: 11px;
    margin-top: 5px;
  }
  .resource-value {
    color: #f8fdff;
    font-weight: 900;
    text-align: right;
  }
  @keyframes overloadPulse {
    from { transform: translateY(0); }
    to { transform: translateY(-2px); }
  }
  .flight-panel {
    overflow: hidden;
  }
  .flight-clock {
    border: 1px solid rgba(59, 215, 255, 0.38);
    border-radius: 8px;
    color: var(--neon);
    background: rgba(3, 9, 20, 0.55);
    padding: 8px 10px;
    min-width: 110px;
    text-align: center;
    font-weight: 900;
  }
  .flight-board-table {
    width: 100%;
    border-collapse: collapse;
    color: #effaff;
    font-size: 13px;
    overflow: hidden;
    border-radius: 10px;
  }
  .flight-board-table th {
    background: rgba(59, 215, 255, 0.14);
    color: var(--neon);
    padding: 10px;
    text-align: left;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0;
    border-bottom: 1px solid rgba(59, 215, 255, 0.28);
  }
  .flight-board-table td {
    background: rgba(3, 9, 20, 0.62);
    border-bottom: 1px solid rgba(59, 215, 255, 0.12);
    padding: 10px;
    font-weight: 800;
  }
  .flight-board-table tr:last-child td {
    border-bottom: none;
  }
  .flight-row.delayed td {
    background: rgba(248, 113, 113, 0.14);
    animation: delayedRowPulse 0.95s infinite alternate;
  }
  .flight-status-badge {
    display: inline-flex;
    justify-content: center;
    min-width: 98px;
    border-radius: 999px;
    padding: 4px 9px;
    font-size: 11px;
    font-weight: 900;
  }
  .flight-status-badge.on-time {
    background: rgba(52, 211, 153, 0.18);
    color: var(--green);
  }
  .flight-status-badge.delayed {
    background: rgba(248, 113, 113, 0.2);
    color: var(--red);
    animation: delayBlink 0.72s infinite alternate;
  }
  .flight-status-badge.boarding {
    background: rgba(59, 130, 246, 0.2);
    color: #93c5fd;
  }
  .flight-status-badge.gate-closed {
    background: rgba(251, 191, 36, 0.18);
    color: var(--busy);
  }
  .flight-status-badge.departed {
    background: rgba(100, 116, 139, 0.24);
    color: #cbd5e1;
  }
  @keyframes delayedRowPulse {
    from { box-shadow: inset 4px 0 0 rgba(248, 113, 113, 0.28); }
    to { box-shadow: inset 4px 0 0 rgba(248, 113, 113, 0.86); }
  }
  @keyframes delayBlink {
    from { filter: brightness(0.86); }
    to { filter: brightness(1.35); }
  }
  @media (max-width: 1100px) {
    .resource-groups { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
</style>
</head>
<body>
  <div class="control-shell">
    <div class="topbar">
      <div>
        <div class="title">Professional Airport Control Center</div>
        <div class="subtitle">Terminal movement map | check-in, security, immigration, lounge, boarding and aircraft flow</div>
      </div>
      <div class="system-chip">TERMINAL OPS ONLINE</div>
    </div>
    <div class="airport-map" id="airport-map">
      <div class="route main"></div>
      <div class="route down"></div>
      <section class="zone entrance"><div class="zone-title">Entrance</div><div class="zone-note">Arrivals enter terminal</div></section>
      <section class="zone checkin-queue"><div class="zone-title">Check-in Queue</div><div class="zone-note">Waiting passengers</div></section>
      <section class="zone checkin-counters"><div class="zone-title">Check-in Counters</div><div class="desk service-desk">Counter 1<div class="progress"></div></div><div class="desk service-desk">Counter 2<div class="progress"></div></div><div class="desk service-desk">Counter 3<div class="progress"></div></div></section>
      <section class="zone security-queue"><div class="zone-title">Security Queue</div><div class="zone-note">Screening line</div></section>
      <section class="zone security-lanes"><div class="zone-title">Security Lanes</div><div class="desk service-desk">Lane 1<div class="progress"></div></div><div class="desk service-desk">Lane 2<div class="progress"></div></div></section>
      <section class="zone immigration-queue"><div class="zone-title">Immigration Queue</div><div class="zone-note">Passport control line</div></section>
      <section class="zone immigration-counters"><div class="zone-title">Immigration Counters</div><div class="desk service-desk">Immigration 1<div class="progress"></div></div><div class="desk service-desk">Immigration 2<div class="progress"></div></div><div class="desk service-desk">Immigration 3<div class="progress"></div></div></section>
      <section class="zone lounge"><div class="zone-title">Waiting Lounge</div><div class="zone-note">Cleared passengers</div></section>
      <section class="zone boarding-queue"><div class="zone-title">Boarding Queue</div><div class="zone-note">Gate line-up</div></section>
      <section class="zone boarding-gates"><div class="zone-title">Boarding Gates</div><div class="desk service-desk">Gate A1<div class="progress"></div></div><div class="desk service-desk">Gate A2<div class="progress"></div></div></section>
      <section class="zone aircraft"><div class="zone-title">Aircraft</div><div class="zone-note">Boarded area</div><div class="aircraft-body"></div></section>
    </div>
    <div class="legend">
      <span>Blue route: active passenger flow</span>
      <span>Yellow dots: passengers</span>
      <span>Green dots: boarded</span>
      <span>Progress bars activate at counters</span>
    </div>
    <div class="operations-grid">
      <section class="control-panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Resource Operations Monitor</div>
            <div class="panel-subtitle">Live service state for counters, lanes, immigration and gates</div>
          </div>
          <div class="bottleneck-alert" id="bottleneck-alert">NORMAL FLOW</div>
        </div>
        <div class="resource-groups">
          <div class="resource-group">
            <div class="resource-group-title">Check-in Counters</div>
            <div class="resource-stack">
              <div class="resource-card" data-resource="checkin-0">
                <div class="resource-name">Counter 1 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
              <div class="resource-card" data-resource="checkin-1">
                <div class="resource-name">Counter 2 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
              <div class="resource-card" data-resource="checkin-2">
                <div class="resource-name">Counter 3 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
            </div>
          </div>
          <div class="resource-group">
            <div class="resource-group-title">Security Lanes</div>
            <div class="resource-stack">
              <div class="resource-card" data-resource="security-0">
                <div class="resource-name">Lane 1 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
              <div class="resource-card" data-resource="security-1">
                <div class="resource-name">Lane 2 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
            </div>
          </div>
          <div class="resource-group">
            <div class="resource-group-title">Immigration Counters</div>
            <div class="resource-stack">
              <div class="resource-card" data-resource="immigration-0">
                <div class="resource-name">Immigration 1 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
              <div class="resource-card" data-resource="immigration-1">
                <div class="resource-name">Immigration 2 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
              <div class="resource-card" data-resource="immigration-2">
                <div class="resource-name">Immigration 3 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
            </div>
          </div>
          <div class="resource-group">
            <div class="resource-group-title">Boarding Gates</div>
            <div class="resource-stack">
              <div class="resource-card" data-resource="boarding-0">
                <div class="resource-name">Gate A1 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
              <div class="resource-card" data-resource="boarding-1">
                <div class="resource-name">Gate A2 <span class="status-pill" data-field="status">OPEN</span></div>
                <div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div>
                <div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div>
                <div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div>
                <div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div>
              </div>
            </div>
          </div>
        </div>
      </section>
      <section class="control-panel flight-panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Airport Flight Information Board</div>
            <div class="panel-subtitle">Gate-level departure status with animated delay alerts</div>
          </div>
          <div class="flight-clock" id="flight-clock">09:00</div>
        </div>
        <table class="flight-board-table" aria-label="Airport Flight Information Board">
          <thead>
            <tr>
              <th>Flight</th>
              <th>Destination</th>
              <th>Gate</th>
              <th>Scheduled</th>
              <th>Estimated</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="flight-board-body"></tbody>
        </table>
      </section>
    </div>
  </div>
<script>
(() => {
  const map = document.getElementById("airport-map");
  const progressBars = Array.from(document.querySelectorAll(".service-desk .progress"));
  const resourceCards = Array.from(document.querySelectorAll(".resource-card"));
  const bottleneckAlert = document.getElementById("bottleneck-alert");
  const flightBoardBody = document.getElementById("flight-board-body");
  const flightClock = document.getElementById("flight-clock");
  const passengerCount = 46;
  const speed = 1.12;
  let lastFlightTick = -1;
  const resources = [
    { id: "checkin-0", group: "checkin", label: "Check-in Counter 1", index: 0, count: 3, serviceStage: "checkinCounter", queue: "checkin", baseWait: 2.4, waitFactor: 0.85, serviceMinutes: 3.4, busyAt: 2, overloadAt: 5 },
    { id: "checkin-1", group: "checkin", label: "Check-in Counter 2", index: 1, count: 3, serviceStage: "checkinCounter", queue: "checkin", baseWait: 2.4, waitFactor: 0.85, serviceMinutes: 3.4, busyAt: 2, overloadAt: 5 },
    { id: "checkin-2", group: "checkin", label: "Check-in Counter 3", index: 2, count: 3, serviceStage: "checkinCounter", queue: "checkin", baseWait: 2.4, waitFactor: 0.85, serviceMinutes: 3.4, busyAt: 2, overloadAt: 5 },
    { id: "security-0", group: "security", label: "Security Lane 1", index: 0, count: 2, serviceStage: "securityLane", queue: "security", baseWait: 3.1, waitFactor: 1.15, serviceMinutes: 3.2, busyAt: 2, overloadAt: 4 },
    { id: "security-1", group: "security", label: "Security Lane 2", index: 1, count: 2, serviceStage: "securityLane", queue: "security", baseWait: 3.1, waitFactor: 1.15, serviceMinutes: 3.2, busyAt: 2, overloadAt: 4 },
    { id: "immigration-0", group: "immigration", label: "Immigration Counter 1", index: 0, count: 3, serviceStage: "immigrationCounter", queue: "immigration", baseWait: 3.8, waitFactor: 1.05, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5 },
    { id: "immigration-1", group: "immigration", label: "Immigration Counter 2", index: 1, count: 3, serviceStage: "immigrationCounter", queue: "immigration", baseWait: 3.8, waitFactor: 1.05, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5 },
    { id: "immigration-2", group: "immigration", label: "Immigration Counter 3", index: 2, count: 3, serviceStage: "immigrationCounter", queue: "immigration", baseWait: 3.8, waitFactor: 1.05, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5, closesAfter: 62 },
    { id: "boarding-0", group: "boarding", label: "Boarding Gate A1", index: 0, count: 2, serviceStage: "boardingGate", queue: "boarding", baseWait: 1.6, waitFactor: 0.72, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5 },
    { id: "boarding-1", group: "boarding", label: "Boarding Gate A2", index: 1, count: 2, serviceStage: "boardingGate", queue: "boarding", baseWait: 1.6, waitFactor: 0.72, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5, opensAfter: 28 }
  ];
  const flights = [
    { flight: "PK-302", destination: "Karachi", gate: "A1", scheduledMinutes: 9 * 60 + 10, delayMinutes: 0, delayStart: 12, boardingStart: 24, gateClosedStart: 43, departedStart: 58, phaseOffset: 0 },
    { flight: "AP-144", destination: "Lahore", gate: "A2", scheduledMinutes: 9 * 60 + 25, delayMinutes: 18, delayStart: 10, boardingStart: 54, gateClosedStart: 70, departedStart: 82, phaseOffset: 8 },
    { flight: "SA-781", destination: "Dubai", gate: "B3", scheduledMinutes: 9 * 60 + 40, delayMinutes: 0, delayStart: 16, boardingStart: 36, gateClosedStart: 55, departedStart: 72, phaseOffset: 34 },
    { flight: "ER-219", destination: "Islamabad", gate: "C1", scheduledMinutes: 10 * 60 + 5, delayMinutes: 24, delayStart: 18, boardingStart: 66, gateClosedStart: 80, departedStart: 92, phaseOffset: 8 },
    { flight: "GB-508", destination: "Doha", gate: "A4", scheduledMinutes: 10 * 60 + 20, delayMinutes: 0, delayStart: 24, boardingStart: 52, gateClosedStart: 75, departedStart: 90, phaseOffset: 72 },
    { flight: "PK-419", destination: "Jeddah", gate: "D2", scheduledMinutes: 10 * 60 + 35, delayMinutes: 35, delayStart: 22, boardingStart: 78, gateClosedStart: 94, departedStart: 108, phaseOffset: 99 }
  ];
  const points = {
    entrance: { x: 82, y: 370 },
    checkinQueue: { x: 238, y: 370 },
    checkinCounter: { x: 444, y: 370 },
    securityQueue: { x: 616, y: 370 },
    securityLane: { x: 804, y: 370 },
    immigrationQueue: { x: 976, y: 370 },
    immigrationCounter: { x: 1168, y: 370 },
    lounge: { x: 398, y: 648 },
    boardingQueue: { x: 682, y: 648 },
    boardingGate: { x: 934, y: 648 },
    aircraft: { x: 1180, y: 648 }
  };
  const stages = [
    { name: "entrance", from: "entrance", to: "checkinQueue", duration: 2.0 },
    { name: "checkinQueue", from: "checkinQueue", to: "checkinQueue", duration: 3.0, queue: "checkin" },
    { name: "checkinCounter", from: "checkinQueue", to: "checkinCounter", duration: 3.4, progress: [0, 1, 2] },
    { name: "securityQueue", from: "checkinCounter", to: "securityQueue", duration: 2.5, queue: "security" },
    { name: "securityLane", from: "securityQueue", to: "securityLane", duration: 3.2, progress: [3, 4] },
    { name: "immigrationQueue", from: "securityLane", to: "immigrationQueue", duration: 2.4, queue: "immigration" },
    { name: "immigrationCounter", from: "immigrationQueue", to: "immigrationCounter", duration: 3.8, progress: [5, 6, 7] },
    { name: "lounge", from: "immigrationCounter", to: "lounge", duration: 4.5 },
    { name: "boardingQueue", from: "lounge", to: "boardingQueue", duration: 3.2, queue: "boarding" },
    { name: "boardingGate", from: "boardingQueue", to: "boardingGate", duration: 3.8, progress: [8, 9] },
    { name: "aircraft", from: "boardingGate", to: "aircraft", duration: 3.2 },
    { name: "boarded", from: "aircraft", to: "aircraft", duration: 999 }
  ];

  function ease(value) {
    return value < 0.5 ? 2 * value * value : 1 - Math.pow(-2 * value + 2, 2) / 2;
  }

  function interpolate(start, end, progress) {
    const curved = ease(Math.max(0, Math.min(progress, 1)));
    return {
      x: start.x + (end.x - start.x) * curved,
      y: start.y + (end.y - start.y) * curved
    };
  }

  function queuePosition(queue, index) {
    const column = index % 5;
    const row = Math.floor(index / 5) % 7;
    const positions = {
      checkin: { x: 208, y: 296, gapX: 23, gapY: 20 },
      security: { x: 582, y: 296, gapX: 21, gapY: 20 },
      immigration: { x: 938, y: 296, gapX: 23, gapY: 20 },
      boarding: { x: 622, y: 626, gapX: 23, gapY: 17 }
    };
    const base = positions[queue];
    return { x: base.x + column * base.gapX, y: base.y + row * base.gapY };
  }

  function phaseFor(localTime, index) {
    let cursor = 0;
    for (const stage of stages) {
      const next = cursor + stage.duration;
      if (localTime <= next) {
        const progress = (localTime - cursor) / stage.duration;
        if (stage.queue && progress < 0.74) {
          return { stage, progress, position: queuePosition(stage.queue, index) };
        }
        return {
          stage,
          progress,
          position: interpolate(points[stage.from], points[stage.to], progress)
        };
      }
      cursor = next;
    }
    return { stage: stages[stages.length - 1], progress: 1, position: points.aircraft };
  }

  function createPassenger(index) {
    const element = document.createElement("div");
    element.className = "passenger";
    element.dataset.id = `P${String(index + 1).padStart(3, "0")}`;
    map.appendChild(element);
    return { index, element, delay: index * 0.45 };
  }

  function createQueueDots() {
    [["checkin", 15], ["security", 12], ["immigration", 13], ["boarding", 10]].forEach(([queue, count]) => {
      for (let index = 0; index < count; index += 1) {
        const dot = document.createElement("div");
        const position = queuePosition(queue, index);
        dot.className = "queue-dot";
        dot.style.left = `${position.x + 7}px`;
        dot.style.top = `${position.y + 7}px`;
        map.appendChild(dot);
      }
    });
  }

  function resetProgressBars() {
    progressBars.forEach((bar) => { bar.style.width = "0%"; });
  }

  function updateProgress(stage, progress, passengerIndex) {
    if (!stage.progress) return;
    const barIndex = stage.progress[passengerIndex % stage.progress.length];
    if (progressBars[barIndex]) progressBars[barIndex].style.width = `${Math.round(progress * 100)}%`;
  }

  function field(card, name) {
    return card.querySelector(`[data-field="${name}"]`);
  }

  function setResourceStatus(card, status) {
    const className = status.toLowerCase();
    const statusField = field(card, "status");
    card.classList.remove("busy", "overloaded", "closed");
    statusField.classList.remove("busy", "overloaded", "closed");
    if (className !== "open") {
      card.classList.add(className);
      statusField.classList.add(className);
    }
    statusField.textContent = status;
  }

  function isClosed(resource, elapsed) {
    if (resource.opensAfter && elapsed < resource.opensAfter) return true;
    if (resource.closesAfter && elapsed > resource.closesAfter) return true;
    return false;
  }

  function buildResourceState(resource, passengerStates, elapsed) {
    const card = resourceCards.find((item) => item.dataset.resource === resource.id);
    const closed = isClosed(resource, elapsed);
    const queuedPassengers = passengerStates.filter((item) => (
      item.state.stage.queue === resource.queue && item.passenger.index % resource.count === resource.index
    ));
    const activePassenger = passengerStates.find((item) => (
      item.state.stage.name === resource.serviceStage && item.passenger.index % resource.count === resource.index
    ));
    const queueLength = closed ? 0 : queuedPassengers.length;
    const remaining = activePassenger ? Math.max(0, resource.serviceMinutes * (1 - activePassenger.state.progress)) : 0;
    const averageWait = closed ? 0 : resource.baseWait + queueLength * resource.waitFactor + (activePassenger ? 0.5 : 0);
    let status = "OPEN";

    if (closed) {
      status = "CLOSED";
    } else if (queueLength >= resource.overloadAt) {
      status = "OVERLOADED";
    } else if (activePassenger || queueLength >= resource.busyAt) {
      status = "BUSY";
    }

    return {
      card,
      resource,
      passenger: activePassenger ? activePassenger.passenger.element.dataset.id : "None",
      queueLength,
      averageWait,
      remaining,
      status,
    };
  }

  function updateResourceCards(passengerStates, elapsed) {
    const states = resources.map((resource) => buildResourceState(resource, passengerStates, elapsed));
    states.forEach((state) => {
      if (!state.card) return;
      field(state.card, "passenger").textContent = state.passenger;
      field(state.card, "queue").textContent = state.queueLength;
      field(state.card, "wait").textContent = `${state.averageWait.toFixed(1)} min`;
      field(state.card, "remaining").textContent = `${state.remaining.toFixed(1)} min`;
      setResourceStatus(state.card, state.status);
    });
    updateBottleneckAlert(states);
  }

  function updateBottleneckAlert(states) {
    const overloaded = states.filter((state) => state.status === "OVERLOADED");
    const busy = states.filter((state) => state.status === "BUSY");
    const closed = states.filter((state) => state.status === "CLOSED");
    let message = "NORMAL FLOW";
    let level = "open";

    if (overloaded.length > 0) {
      const worst = overloaded.sort((left, right) => right.queueLength - left.queueLength)[0];
      message = `BOTTLENECK: ${worst.resource.label} | Queue ${worst.queueLength}`;
      level = "overloaded";
    } else if (busy.length > 0) {
      const busiest = busy.sort((left, right) => right.queueLength - left.queueLength)[0];
      message = `BUSY: ${busiest.resource.label} | Queue ${busiest.queueLength}`;
      level = "busy";
    } else if (closed.length > 0) {
      message = `${closed.length} RESOURCE CLOSED`;
      level = "closed";
    }

    bottleneckAlert.classList.remove("busy", "overloaded", "closed");
    if (level !== "open") bottleneckAlert.classList.add(level);
    bottleneckAlert.textContent = message;
  }

  function formatTime(totalMinutes) {
    const dayMinutes = 24 * 60;
    const normalized = ((Math.round(totalMinutes) % dayMinutes) + dayMinutes) % dayMinutes;
    const hours = String(Math.floor(normalized / 60)).padStart(2, "0");
    const minutes = String(normalized % 60).padStart(2, "0");
    return `${hours}:${minutes}`;
  }

  function flightStatusFor(flight, elapsed) {
    const cycle = (elapsed + (flight.phaseOffset || 0)) % 116;
    const hasDelay = flight.delayMinutes > 0;
    let status = "ON TIME";
    let estimatedMinutes = flight.scheduledMinutes;

    if (hasDelay && cycle >= flight.delayStart && cycle < flight.boardingStart) {
      status = "DELAYED";
      estimatedMinutes += flight.delayMinutes;
    } else if (cycle >= flight.departedStart) {
      status = "DEPARTED";
      estimatedMinutes += flight.delayMinutes;
    } else if (cycle >= flight.gateClosedStart) {
      status = "GATE CLOSED";
      estimatedMinutes += flight.delayMinutes;
    } else if (cycle >= flight.boardingStart) {
      status = "BOARDING";
      estimatedMinutes += flight.delayMinutes;
    }

    return { status, estimatedMinutes };
  }

  function flightStatusClass(status) {
    return status.toLowerCase().replaceAll(" ", "-");
  }

  function updateFlightBoard(elapsed) {
    const tick = Math.floor(elapsed * 2);
    if (tick === lastFlightTick) return;
    lastFlightTick = tick;
    flightClock.textContent = formatTime(9 * 60 + elapsed);
    flightBoardBody.innerHTML = flights.map((flight) => {
      const state = flightStatusFor(flight, elapsed);
      const className = flightStatusClass(state.status);
      return `
        <tr class="flight-row ${className}">
          <td>${flight.flight}</td>
          <td>${flight.destination}</td>
          <td>${flight.gate}</td>
          <td>${formatTime(flight.scheduledMinutes)}</td>
          <td>${formatTime(state.estimatedMinutes)}</td>
          <td><span class="flight-status-badge ${className}">${state.status}</span></td>
        </tr>
      `;
    }).join("");
  }

  createQueueDots();
  const passengers = Array.from({ length: passengerCount }, (_, index) => createPassenger(index));
  const startedAt = performance.now();

  function animate(now) {
    const elapsed = ((now - startedAt) / 1000) * speed;
    const passengerStates = [];
    resetProgressBars();
    passengers.forEach((passenger) => {
      const localTime = elapsed - passenger.delay;
      if (localTime < 0) {
        passenger.element.style.opacity = "0";
        return;
      }
      const state = phaseFor(localTime, passenger.index);
      passenger.element.style.opacity = "1";
      passenger.element.style.transform = `translate(${state.position.x}px, ${state.position.y}px)`;
      passenger.element.classList.toggle("boarded", ["aircraft", "boarded"].includes(state.stage.name));
      passengerStates.push({ passenger, state });
      updateProgress(state.stage, state.progress, passenger.index);
    });
    updateResourceCards(passengerStates, elapsed);
    updateFlightBoard(elapsed);
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
</script>
</body>
</html>
"""


def render_professional_airport_control_center_page() -> None:
    """Render the professional airport control center page."""

    render_professional_streamlit_css()
    delay_data = load_professional_delay_data()
    model = load_professional_delay_model()
    render_delay_prediction_section(delay_data, model)

    components.html(build_professional_control_center_html(), height=1380, scrolling=True)
