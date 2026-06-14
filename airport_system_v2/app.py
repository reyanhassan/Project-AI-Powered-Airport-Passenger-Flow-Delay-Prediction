"""AI-Powered Airport Passenger Flow & Delay Prediction — Streamlit app.

A single-page, five-tab control centre that ties together a synthetic airline
dataset, three trained ML classifiers, a SimPy passenger-flow simulation, a live
animated airport map, and an interactive delay-prediction tool. The ML model is
wired into the simulation's flight board, so flight statuses are genuine model
predictions rather than hardcoded values.

Run with:  ``streamlit run app.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import control_center as cc
from src import data_pipeline as dp
from src import ml_pipeline as ml
from src import simulation as sim
from src import visualizations as viz

# --------------------------------------------------------------------------- #
# Page configuration & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Airport Flow & Delay AI",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      div[data-testid="stMetric"] {
          background: #161b26; border: 1px solid #232a36;
          border-radius: 12px; padding: 14px 16px;
      }
      .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 8px 18px; }
      .badge { display:inline-block; padding:6px 18px; border-radius:999px;
               font-weight:700; font-size:1.1rem; }
      .badge-ok { background:rgba(52,211,153,0.15); color:#34d399; border:1px solid #34d399; }
      .badge-bad{ background:rgba(248,113,113,0.15); color:#f87171; border:1px solid #f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached data & model loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    """Load (or generate on first run) the raw airline-delay dataset."""
    return dp.get_or_create_dataset()


@st.cache_data(show_spinner=False)
def load_clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned dataset for a given raw input."""
    return dp.clean_dataset(raw)


@st.cache_resource(show_spinner=False)
def load_saved_model():
    """Load the persisted best model (cached as a resource), or ``None``."""
    return ml.load_model()


@st.cache_data(show_spinner=False)
def run_cached_simulation(cfg_tuple: tuple) -> dict:
    """Run the simulation for a hashable config tuple and return chart-ready data."""
    config = sim.SimulationConfig(*cfg_tuple)
    passengers = sim.run_simulation(config)
    return {
        "stats": sim.compute_statistics(passengers),
        "summary": sim.passengers_dataframe(passengers),
        "stage_waits": sim.stage_wait_dataframe(passengers),
        "queue_timeline": sim.queue_timeline(passengers),
        "frames": sim.animation_frames(passengers),
    }


def get_active_model():
    """Return the in-session trained model if present, else the saved one."""
    if "training" in st.session_state:
        return st.session_state["training"]["best_model"]
    return load_saved_model()


# --------------------------------------------------------------------------- #
# Tab 1 — Dataset & preprocessing
# --------------------------------------------------------------------------- #
def render_dataset_tab() -> None:
    """Render dataset preview, EDA charts and before/after cleaning stats."""
    st.subheader("Dataset & Preprocessing")
    st.caption("A realistic 3,000-flight airline delay dataset is generated automatically. "
               "Upload your own CSV with the same schema to override it.")

    upload = st.file_uploader("Upload airline_delay.csv (optional)", type=["csv"])
    raw = pd.read_csv(upload) if upload is not None else load_raw_dataset()
    cleaned = dp.clean_dataset(raw)

    before, after = dp.summarize_dataset(raw), dp.summarize_dataset(cleaned)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{after['rows']:,}", delta=f"{after['rows'] - before['rows']:,}")
    c2.metric("Missing cells removed", f"{before['missing_cells']:,}", delta=f"-{before['missing_cells']:,}",
              delta_color="inverse")
    c3.metric("Duplicate rows removed", f"{before['duplicate_rows']:,}")
    c4.metric("Overall delay rate", f"{after['delay_rate'] * 100:.1f}%")

    st.markdown("##### Raw data preview")
    st.dataframe(raw.head(12), use_container_width=True, hide_index=True)

    with st.expander("Preprocessing steps applied", expanded=False):
        st.markdown(
            "- Dropped duplicate flight records\n"
            "- Filled missing **categorical** values with the column mode\n"
            "- Filled missing **numeric** values with the column median and cast to integers\n"
            "- Coerced the `delayed` target to a clean 0/1 label"
        )

    g1, g2 = st.columns(2)
    g1.plotly_chart(viz.missing_value_heatmap(raw), use_container_width=True)
    g2.plotly_chart(viz.passenger_distribution(cleaned), use_container_width=True)

    g3, g4 = st.columns(2)
    g3.plotly_chart(viz.delay_rate_bar(cleaned, "airline", "Delay Rate by Airline"),
                    use_container_width=True)
    g4.plotly_chart(viz.delay_rate_bar(cleaned, "weather_condition", "Delay Rate by Weather"),
                    use_container_width=True)

    st.plotly_chart(viz.correlation_heatmap(cleaned), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 2 — Model training & comparison
# --------------------------------------------------------------------------- #
def render_training_tab() -> None:
    """Render the model-training controls, metrics and comparison charts."""
    st.subheader("ML Model Training & Comparison")
    st.caption("Train Logistic Regression, Decision Tree and Random Forest, then compare them. "
               "The best model by F1 score is saved to `models/best_model.joblib`.")

    if st.button("🚀 Train all models", type="primary"):
        raw = load_raw_dataset()
        X, y = dp.split_features_target(dp.clean_dataset(raw))

        progress = st.progress(0.0, text="Preparing…")
        results = ml.train_models(X, y, progress_callback=lambda f, l: progress.progress(f, text=l))
        ml.save_model(results["best_model"])
        load_saved_model.clear()  # refresh the cached resource
        st.session_state["training"] = results
        progress.empty()
        st.success(f"Training complete — best model: **{results['best_model_name']}** "
                   f"(saved to models/best_model.joblib)")

    if "training" not in st.session_state:
        st.info("Click **Train all models** to begin.")
        return

    results = st.session_state["training"]
    metrics = results["metrics"]

    st.markdown("##### Metrics")
    st.dataframe(
        metrics.style.format({m: "{:.3f}" for m in ml.METRIC_COLUMNS})
        .highlight_max(subset=ml.METRIC_COLUMNS, color="#1e5f3a"),
        use_container_width=True, hide_index=True,
    )

    st.plotly_chart(viz.model_comparison_bar(metrics), use_container_width=True)

    best = results["best_model_name"]
    st.markdown(f"##### Best model: `{best}`")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.confusion_matrix_heatmap(results["confusion_matrices"][best], best),
                    use_container_width=True)
    c2.plotly_chart(viz.feature_importance_bar(results["rf_importance"],
                    title="Random Forest Feature Importance"), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 3 — Simulation analytics
# --------------------------------------------------------------------------- #
def _simulation_controls(key_prefix: str) -> sim.SimulationConfig:
    """Render the shared sidebar-style simulation sliders and return a config."""
    c1, c2, c3 = st.columns(3)
    num = c1.slider("Passengers", 10, 300, 90, key=f"{key_prefix}_num")
    interval = c1.slider("Avg arrival interval (min)", 0.3, 5.0, 1.5, 0.1, key=f"{key_prefix}_int")
    counters = c2.slider("Check-in counters", 1, 10, 3, key=f"{key_prefix}_ci")
    lanes = c2.slider("Security lanes", 1, 10, 3, key=f"{key_prefix}_se")
    gates = c3.slider("Boarding gates", 1, 8, 2, key=f"{key_prefix}_bo")
    seed = c3.number_input("Random seed", 0, 9999, 42, key=f"{key_prefix}_seed")
    return sim.SimulationConfig(
        num_passengers=num, arrival_interval=interval, check_in_counters=counters,
        security_lanes=lanes, boarding_gates=gates, random_seed=int(seed),
    )


def _config_tuple(cfg: sim.SimulationConfig) -> tuple:
    """Flatten a config into a hashable tuple for caching."""
    return (cfg.num_passengers, cfg.arrival_interval, cfg.check_in_counters,
            cfg.security_lanes, cfg.boarding_gates, cfg.random_seed)


def render_simulation_tab() -> None:
    """Render the SimPy analytics: summary metrics and four Plotly charts."""
    st.subheader("Passenger Flow Simulation")
    st.caption("Model an airport terminal with SimPy. Tune resources and observe the impact "
               "on queues and journey times.")

    cfg = _simulation_controls("sim")
    data = run_cached_simulation(_config_tuple(cfg))
    stats = data["stats"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total passengers", stats["total_passengers"])
    c2.metric("Avg total wait", f"{stats['avg_total_wait']:.1f} min")
    c3.metric("Busiest stage", stats["busiest_stage"])
    c4.metric("Congestion", stats["congestion_level"])

    st.plotly_chart(viz.queue_timeline_animated(data["queue_timeline"]), use_container_width=True)

    g1, g2 = st.columns(2)
    g1.plotly_chart(viz.stage_wait_bar(data["stage_waits"]), use_container_width=True)
    g2.plotly_chart(viz.journey_time_histogram(data["summary"]), use_container_width=True)
    st.plotly_chart(viz.arrival_vs_wait_scatter(data["summary"]), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 4 — Live animated control center (ML-driven flight board)
# --------------------------------------------------------------------------- #
def render_live_tab() -> None:
    """Render the animated airport control center, wired to the trained model."""
    st.subheader("Live Animated Simulation")

    if not ml.model_exists():
        st.warning("🔒 Train models first in the **ML Training** tab to power the control center.")
        return

    st.caption("A live terminal control room: passenger dots flow through check-in → security → "
               "immigration → boarding while the Resource Operations Monitor and congestion-aware "
               "flight board update in real time. Clearance depends on passenger readiness, "
               "terminal queues, gate availability and ML risk. Use the in-map speed and step controls.")

    c1, c2 = st.columns([3, 1])
    all_flights = [t["flight"] for t in cc.FLIGHT_TEMPLATES]
    selected = c1.multiselect(
        "Priority flights for the simulation", options=all_flights, default=all_flights,
        help="Selected flights appear first; any empty gates are auto-filled so all six gates keep operating.",
    )
    if not selected:
        st.info("No priority selected, so the control center will auto-fill all six gates.")
        selected = all_flights
    display_flights = (selected + [flight for flight in all_flights if flight not in selected])[:len(all_flights)]
    seed = c2.number_input("Scenario seed", 0, 9999, 42, key="live_seed",
                           help="Reshuffle the flights scored by the model.")

    rush_options = {
        "Light flow": 0.75,
        "Normal flow": 1.0,
        "Busy terminal": 1.25,
        "Peak rush": 1.55,
    }
    with st.expander("Scenario controls: force delay and terminal pressure", expanded=True):
        s1, s2, s3, s4 = st.columns([1.1, 1.1, 1.5, 1])
        force_delay = s1.toggle("Force flight delay", value=False, key="live_force_delay")
        delayed_flight = s2.selectbox("Flight", options=display_flights, key="live_delay_flight")
        delay_reason = s3.selectbox(
            "Delay reason",
            options=list(cc.DELAY_REASON_BOTTLENECK.keys()),
            key="live_delay_reason",
        )
        delay_minutes = s4.slider("Delay min", 5, 90, 24, 1, key="live_delay_minutes")
        rush_label = st.select_slider(
            "Terminal pressure",
            options=list(rush_options.keys()),
            value="Normal flow",
            help="Higher pressure creates more passengers, tighter arrival spacing, and stricter clearance decisions.",
            key="live_rush_level",
        )
        if force_delay:
            st.info(
                f"{delayed_flight} will be held for about {delay_minutes} minutes because of "
                f"{delay_reason}. Bottleneck: {cc.DELAY_REASON_BOTTLENECK[delay_reason]}."
            )

    scenario = {
        "force_delay": force_delay,
        "flight": delayed_flight,
        "reason": delay_reason,
        "delay_minutes": int(delay_minutes),
        "rush_multiplier": rush_options[rush_label],
    }

    model = get_active_model()
    flight_rows = cc.build_flight_rows(model, seed=int(seed), selected=display_flights, scenario=scenario)

    delayed = sum(1 for f in flight_rows if f["delayMinutes"] > 0 or f["delayProbability"] >= 0.55)
    avg_prob = np.mean([f["delayProbability"] for f in flight_rows]) if flight_rows else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Planes in simulation", len(flight_rows))
    m2.metric("Predicted delayed", delayed)
    m3.metric("Avg delay probability", f"{avg_prob * 100:.0f}%")
    m4.metric("Terminal pressure", rush_label)

    # Taller map needs more height when more planes (and longer boarding lanes) are shown.
    components.html(cc.build_control_center_html(flight_rows), height=1660, scrolling=True)


# --------------------------------------------------------------------------- #
# Tab 5 — Prediction tool
# --------------------------------------------------------------------------- #
def render_prediction_tab() -> None:
    """Render the single-flight delay-prediction form and explanation."""
    st.subheader("ML Delay Prediction Tool")

    model = get_active_model()
    if model is None:
        st.warning("🔒 Train models first in the **ML Training** tab to enable predictions.")
        return

    st.caption("Enter a flight's details to predict whether it will be delayed.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        airline = c1.selectbox("Airline", dp.AIRLINES)
        origin = c1.selectbox("Origin airport", dp.AIRPORTS)
        destination = c1.selectbox("Destination airport", dp.AIRPORTS, index=1)
        flight_day = c2.selectbox("Flight day", dp.DAYS)
        weather = c2.selectbox("Weather", dp.WEATHER_CONDITIONS)
        scheduled_hour = c2.slider("Scheduled hour", 5, 23, 8)
        passenger_count = c3.slider("Passenger count", 55, 285, 165)
        previous_delay = c3.slider("Previous delay (min)", 0, 180, 10)
        security_wait = c3.slider("Security wait (min)", 4, 90, 15)
        gate_changes = c1.slider("Gate changes", 0, 4, 0)
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    features = {
        "airline": airline, "origin_airport": origin, "destination_airport": destination,
        "flight_day": flight_day, "weather_condition": weather, "scheduled_hour": scheduled_hour,
        "passenger_count": passenger_count, "previous_delay_minutes": previous_delay,
        "security_wait_minutes": security_wait, "gate_changes": gate_changes,
    }
    label, probability = ml.predict_delay(model, features)

    left, right = st.columns([1, 1])
    with left:
        if label == 1:
            st.markdown('<span class="badge badge-bad">⚠ Predicted: DELAYED</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-ok">✓ Predicted: ON TIME</span>',
                        unsafe_allow_html=True)
        st.metric("Delay probability", f"{probability * 100:.1f}%")
    right.plotly_chart(viz.probability_gauge(probability), use_container_width=True)

    st.plotly_chart(viz.contribution_bar(ml.compute_feature_importance(model)),
                    use_container_width=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Compose the header, sidebar and the five primary tabs."""
    st.title("🛫 AI-Powered Airport Passenger Flow & Delay Prediction")

    with st.sidebar:
        st.header("Control Center")
        st.markdown("A unified tool for **delay prediction** and **terminal flow simulation**.")
        st.divider()
        st.markdown("**Model status**")
        if ml.model_exists():
            name = st.session_state.get("training", {}).get("best_model_name", "saved model")
            st.success(f"Trained ✓ ({name})")
        else:
            st.error("Not trained yet")
        st.divider()
        st.caption("Built with Streamlit · scikit-learn · SimPy · Plotly")

    tabs = st.tabs([
        "📊 Dataset & Preprocessing",
        "🤖 ML Training & Comparison",
        "🎛 Flow Simulation",
        "🎬 Live Simulation",
        "🔮 Prediction Tool",
    ])
    with tabs[0]:
        render_dataset_tab()
    with tabs[1]:
        render_training_tab()
    with tabs[2]:
        render_simulation_tab()
    with tabs[3]:
        render_live_tab()
    with tabs[4]:
        render_prediction_tab()


if __name__ == "__main__":
    main()
