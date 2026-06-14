"""Plotly chart builders for every tab of the dashboard.

All figures use the ``plotly_dark`` template and a shared colour palette so the
app looks consistent. Each function returns a :class:`plotly.graph_objects.Figure`
and never touches Streamlit directly, keeping the visual layer testable and
decoupled from the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data_pipeline import NUMERIC_FEATURES, TARGET_COLUMN

TEMPLATE = "plotly_dark"
ACCENT = "#38bdf8"
PALETTE = ["#38bdf8", "#a78bfa", "#f472b6", "#34d399"]
STATUS_COLORS = {"waiting": "#fbbf24", "serving": "#34d399", "done": "#60a5fa"}
TRANSPARENT = "rgba(0,0,0,0)"


def _base_layout(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    """Apply shared layout defaults (template, transparent background, margins)."""
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(color="#e6edf3"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Tab 1 — Dataset & preprocessing
# --------------------------------------------------------------------------- #
def missing_value_heatmap(data: pd.DataFrame) -> go.Figure:
    """Heatmap of missing cells (yellow = missing) across rows and columns."""
    mask = data.isna().astype(int)
    fig = go.Figure(go.Heatmap(
        z=mask.values,
        x=mask.columns,
        y=list(range(len(mask))),
        colorscale=[[0, "#161b26"], [1, "#fbbf24"]],
        showscale=False,
    ))
    fig.update_yaxes(showticklabels=False, title="Rows")
    return _base_layout(fig, title="Missing Values (yellow = missing)")


def delay_rate_bar(data: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Bar chart of delay rate (%) grouped by a categorical column."""
    rate = (data.groupby(column)[TARGET_COLUMN].mean() * 100).sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=rate.index, y=rate.values, marker_color=ACCENT,
        text=[f"{v:.1f}%" for v in rate.values], textposition="outside",
    ))
    fig.update_yaxes(title="Delay rate (%)", range=[0, max(rate.values) * 1.2 if len(rate) else 100])
    return _base_layout(fig, title=title)


def passenger_distribution(data: pd.DataFrame) -> go.Figure:
    """Histogram of the passenger-count distribution."""
    fig = px.histogram(data, x="passenger_count", nbins=40, color_discrete_sequence=[PALETTE[1]])
    fig.update_traces(marker_line_color="#0e1117", marker_line_width=0.5)
    fig.update_layout(bargap=0.05)
    return _base_layout(fig, title="Passenger Count Distribution")


def correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    """Correlation heatmap across the numeric features and the target."""
    corr = data[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    return _base_layout(fig, height=420, title="Numeric Feature Correlation")


# --------------------------------------------------------------------------- #
# Tab 2 — Model training & comparison
# --------------------------------------------------------------------------- #
def model_comparison_bar(metrics: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all models across all four metrics."""
    melted = metrics.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
                          var_name="metric", value_name="score")
    fig = px.bar(melted, x="metric", y="score", color="model", barmode="group",
                 color_discrete_sequence=PALETTE, text_auto=".3f")
    fig.update_yaxes(range=[0, 1.05], title="Score")
    fig.update_xaxes(title=None)
    return _base_layout(fig, title="Model Performance Comparison")


def confusion_matrix_heatmap(matrix: np.ndarray, model_name: str) -> go.Figure:
    """Annotated confusion-matrix heatmap for one model."""
    labels = ["On Time", "Delayed"]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, colorscale="Blues",
        text=matrix, texttemplate="%{text}", textfont=dict(size=18),
        showscale=False,
    ))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return _base_layout(fig, title=f"Confusion Matrix — {model_name}")


def feature_importance_bar(importance: pd.DataFrame, top_n: int = 12,
                           title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart of the top-N most important features."""
    top = importance.head(top_n).iloc[::-1]
    colors = [ACCENT if v >= 0 else "#f472b6" for v in top["importance"]]
    fig = go.Figure(go.Bar(
        x=top["magnitude"], y=top["feature"], orientation="h", marker_color=colors,
    ))
    fig.update_xaxes(title="Importance")
    return _base_layout(fig, height=420, title=title)


# --------------------------------------------------------------------------- #
# Tab 3 — Simulation analytics
# --------------------------------------------------------------------------- #
def queue_timeline_animated(timeline: pd.DataFrame) -> go.Figure:
    """Animated line chart of queue length per stage, revealed over time.

    Built with explicit Plotly frames: each frame extends the per-stage lines up
    to the current time point, producing a drawing-on effect with a play/slider.
    """
    stages = list(timeline["stage"].unique())
    times = sorted(timeline["time"].unique())
    color_map = {stage: PALETTE[i] for i, stage in enumerate(stages)}

    def traces_up_to(t_index: int) -> list[go.Scatter]:
        sliced = []
        for stage in stages:
            stage_df = timeline[timeline["stage"] == stage].sort_values("time")
            stage_df = stage_df[stage_df["time"] <= times[t_index]]
            sliced.append(go.Scatter(
                x=stage_df["time"], y=stage_df["queue_length"], mode="lines",
                name=stage, line=dict(color=color_map[stage], width=3),
            ))
        return sliced

    fig = go.Figure(data=traces_up_to(0))
    fig.frames = [go.Frame(data=traces_up_to(i), name=str(times[i])) for i in range(len(times))]

    max_q = timeline["queue_length"].max()
    fig.update_xaxes(title="Simulation time (min)", range=[0, max(times) if times else 1])
    fig.update_yaxes(title="Queue length", range=[0, max_q + 2 if max_q else 5])
    _base_layout(fig, title="Queue Length Over Time")
    _attach_player(fig, [str(t) for t in times], frame_ms=120, label="Time")
    return fig


def stage_wait_bar(stage_waits: pd.DataFrame) -> go.Figure:
    """Bar chart of average wait time per stage."""
    fig = go.Figure(go.Bar(
        x=stage_waits["stage"], y=stage_waits["avg_wait"],
        marker_color=PALETTE[:3], text=[f"{v:.1f}" for v in stage_waits["avg_wait"]],
        textposition="outside",
    ))
    fig.update_yaxes(title="Avg wait (min)")
    return _base_layout(fig, title="Average Wait per Stage")


def journey_time_histogram(passengers: pd.DataFrame) -> go.Figure:
    """Histogram of total passenger journey times."""
    fig = px.histogram(passengers, x="journey_time", nbins=30, color_discrete_sequence=[PALETTE[3]])
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(title="Journey time (min)")
    return _base_layout(fig, title="Journey Time Distribution")


def arrival_vs_wait_scatter(passengers: pd.DataFrame) -> go.Figure:
    """Scatter of arrival time against total wait, coloured by wait length."""
    fig = px.scatter(passengers, x="arrival_time", y="total_wait", color="total_wait",
                     color_continuous_scale="Plasma", size="journey_time", size_max=14)
    fig.update_xaxes(title="Arrival time (min)")
    fig.update_yaxes(title="Total wait (min)")
    return _base_layout(fig, title="Arrival Time vs Total Wait")


# --------------------------------------------------------------------------- #
# Tab 4 — Live animated simulation
# --------------------------------------------------------------------------- #
def _attach_player(fig: go.Figure, frame_names: list[str], frame_ms: int, label: str) -> None:
    """Attach Plotly play/pause buttons and a slider driving ``frame_names``."""
    def frame_args(duration: int) -> dict:
        return {"frame": {"duration": duration, "redraw": True},
                "mode": "immediate", "transition": {"duration": 0}}

    fig.update_layout(
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.02, "y": 1.12, "xanchor": "left",
            "bgcolor": "#161b26", "bordercolor": ACCENT,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, frame_args(frame_ms)]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], frame_args(0)]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.12, "len": 0.85, "y": 1.06,
            "currentvalue": {"prefix": f"{label}: ", "font": {"color": ACCENT}},
            "steps": [{"label": name, "method": "animate",
                       "args": [[name], frame_args(0)]} for name in frame_names],
        }],
    )


def airport_animation(frames_df: pd.DataFrame, frame_ms: int = 200) -> go.Figure:
    """Animated map of passengers flowing through the three airport zones.

    Renders zone backgrounds as shapes, labels them with annotations, and animates
    passenger dots (coloured by status) across frames using native Plotly controls.

    Args:
        frames_df: Output of :func:`simulation.animation_frames`.
        frame_ms: Milliseconds per frame (controls playback speed).

    Returns:
        A fully configured animated Plotly figure.
    """
    from .simulation import ZONE_X

    zones = [("check_in", "🛄 Check-in", "#1e3a5f"),
             ("security", "🛂 Security", "#3a2f5f"),
             ("boarding", "🛫 Boarding", "#5f1e3a"),
             ("done", "✅ Departed", "#1e5f3a")]
    frame_ids = sorted(frames_df["frame"].unique()) if not frames_df.empty else [0]

    def scatter_for(frame_id: int) -> go.Scatter:
        sub = frames_df[frames_df["frame"] == frame_id]
        return go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers",
            marker=dict(size=11, color=[STATUS_COLORS[s] for s in sub["status"]],
                        line=dict(width=1, color="#0e1117")),
            text=[f"Passenger {p} — {s}" for p, s in zip(sub["passenger_id"], sub["status"])],
            hoverinfo="text", showlegend=False,
        )

    fig = go.Figure(data=[scatter_for(frame_ids[0])])
    fig.frames = [go.Frame(data=[scatter_for(f)], name=str(f)) for f in frame_ids]

    # Zone backgrounds + labels.
    for zone, label, color in zones:
        cx = ZONE_X[zone]
        fig.add_shape(type="rect", x0=cx - 1.1, x1=cx + 1.1, y0=0, y1=10,
                      fillcolor=color, opacity=0.35, line_width=0, layer="below")
        fig.add_annotation(x=cx, y=10.4, text=label, showarrow=False,
                           font=dict(size=14, color="#e6edf3"))

    # Status legend (manual, since markers carry per-point colour).
    for status, color in STATUS_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=11, color=color), name=status.title()))

    fig.update_xaxes(range=[0, 12], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.5, 11], showgrid=False, zeroline=False, showticklabels=False)
    _base_layout(fig, height=520)
    fig.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
    _attach_player(fig, [str(f) for f in frame_ids], frame_ms=frame_ms, label="Frame")
    return fig


# --------------------------------------------------------------------------- #
# Tab 5 — Prediction tool
# --------------------------------------------------------------------------- #
def probability_gauge(probability: float) -> go.Figure:
    """Gauge chart showing the delay probability as a 0–100% dial."""
    pct = probability * 100
    color = "#34d399" if pct < 40 else "#fbbf24" if pct < 65 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#e6edf3"},
            "bar": {"color": color},
            "bgcolor": TRANSPARENT,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(52,211,153,0.18)"},
                {"range": [40, 65], "color": "rgba(251,191,36,0.18)"},
                {"range": [65, 100], "color": "rgba(248,113,113,0.18)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
        },
        title={"text": "Delay Probability"},
    ))
    return _base_layout(fig, height=320)


def contribution_bar(importance: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Horizontal bar of the features pushing a prediction toward/away from delay.

    Positive (toward delay) bars are red, protective bars are green — a simple,
    SHAP-style explanation derived from the model's feature weights/importances.
    """
    top = importance.head(top_n).iloc[::-1]
    colors = ["#f87171" if v >= 0 else "#34d399" for v in top["importance"]]
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"], orientation="h", marker_color=colors,
    ))
    fig.update_xaxes(title="← protective | pushes toward delay →")
    return _base_layout(fig, height=360, title="What's driving this prediction")
