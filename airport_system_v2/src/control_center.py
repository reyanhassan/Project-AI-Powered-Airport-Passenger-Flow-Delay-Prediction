"""Animated airport control-center component (HTML/CSS/JS).

This module builds a self-contained, embeddable control room: passenger dots
flow through eleven terminal zones (entrance → check-in → security → immigration
→ lounge → boarding → aircraft) via a smooth ``requestAnimationFrame`` loop, a
live Resource Operations Monitor tracks each counter/lane/gate, and a flight board
animates departure statuses.

The flight rows are produced by :func:`build_flight_rows`, which queries the
trained ``best_model.joblib`` so the board reflects genuine ML delay predictions
rather than scripted values. The visual layer is rendered through
``streamlit.components.v1.html`` and needs no Python on the client side.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import ml_pipeline as ml
from .data_pipeline import AIRLINES, AIRPORTS, DAYS, FEATURE_COLUMNS, WEATHER_CONDITIONS

# Friendly destination names for the flight board.
AIRPORT_DESTINATIONS = {
    "BKK": "Bangkok", "DOH": "Doha", "DXB": "Dubai", "ISB": "Islamabad",
    "IST": "Istanbul", "JFK": "New York", "KHI": "Karachi", "LHE": "Lahore",
    "LHR": "London", "SIN": "Singapore",
}

# Fixed visual templates (code, gate, schedule offset, animation phase, colour).
FLIGHT_TEMPLATES = [
    {"flight": "PK-302", "gate": "A1", "minute": 10, "phaseOffset": 0, "color": "#38bdf8"},
    {"flight": "AP-144", "gate": "A2", "minute": 25, "phaseOffset": 8, "color": "#f472b6"},
    {"flight": "SA-781", "gate": "B3", "minute": 40, "phaseOffset": 34, "color": "#a78bfa"},
    {"flight": "ER-219", "gate": "C1", "minute": 5, "phaseOffset": 8, "color": "#34d399"},
    {"flight": "GB-508", "gate": "A4", "minute": 20, "phaseOffset": 72, "color": "#facc15"},
    {"flight": "PK-419", "gate": "D2", "minute": 35, "phaseOffset": 99, "color": "#fb7185"},
]


def _risk_level(probability: float) -> str:
    """Map a delay probability to a Low/Medium/High operations risk label."""
    if probability >= 0.55:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def _estimate_delay_minutes(probability: float, previous_delay: float,
                            security_wait: float, gate_changes: int) -> int:
    """Estimate delay minutes from the model probability and operational inputs."""
    estimate = probability * 55 + previous_delay * 0.22 + security_wait * 0.18 + gate_changes * 5 - 18
    return max(0, int(round(estimate)))


def build_flight_rows(model, seed: int = 42, n: int = 6,
                      selected: list[str] | None = None) -> list[dict]:
    """Generate ML-driven flight-board rows for the animated control center.

    Each flight gets realistic random features, scored by the trained model. The
    predicted probability drives the estimated delay, risk level, delay reason,
    waiting passengers and bottleneck stage shown on the board.

    Args:
        model: A fitted model pipeline (or ``None`` to fall back to neutral rows).
        seed: Seed for reproducible flight features.
        n: Number of flights to generate when ``selected`` is not given.
        selected: Optional list of flight codes (e.g. ``["PK-302", "AP-144"]``) to
            include; when provided, only those templates are used.

    Returns:
        A list of JSON-serialisable flight-row dictionaries consumed by the JS map.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    if selected:
        templates = [t for t in FLIGHT_TEMPLATES if t["flight"] in selected]
    else:
        templates = FLIGHT_TEMPLATES[:max(1, min(n, len(FLIGHT_TEMPLATES)))]

    for i, template in enumerate(templates):
        origin = AIRPORTS[int(rng.integers(len(AIRPORTS)))]
        destination = AIRPORTS[int(rng.integers(len(AIRPORTS)))]
        # Realistic, right-skewed operational inputs so the board shows a natural
        # mix of on-time and delayed flights rather than uniformly high risk.
        weather = rng.choice(WEATHER_CONDITIONS, p=[0.52, 0.22, 0.15, 0.07, 0.04])
        previous_delay = int(min(150, max(0, rng.exponential(14))))
        security_wait = int(min(80, max(4, rng.normal(16, 8))))
        gate_changes = int(rng.choice([0, 1, 2, 3], p=[0.68, 0.2, 0.08, 0.04]))
        features = {
            "airline": AIRLINES[int(rng.integers(len(AIRLINES)))],
            "origin_airport": origin,
            "destination_airport": destination,
            "flight_day": DAYS[int(rng.integers(len(DAYS)))],
            "weather_condition": str(weather),
            "scheduled_hour": int(rng.integers(6, 21)),
            "passenger_count": int(min(285, max(55, rng.normal(165, 38)))),
            "previous_delay_minutes": previous_delay,
            "security_wait_minutes": security_wait,
            "gate_changes": gate_changes,
        }

        if model is not None:
            _, probability = ml.predict_delay(model, features)
        else:
            probability = 0.3

        estimated_delay = _estimate_delay_minutes(
            probability, features["previous_delay_minutes"],
            features["security_wait_minutes"], features["gate_changes"],
        )
        if probability >= 0.55 and estimated_delay == 0:
            estimated_delay = 8
        elif probability < 0.35:
            estimated_delay = 0

        bottleneck = ["Security", "Check-in", "Immigration", "Boarding"][i % 4]
        passengers_waiting = max(0, int(round(probability * 42 + estimated_delay * 0.45)))
        if estimated_delay == 0 and probability < 0.45:
            reason, bottleneck = "None", "None"
            passengers_waiting //= 3
        elif probability >= 0.65:
            reason = "High Passenger Arrival Rate"
        elif features["gate_changes"] > 0:
            reason = "Gate Change"
        elif features["previous_delay_minutes"] > 20:
            reason = "Previous Flight Delay"
        else:
            reason = f"{bottleneck} Queue"

        scheduled_minutes = features["scheduled_hour"] * 60 + int(template["minute"])
        rows.append({
            "flight": template["flight"],
            "destination": AIRPORT_DESTINATIONS.get(destination, destination),
            "gate": template["gate"],
            "scheduledMinutes": scheduled_minutes,
            "delayMinutes": estimated_delay,
            "delayProbability": round(float(probability), 3),
            "riskLevel": _risk_level(probability),
            "delayReason": reason,
            "passengersWaiting": passengers_waiting,
            "bottleneckStage": bottleneck,
            "delayStart": 10 + i * 2,
            "boardingStart": 54 + i * 3,
            "gateClosedStart": 70 + i * 3,
            "departedStart": 82 + i * 4,
            "phaseOffset": int(template["phaseOffset"]),
            "color": template["color"],
        })

    return rows


def build_control_center_html(flight_rows: list[dict]) -> str:
    """Return the full HTML document for the animated control center.

    Args:
        flight_rows: Output of :func:`build_flight_rows`.

    Returns:
        A complete HTML/CSS/JS string ready for ``components.html``.
    """
    flights_json = json.dumps(flight_rows)
    return _HTML_TEMPLATE.replace("__FLIGHT_DATA__", flights_json)


# --------------------------------------------------------------------------- #
# Static HTML/CSS/JS template (token __FLIGHT_DATA__ replaced at build time).
# Kept as a plain string (not an f-string) so JS ``${}`` and CSS ``{}`` are safe.
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  :root {
    --bg: #030914; --panel: #071624; --panel-soft: #0b2031; --line: #1f78a8;
    --neon: #3bd7ff; --text: #ecf8ff; --muted: #8ea9bb; --yellow: #ffd166;
    --green: #34d399; --blue: #3b82f6; --red: #f87171; --busy: #fbbf24; --closed: #64748b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: Inter, Segoe UI, Arial, sans-serif; }
  .control-shell { min-height: 960px; padding: 18px;
    background:
      radial-gradient(circle at 18% 0%, rgba(59,215,255,0.18), transparent 26%),
      linear-gradient(rgba(59,215,255,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(59,215,255,0.04) 1px, transparent 1px), #030914;
    background-size: auto, 34px 34px, 34px 34px; }
  .topbar { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 14px; }
  .title { font-size: 34px; font-weight: 900; color: #f3fbff; }
  .subtitle { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .system-chip { padding: 10px 14px; border: 1px solid rgba(59,215,255,0.42); border-radius: 8px;
    color: var(--neon); background: rgba(7,22,36,0.9); font-weight: 900;
    box-shadow: 0 0 24px rgba(59,215,255,0.12); }
  .sim-control-strip { display: flex; justify-content: space-between; gap: 12px; align-items: center;
    border: 1px solid rgba(59,215,255,0.32); border-radius: 10px; background: rgba(7,22,36,0.86);
    padding: 10px 12px; margin-bottom: 14px; }
  .sim-control-title { color: #f3fbff; font-size: 13px; font-weight: 900; text-transform: uppercase; }
  .sim-control-buttons { display: flex; flex-wrap: wrap; gap: 8px; }
  .sim-button { border: 1px solid rgba(59,215,255,0.38); border-radius: 8px; background: rgba(3,9,20,0.64);
    color: var(--muted); cursor: pointer; font-size: 12px; font-weight: 900; padding: 8px 11px;
    transition: border-color 0.2s ease, color 0.2s ease, background 0.2s ease; }
  .sim-button.active { border-color: rgba(59,215,255,0.9); background: rgba(59,215,255,0.18);
    color: var(--neon); box-shadow: 0 0 16px rgba(59,215,255,0.16); }
  .sim-button:disabled { cursor: not-allowed; opacity: 0.45; }
  .airport-map { position: relative; height: 910px; border: 1px solid rgba(59,215,255,0.34);
    border-radius: 14px; background: linear-gradient(135deg, rgba(59,215,255,0.08), transparent 44%),
    rgba(7,22,36,0.94); overflow: hidden;
    box-shadow: inset 0 0 60px rgba(59,215,255,0.07), 0 18px 48px rgba(0,0,0,0.32); }
  .zone { position: absolute; border: 1px solid rgba(59,215,255,0.44); border-radius: 10px;
    background: rgba(11,32,49,0.88); padding: 10px; min-height: 78px; }
  .zone-title { color: var(--neon); font-size: 12px; font-weight: 900; text-transform: uppercase; }
  .zone-note { color: var(--muted); font-size: 11px; margin-top: 4px; }
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
  .apron { left: 24px; top: 742px; width: 1238px; height: 150px; }
  .desk { position: relative; border: 1px solid rgba(59,215,255,0.42); background: rgba(59,215,255,0.11);
    border-radius: 7px; padding: 7px; margin-top: 8px; color: #e8f8ff; font-size: 11px; font-weight: 800; overflow: hidden; }
  .desk .progress { position: absolute; left: 0; bottom: 0; height: 3px; width: 0%;
    background: linear-gradient(90deg, var(--green), var(--neon)); transition: width 0.28s ease; }
  .route { position: absolute; height: 3px; background: linear-gradient(90deg, transparent, var(--neon), transparent);
    opacity: 0.78; transform-origin: left center; }
  .route.main { left: 142px; top: 370px; width: 1110px; }
  .route.down { left: 460px; top: 514px; width: 520px; transform: rotate(22deg); }
  .aircraft-body { position: absolute; right: 18px; bottom: 22px; width: 112px; height: 40px;
    border-radius: 52px 12px 12px 52px; background: linear-gradient(90deg, #e8f8ff, #75bdf1); }
  .aircraft-body::before, .aircraft-body::after { content: ""; position: absolute; left: 43px;
    width: 48px; height: 10px; background: #75bdf1; }
  .aircraft-body::before { top: -10px; transform: rotate(-23deg); }
  .aircraft-body::after { bottom: -10px; transform: rotate(23deg); }
  /* Aircraft stands: one parking stand per flight, each holding exactly one plane. */
  .plane-stand { position: absolute; width: 176px; height: 104px; transform: translate(-50%, -50%);
    border: 1px dashed rgba(59,215,255,0.35); border-radius: 12px; background: rgba(3,9,20,0.42); z-index: 6; }
  .plane-stand .stand-gate { position: absolute; top: 6px; left: 9px; color: var(--muted);
    font-size: 10px; font-weight: 900; letter-spacing: 0.05em; text-transform: uppercase; }
  .plane-stand.delayed { border-color: rgba(248,113,113,0.6); background: rgba(248,113,113,0.08); }
  .plane-stand.departed { border-color: rgba(100,116,139,0.38); opacity: 0.5; }
  .plane { position: absolute; transform: translate(-50%, -50%); z-index: 15;
    display: flex; flex-direction: column; align-items: center; gap: 3px; transition: opacity 0.7s ease; }
  .plane-icon { font-size: 44px; line-height: 1; color: var(--neon);
    transition: transform 0.9s ease; filter: drop-shadow(0 0 7px rgba(59,215,255,0.5)); }
  .plane-label { font-size: 11px; font-weight: 900; color: #eefaff; text-shadow: 0 1px 3px #000; white-space: nowrap; }
  .plane-tag { font-size: 9px; font-weight: 900; padding: 1px 7px; border-radius: 999px;
    background: rgba(52,211,153,0.18); color: var(--green); }
  .plane.delayed .plane-icon { color: #f87171 !important;
    filter: drop-shadow(0 0 9px rgba(248,113,113,0.75)); animation: planePulse 0.8s infinite alternate; }
  .plane.delayed .plane-tag { background: rgba(248,113,113,0.2); color: var(--red); }
  .plane.boarding .plane-tag { background: rgba(59,130,246,0.2); color: #93c5fd; }
  .plane.departed { opacity: 0; }
  .plane.departed .plane-icon { transform: translate(210px, -280px) rotate(8deg); }
  @keyframes planePulse { from { filter: drop-shadow(0 0 6px rgba(248,113,113,0.5)); }
    to { filter: drop-shadow(0 0 15px rgba(248,113,113,0.95)); } }
  .legend { display: flex; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 12px; }
  .legend span { border: 1px solid rgba(59,215,255,0.28); background: rgba(7,22,36,0.88);
    border-radius: 999px; padding: 6px 10px; }
  .passenger { position: absolute; left: 0; top: 0; width: 18px; height: 18px; border-radius: 50%;
    background: var(--yellow); border: 2px solid #fff0b8; box-shadow: 0 0 15px rgba(255,209,102,0.64);
    transform: translate(-40px, -40px); transition: opacity 0.2s ease; z-index: 20; }
  .passenger::after { content: attr(data-label); position: absolute; top: 21px; left: 50%;
    transform: translateX(-50%); font-size: 9px; font-weight: 900; color: #eefaff;
    text-shadow: 0 1px 4px #000; white-space: nowrap; }
  .passenger.boarded { background: var(--green); border-color: #d9ffec; box-shadow: 0 0 15px rgba(52,211,153,0.68); }
  /* Passengers belonging to a delayed flight get a pulsing red ring so the delayed flight is recognisable in the crowd. */
  .passenger.delayed-flight { border-color: #fecaca; box-shadow: 0 0 16px rgba(248,113,113,0.85); }
  .passenger.delayed-flight::before { content: ""; position: absolute; inset: -4px; border-radius: 50%;
    border: 2px solid rgba(248,113,113,0.9); animation: ringPulse 0.9s infinite alternate; }
  @keyframes ringPulse { from { opacity: 0.45; } to { opacity: 1; } }
  .passenger-timer { position: absolute; top: 34px; left: 50%; transform: translateX(-50%); min-width: 86px;
    border: 1px solid rgba(59,215,255,0.28); border-radius: 6px; background: rgba(3,9,20,0.82);
    color: #d8f6ff; font-size: 9px; font-weight: 900; padding: 2px 4px; text-align: center; white-space: nowrap; }
  .queue-dot { position: absolute; width: 8px; height: 8px; border-radius: 50%;
    background: rgba(59,215,255,0.72); box-shadow: 0 0 11px rgba(59,215,255,0.7); z-index: 8; }
  .flight-group-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .flight-group-badge { border: 1px solid rgba(59,215,255,0.25); border-left: 5px solid var(--flight-color);
    border-radius: 8px; background: rgba(7,22,36,0.88); color: #eefaff; font-size: 12px; font-weight: 900; padding: 7px 10px; }
  .operations-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; margin-top: 16px; }
  .control-panel { border: 1px solid rgba(59,215,255,0.32); border-radius: 14px; background: rgba(7,22,36,0.92);
    box-shadow: 0 18px 44px rgba(0,0,0,0.24); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
  .panel-title { color: #f4fbff; font-size: 20px; font-weight: 900; }
  .panel-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .resource-groups { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
  .resource-group { border: 1px solid rgba(59,215,255,0.22); border-radius: 10px; background: rgba(3,9,20,0.56); padding: 10px; }
  .resource-group-title { color: var(--neon); font-size: 13px; font-weight: 900; text-transform: uppercase; margin-bottom: 8px; }
  .resource-stack { display: grid; gap: 8px; }
  .resource-card { border: 1px solid rgba(52,211,153,0.42); border-left: 5px solid var(--green); border-radius: 8px;
    background: rgba(11,32,49,0.88); padding: 9px; min-height: 145px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease; }
  .resource-card.busy { border-color: rgba(251,191,36,0.62); border-left-color: var(--busy); box-shadow: 0 0 18px rgba(251,191,36,0.12); }
  .resource-card.overloaded { border-color: rgba(248,113,113,0.72); border-left-color: var(--red);
    box-shadow: 0 0 22px rgba(248,113,113,0.2); animation: overloadPulse 1s infinite alternate; }
  .resource-card.closed { border-color: rgba(100,116,139,0.52); border-left-color: var(--closed); opacity: 0.72; }
  .resource-name { display: flex; justify-content: space-between; gap: 8px; color: #eefaff; font-size: 13px; font-weight: 900; margin-bottom: 7px; }
  .status-pill { border-radius: 999px; padding: 3px 7px; background: rgba(52,211,153,0.18); color: var(--green); font-size: 10px; font-weight: 900; }
  .status-pill.busy { background: rgba(251,191,36,0.18); color: var(--busy); }
  .status-pill.overloaded { background: rgba(248,113,113,0.18); color: var(--red); }
  .status-pill.closed { background: rgba(100,116,139,0.2); color: #cbd5e1; }
  .bottleneck-alert { border: 1px solid rgba(52,211,153,0.44); border-radius: 8px; background: rgba(52,211,153,0.1);
    color: var(--green); padding: 9px 11px; min-width: 260px; text-align: right; font-size: 12px; font-weight: 900; }
  .bottleneck-alert.busy { border-color: rgba(251,191,36,0.58); background: rgba(251,191,36,0.12); color: var(--busy); }
  .bottleneck-alert.overloaded { border-color: rgba(248,113,113,0.7); background: rgba(248,113,113,0.13); color: var(--red);
    animation: overloadPulse 0.75s infinite alternate; }
  .bottleneck-alert.closed { border-color: rgba(100,116,139,0.54); background: rgba(100,116,139,0.12); color: #cbd5e1; }
  .resource-row { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 11px; margin-top: 5px; }
  .resource-value { color: #f8fdff; font-weight: 900; text-align: right; }
  @keyframes overloadPulse { from { transform: translateY(0); } to { transform: translateY(-2px); } }
  .flight-panel { overflow: hidden; }
  .flight-clock { border: 1px solid rgba(59,215,255,0.38); border-radius: 8px; color: var(--neon);
    background: rgba(3,9,20,0.55); padding: 8px 10px; min-width: 110px; text-align: center; font-weight: 900; }
  .flight-board-table { width: 100%; border-collapse: collapse; color: #effaff; font-size: 13px; overflow: hidden; border-radius: 10px; }
  .flight-board-table th { background: rgba(59,215,255,0.14); color: var(--neon); padding: 10px; text-align: left;
    text-transform: uppercase; font-size: 11px; border-bottom: 1px solid rgba(59,215,255,0.28); }
  .flight-board-table td { background: rgba(3,9,20,0.62); border-bottom: 1px solid rgba(59,215,255,0.12); padding: 10px; font-weight: 800; }
  .flight-board-table tr:last-child td { border-bottom: none; }
  .flight-row.delayed td { background: rgba(248,113,113,0.14); animation: delayedRowPulse 0.95s infinite alternate; }
  .flight-status-badge { display: inline-flex; justify-content: center; min-width: 98px; border-radius: 999px;
    padding: 4px 9px; font-size: 11px; font-weight: 900; }
  .flight-status-badge.on-time { background: rgba(52,211,153,0.18); color: var(--green); }
  .flight-status-badge.delayed { background: rgba(248,113,113,0.2); color: var(--red); animation: delayBlink 0.72s infinite alternate; }
  .flight-status-badge.boarding { background: rgba(59,130,246,0.2); color: #93c5fd; }
  .flight-status-badge.gate-closed { background: rgba(251,191,36,0.18); color: var(--busy); }
  .flight-status-badge.departed { background: rgba(100,116,139,0.24); color: #cbd5e1; }
  @keyframes delayedRowPulse { from { box-shadow: inset 4px 0 0 rgba(248,113,113,0.28); } to { box-shadow: inset 4px 0 0 rgba(248,113,113,0.86); } }
  @keyframes delayBlink { from { filter: brightness(0.86); } to { filter: brightness(1.35); } }
  @media (max-width: 1100px) { .resource-groups { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
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
    <div class="sim-control-strip" aria-label="Simulation Speed Control">
      <div>
        <div class="sim-control-title">Simulation Speed Control</div>
        <div class="subtitle">Default: Slow | Auto or Step-by-Step demonstration mode</div>
      </div>
      <div class="sim-control-buttons">
        <button class="sim-button" data-speed="0.25">Very Slow</button>
        <button class="sim-button active" data-speed="0.65">Slow</button>
        <button class="sim-button" data-speed="1.12">Normal</button>
        <button class="sim-button" data-speed="2.1">Fast</button>
        <button class="sim-button active" data-mode="auto">Auto Simulation</button>
        <button class="sim-button" data-mode="step">Step-by-Step Simulation</button>
        <button class="sim-button" id="next-step-button" disabled>Next Step</button>
      </div>
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
      <section class="zone apron"><div class="zone-title">Aircraft Stands &amp; Boarding Lanes</div><div class="zone-note">Each flight has its own plane &amp; boarding lane | red = delayed | plane departs and clears the stand once boarded</div></section>
    </div>
    <div class="legend">
      <span>Blue route: active passenger flow</span>
      <span>Coloured dots: passengers (by flight)</span>
      <span>Green dots: boarded</span>
      <span>Red plane / red ring: delayed flight</span>
      <span>Empty stand: aircraft has departed</span>
    </div>
    <div class="flight-group-strip" id="flight-group-strip" aria-label="Flight Passenger Groups"></div>
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
              <div class="resource-card" data-resource="checkin-0"><div class="resource-name">Counter 1 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="checkin-1"><div class="resource-name">Counter 2 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="checkin-2"><div class="resource-name">Counter 3 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
            </div>
          </div>
          <div class="resource-group">
            <div class="resource-group-title">Security Lanes</div>
            <div class="resource-stack">
              <div class="resource-card" data-resource="security-0"><div class="resource-name">Lane 1 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="security-1"><div class="resource-name">Lane 2 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
            </div>
          </div>
          <div class="resource-group">
            <div class="resource-group-title">Immigration Counters</div>
            <div class="resource-stack">
              <div class="resource-card" data-resource="immigration-0"><div class="resource-name">Immigration 1 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="immigration-1"><div class="resource-name">Immigration 2 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="immigration-2"><div class="resource-name">Immigration 3 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
            </div>
          </div>
          <div class="resource-group">
            <div class="resource-group-title">Boarding Gates</div>
            <div class="resource-stack">
              <div class="resource-card" data-resource="boarding-0"><div class="resource-name">Gate A1 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="boarding-1"><div class="resource-name">Gate A2 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
            </div>
          </div>
        </div>
      </section>
      <section class="control-panel flight-panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Airport Flight Information Board</div>
            <div class="panel-subtitle">ML-predicted gate-level departure status with animated delay alerts</div>
          </div>
          <div class="flight-clock" id="flight-clock">09:00</div>
        </div>
        <table class="flight-board-table" aria-label="Airport Flight Information Board">
          <thead><tr>
            <th>Flight</th><th>Destination</th><th>Gate</th><th>Scheduled</th><th>Estimated</th>
            <th>Status</th><th>Delay Reason</th><th>Waiting</th><th>Bottleneck</th>
          </tr></thead>
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
  const flightGroupStrip = document.getElementById("flight-group-strip");
  const speedButtons = Array.from(document.querySelectorAll("[data-speed]"));
  const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
  const nextStepButton = document.getElementById("next-step-button");
  let speed = 0.65;
  let simulationMode = "auto";
  let stepIndex = 0;
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
  const flights = __FLIGHT_DATA__;
  // ~8 passengers per selected flight, so each plane's boarding lane stays readable.
  const passengerCount = Math.max(12, flights.length * 8);
  const points = {
    entrance: { x: 82, y: 370 }, checkinQueue: { x: 238, y: 370 }, checkinCounter: { x: 444, y: 370 },
    securityQueue: { x: 616, y: 370 }, securityLane: { x: 804, y: 370 }, immigrationQueue: { x: 976, y: 370 },
    immigrationCounter: { x: 1168, y: 370 }, lounge: { x: 398, y: 648 }, boardingQueue: { x: 682, y: 648 },
    boardingGate: { x: 934, y: 648 }, aircraft: { x: 1180, y: 648 }
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
    { name: "boardingApproach", from: "lounge", to: "boardingLaneEntry", duration: 3.4, perFlight: true },
    { name: "boardingLane", from: "boardingLaneEntry", to: "boardingLaneEntry", duration: 3.2, queue: "boarding", perFlight: true },
    { name: "boardingGate", from: "boardingLaneEntry", to: "planeStand", duration: 3.0, progress: [8, 9], perFlight: true },
    { name: "boarded", from: "planeStand", to: "planeStand", duration: 999, perFlight: true }
  ];
  const stepMoments = stages.slice(0, -1).reduce((moments, stage, index) => {
    const previous = index === 0 ? 0 : moments[index - 1];
    moments.push(previous + Math.min(stage.duration, 5));
    return moments;
  }, [0]);
  function ease(value) { return value < 0.5 ? 2 * value * value : 1 - Math.pow(-2 * value + 2, 2) / 2; }
  function interpolate(start, end, progress) {
    const curved = ease(Math.max(0, Math.min(progress, 1)));
    return { x: start.x + (end.x - start.x) * curved, y: start.y + (end.y - start.y) * curved };
  }
  function queuePosition(queue, index) {
    const column = index % 5; const row = Math.floor(index / 5) % 7;
    const positions = {
      checkin: { x: 208, y: 296, gapX: 23, gapY: 20 }, security: { x: 582, y: 296, gapX: 21, gapY: 20 },
      immigration: { x: 938, y: 296, gapX: 23, gapY: 20 }, boarding: { x: 622, y: 626, gapX: 23, gapY: 17 }
    };
    const base = positions[queue];
    return { x: base.x + column * base.gapX, y: base.y + row * base.gapY };
  }
  function pointFor(name, passenger) {
    // Boarding waypoints are per-flight: every plane has its own lane entry and stand.
    if (name === "boardingLaneEntry") return { x: planeX(passenger.flightIndex) - 86, y: APRON_Y };
    if (name === "planeStand") return { x: planeX(passenger.flightIndex), y: APRON_Y };
    return points[name];
  }
  function boardingLanePosition(passenger) {
    // A separate horizontal boarding queue per flight, feeding right into its plane.
    const entryX = planeX(passenger.flightIndex) - 86;
    const column = passenger.flightSlot % 6;
    const row = Math.floor(passenger.flightSlot / 6) % 2;
    return { x: entryX + column * 12, y: APRON_Y + (row ? 17 : -3) };
  }
  function phaseFor(localTime, passenger) {
    let cursor = 0;
    for (const stage of stages) {
      const next = cursor + stage.duration;
      if (localTime <= next) {
        const progress = (localTime - cursor) / stage.duration;
        if (stage.queue === "boarding" && progress < 0.68) return { stage, progress, position: boardingLanePosition(passenger) };
        if (stage.queue && progress < 0.74) return { stage, progress, position: queuePosition(stage.queue, passenger.index) };
        return { stage, progress, position: interpolate(pointFor(stage.from, passenger), pointFor(stage.to, passenger), progress) };
      }
      cursor = next;
    }
    return { stage: stages[stages.length - 1], progress: 1, position: pointFor("planeStand", passenger) };
  }
  function timerText(state) {
    if (state.stage.name === "boarded") return "Boarded ✓";
    if (state.stage.queue || state.stage.name === "lounge" || state.stage.name === "boardingApproach")
      return `Waiting: ${(state.progress * state.stage.duration).toFixed(1)} min`;
    if (state.stage.progress) {
      const remaining = Math.max(0, state.stage.duration * (1 - state.progress));
      return `Service Remaining: ${remaining.toFixed(1)} min`;
    }
    return "";
  }
  function createPassenger(index) {
    const element = document.createElement("div");
    const flightIndex = index % flights.length;
    const flightSlot = Math.floor(index / flights.length);
    const flight = flights[flightIndex];
    const passengerId = `P${String(index + 1).padStart(3, "0")}`;
    element.className = "passenger";
    element.dataset.id = passengerId; element.dataset.flight = flight.flight;
    element.dataset.label = `${passengerId} / ${flight.flight}`;
    element.style.background = flight.color || "var(--yellow)";
    element.style.borderColor = flight.color || "#fff0b8";
    if (flight.delayMinutes > 0 || flight.delayProbability >= 0.55) element.classList.add("delayed-flight");
    const timer = document.createElement("span");
    timer.className = "passenger-timer"; element.appendChild(timer); map.appendChild(element);
    return { index, element, timer, delay: index * 0.45, flight, flightIndex, flightSlot };
  }
  function renderFlightGroups() {
    flightGroupStrip.innerHTML = flights.map((flight) => `
      <span class="flight-group-badge" style="--flight-color:${flight.color || "#3bd7ff"}">
        ${flight.flight} / ${flight.destination}
      </span>`).join("");
  }
  function createQueueDots() {
    [["checkin", 15], ["security", 12], ["immigration", 13]].forEach(([queue, count]) => {
      for (let index = 0; index < count; index += 1) {
        const dot = document.createElement("div"); const position = queuePosition(queue, index);
        dot.className = "queue-dot"; dot.style.left = `${position.x + 7}px`; dot.style.top = `${position.y + 7}px`;
        map.appendChild(dot);
      }
    });
  }
  // ----- Aircraft stands: one plane per flight (one gate = one plane) -----
  const planeEls = [];
  const standEls = [];
  const planeSpacing = 1180 / Math.max(flights.length, 1);
  function planeX(index) { return 70 + planeSpacing * (index + 0.5); }
  const APRON_Y = 812;
  function setupPlanes() {
    flights.forEach((flight, index) => {
      const x = planeX(index);
      const stand = document.createElement("div");
      stand.className = "plane-stand";
      stand.style.left = `${x}px`;
      stand.style.top = `${APRON_Y}px`;
      stand.innerHTML = `<div class="stand-gate">Stand ${flight.gate}</div>`;
      map.appendChild(stand);
      standEls.push(stand);

      const plane = document.createElement("div");
      plane.className = "plane";
      plane.style.left = `${x}px`;
      plane.style.top = `${APRON_Y}px`;
      plane.innerHTML = `<div class="plane-icon">✈</div>` +
        `<div class="plane-label">${flight.flight}</div>` +
        `<div class="plane-tag">ON TIME</div>`;
      plane.querySelector(".plane-icon").style.color = flight.color || "#3bd7ff";
      map.appendChild(plane);
      planeEls.push(plane);
    });
  }
  function updatePlanes(elapsed) {
    flights.forEach((flight, index) => {
      const plane = planeEls[index];
      const stand = standEls[index];
      if (!plane || !stand) return;
      const state = flightStatusFor(flight, elapsed);
      const isDelayed = flight.delayMinutes > 0 || flight.delayProbability >= 0.55;
      const departed = state.status === "DEPARTED";
      const boarding = state.status === "BOARDING" || state.status === "GATE CLOSED";
      plane.classList.toggle("departed", departed);
      plane.classList.toggle("delayed", isDelayed && !departed);
      plane.classList.toggle("boarding", boarding && !departed);
      stand.classList.toggle("delayed", isDelayed && !departed);
      stand.classList.toggle("departed", departed);
      plane.querySelector(".plane-tag").textContent = state.status;
    });
  }
  function resetProgressBars() { progressBars.forEach((bar) => { bar.style.width = "0%"; }); }
  function updateProgress(stage, progress, passengerIndex) {
    if (!stage.progress) return;
    const barIndex = stage.progress[passengerIndex % stage.progress.length];
    if (progressBars[barIndex]) progressBars[barIndex].style.width = `${Math.round(progress * 100)}%`;
  }
  function field(card, name) { return card.querySelector(`[data-field="${name}"]`); }
  function setResourceStatus(card, status) {
    const className = status.toLowerCase(); const statusField = field(card, "status");
    card.classList.remove("busy", "overloaded", "closed");
    statusField.classList.remove("busy", "overloaded", "closed");
    if (className !== "open") { card.classList.add(className); statusField.classList.add(className); }
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
      item.state.stage.queue === resource.queue && item.passenger.index % resource.count === resource.index));
    const activePassenger = passengerStates.find((item) => (
      item.state.stage.name === resource.serviceStage && item.passenger.index % resource.count === resource.index));
    const queueLength = closed ? 0 : queuedPassengers.length;
    const remaining = activePassenger ? Math.max(0, resource.serviceMinutes * (1 - activePassenger.state.progress)) : 0;
    const averageWait = closed ? 0 : resource.baseWait + queueLength * resource.waitFactor + (activePassenger ? 0.5 : 0);
    let status = "OPEN";
    if (closed) status = "CLOSED";
    else if (queueLength >= resource.overloadAt) status = "OVERLOADED";
    else if (activePassenger || queueLength >= resource.busyAt) status = "BUSY";
    return { card, resource, passenger: activePassenger ? activePassenger.passenger.element.dataset.label : "None",
      queueLength, averageWait, remaining, status };
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
    let message = "NORMAL FLOW"; let level = "open";
    if (overloaded.length > 0) {
      const worst = overloaded.sort((left, right) => right.queueLength - left.queueLength)[0];
      message = `BOTTLENECK: ${worst.resource.label} | Queue ${worst.queueLength}`; level = "overloaded";
    } else if (busy.length > 0) {
      const busiest = busy.sort((left, right) => right.queueLength - left.queueLength)[0];
      message = `BUSY: ${busiest.resource.label} | Queue ${busiest.queueLength}`; level = "busy";
    } else if (closed.length > 0) { message = `${closed.length} RESOURCE CLOSED`; level = "closed"; }
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
    const hasDelay = flight.delayMinutes > 0 || flight.delayProbability >= 0.55;
    let status = "ON TIME"; let estimatedMinutes = flight.scheduledMinutes;
    if (hasDelay && cycle >= flight.delayStart && cycle < flight.boardingStart) { status = "DELAYED"; estimatedMinutes += flight.delayMinutes; }
    else if (cycle >= flight.departedStart) { status = "DEPARTED"; estimatedMinutes += flight.delayMinutes; }
    else if (cycle >= flight.gateClosedStart) { status = "GATE CLOSED"; estimatedMinutes += flight.delayMinutes; }
    else if (cycle >= flight.boardingStart) { status = "BOARDING"; estimatedMinutes += flight.delayMinutes; }
    return { status, estimatedMinutes };
  }
  function flightStatusClass(status) { return status.toLowerCase().replaceAll(" ", "-"); }
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
          <td>${flight.flight}</td><td>${flight.destination}</td><td>${flight.gate}</td>
          <td>${formatTime(flight.scheduledMinutes)}</td><td>${formatTime(state.estimatedMinutes)}</td>
          <td><span class="flight-status-badge ${className}">${state.status}</span></td>
          <td>${flight.delayReason || "None"}</td><td>${flight.passengersWaiting || 0}</td>
          <td>${flight.bottleneckStage || "None"}</td>
        </tr>`;
    }).join("");
  }
  speedButtons.forEach((button) => {
    button.addEventListener("click", () => {
      speed = Number(button.dataset.speed);
      speedButtons.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
    });
  });
  modeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      simulationMode = button.dataset.mode;
      modeButtons.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      nextStepButton.disabled = simulationMode !== "step";
    });
  });
  nextStepButton.addEventListener("click", () => {
    if (simulationMode !== "step") return;
    stepIndex = (stepIndex + 1) % stepMoments.length;
    simulationElapsed = stepMoments[stepIndex];
  });
  createQueueDots();
  renderFlightGroups();
  setupPlanes();
  const passengers = Array.from({ length: passengerCount }, (_, index) => createPassenger(index));
  let simulationElapsed = 0;
  let lastFrameTime = performance.now();
  function animate(now) {
    const frameDelta = Math.min((now - lastFrameTime) / 1000, 0.08);
    lastFrameTime = now;
    if (simulationMode === "auto") simulationElapsed += frameDelta * speed;
    const elapsed = simulationElapsed;
    const passengerStates = [];
    resetProgressBars();
    passengers.forEach((passenger) => {
      const localTime = elapsed - passenger.delay;
      if (localTime < 0) { passenger.element.style.opacity = "0"; return; }
      const state = phaseFor(localTime, passenger);
      const boarded = state.stage.name === "boarded";
      // Once boarded, the passenger fades into their aircraft and leaves the floor.
      passenger.element.style.opacity = boarded ? "0" : "1";
      passenger.element.style.transform = `translate(${state.position.x}px, ${state.position.y}px)`;
      passenger.element.classList.toggle("boarded", state.stage.name === "boardingGate" || boarded);
      passenger.timer.textContent = timerText(state);
      passengerStates.push({ passenger, state });
      updateProgress(state.stage, state.progress, passenger.index);
    });
    updateResourceCards(passengerStates, elapsed);
    updateFlightBoard(elapsed);
    updatePlanes(elapsed);
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
})();
</script>
</body>
</html>
"""
