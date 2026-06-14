"""Animated airport control-center component (HTML/CSS/JS).

This module builds a self-contained, embeddable control room: passenger dots
flow through eleven terminal zones (entrance → check-in → security → immigration
→ lounge → boarding → aircraft) via a smooth ``requestAnimationFrame`` loop, a
live Resource Operations Monitor tracks each counter/lane/gate, and a flight board
animates departure statuses.

The flight rows are produced by :func:`build_flight_rows`, which queries the
trained ``best_model.joblib`` for risk inputs. The browser simulation then uses
live terminal congestion, passenger readiness, and gate availability to decide
actual clearance, boarding, and delay state. The visual layer is rendered through
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
    {"flight": "SA-781", "gate": "B1", "minute": 40, "phaseOffset": 16, "color": "#a78bfa"},
    {"flight": "ER-219", "gate": "B2", "minute": 5, "phaseOffset": 24, "color": "#34d399"},
    {"flight": "GB-508", "gate": "C1", "minute": 20, "phaseOffset": 32, "color": "#facc15"},
    {"flight": "PK-419", "gate": "C2", "minute": 35, "phaseOffset": 40, "color": "#fb7185"},
]

DELAY_REASON_BOTTLENECK = {
    "High Passenger Arrival Rate": "Check-in",
    "Check-in Counter Congestion": "Check-in",
    "Security Queue": "Security",
    "Immigration Queue": "Immigration",
    "Gate Change": "Boarding",
    "Previous Flight Delay": "Aircraft",
    "Inbound Aircraft Not Ready": "Aircraft",
    "Weather Disruption": "Air Traffic",
    "Technical Inspection": "Aircraft",
    "Crew Availability": "Operations",
}


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
                      selected: list[str] | None = None,
                      scenario: dict | None = None) -> list[dict]:
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
        scenario: Optional live-control overrides for forced delays and crowd level.

    Returns:
        A list of JSON-serialisable flight-row dictionaries consumed by the JS map.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    scenario = scenario or {}
    force_delay = bool(scenario.get("force_delay", False))
    forced_flight = str(scenario.get("flight", ""))
    forced_reason = str(scenario.get("reason", "High Passenger Arrival Rate"))
    forced_minutes = int(scenario.get("delay_minutes", 24))
    rush_multiplier = float(scenario.get("rush_multiplier", 1.0))

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
        planned_clearance_start = 32 + i * 6
        planned_boarding_start = planned_clearance_start + 5
        planned_gate_closed_start = planned_boarding_start + 15
        planned_departed_start = planned_gate_closed_start + 10
        inbound_hold_minutes = int(min(22, max(0, round(estimated_delay * 0.35))))
        minimum_ready_ratio = round(0.68 + min(0.16, probability * 0.18), 2)
        pressure_hold_limit = max(7, 14 - i)
        forced_delay = force_delay and template["flight"] == forced_flight
        if forced_delay:
            forced_minutes = max(5, min(120, forced_minutes))
            estimated_delay = max(estimated_delay, forced_minutes)
            probability = max(float(probability), 0.82)
            reason = forced_reason
            bottleneck = DELAY_REASON_BOTTLENECK.get(forced_reason, "Operations")
            passengers_waiting = max(passengers_waiting, int(round(18 + forced_minutes * 0.55)))
            inbound_hold_minutes = max(inbound_hold_minutes, int(round(forced_minutes * 0.55)))
            minimum_ready_ratio = max(minimum_ready_ratio, 0.86)
            pressure_hold_limit = max(4, pressure_hold_limit - int(round(forced_minutes / 12)))

        rows.append({
            "flight": template["flight"],
            "nextFlight": f"NX-{720 + i * 37}",
            "destination": AIRPORT_DESTINATIONS.get(destination, destination),
            "gate": template["gate"],
            "scheduledMinutes": scheduled_minutes,
            "delayMinutes": estimated_delay,
            "delayProbability": round(float(probability), 3),
            "riskLevel": _risk_level(probability),
            "delayReason": reason,
            "passengersWaiting": passengers_waiting,
            "bottleneckStage": bottleneck,
            "forcedDelay": forced_delay,
            "forcedDelayMinutes": forced_minutes if forced_delay else 0,
            "scenarioRushMultiplier": round(rush_multiplier, 2),
            "plannedClearanceStart": planned_clearance_start,
            "plannedBoardingStart": planned_boarding_start,
            "plannedGateClosedStart": planned_gate_closed_start,
            "plannedDepartedStart": planned_departed_start,
            "clearanceStart": planned_clearance_start,
            "boardingStart": planned_boarding_start,
            "gateClosedStart": planned_gate_closed_start,
            "departedStart": planned_departed_start,
            "turnaroundStart": planned_departed_start + 12,
            "inboundHoldMinutes": inbound_hold_minutes,
            "minimumReadyRatio": minimum_ready_ratio,
            "pressureHoldLimit": pressure_hold_limit,
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
  .control-shell { min-height: 1080px; padding: 18px;
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
  .fullscreen-button { border-color: rgba(52,211,153,0.56); color: #d9ffec; }
  .fullscreen-button.active { border-color: rgba(52,211,153,0.9); background: rgba(52,211,153,0.2); color: var(--green); }
  .control-shell:fullscreen { width: 100vw; height: 100vh; min-height: 100vh; overflow: auto; padding: 16px; }
  .control-shell.fullscreen-fallback { position: fixed; inset: 0; z-index: 99999; width: 100vw; height: 100vh;
    min-height: 100vh; overflow: auto; padding: 16px; }
  .control-shell:fullscreen .airport-map,
  .control-shell.fullscreen-fallback .airport-map { height: 1030px; }
  .airport-map { position: relative; height: 1030px; border: 1px solid rgba(59,215,255,0.34);
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
  .lounge { left: 248px; top: 588px; width: 292px; height: 122px; }
  .boarding-queue { left: 586px; top: 588px; width: 174px; height: 122px; }
  .boarding-gates { left: 24px; top: 724px; width: 1238px; height: 96px; box-sizing: border-box;
    display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 6px 8px; align-content: start; }
  .boarding-gates .zone-title, .boarding-gates .zone-note { grid-column: 1 / -1; }
  .boarding-gates .desk { margin-top: 0; padding: 6px; min-height: 28px; text-align: center; }
  .apron { left: 24px; top: 836px; width: 1238px; height: 176px; box-sizing: border-box; }
  .queue-zone { overflow: hidden; }
  .queue-lanes { position: absolute; left: 12px; right: 12px; top: 68px; bottom: 18px;
    display: grid; gap: 10px; pointer-events: none; }
  .queue-lanes.lanes-2 { grid-template-rows: repeat(2, minmax(0, 1fr)); }
  .queue-lanes.lanes-3 { grid-template-rows: repeat(3, minmax(0, 1fr)); }
  .queue-lane { position: relative; border: 1px solid rgba(59,215,255,0.18); border-radius: 8px;
    background: linear-gradient(90deg, rgba(59,215,255,0.1), rgba(59,215,255,0.03)); }
  .queue-lane::before { content: attr(data-label); position: absolute; left: 7px; top: 50%;
    transform: translateY(-50%); color: rgba(216,246,255,0.72); font-size: 8px; font-weight: 900; }
  .queue-lane::after { content: ""; position: absolute; left: 31px; right: 8px; top: 50%; height: 2px;
    transform: translateY(-50%); border-radius: 999px;
    background: linear-gradient(90deg, rgba(59,215,255,0.82), rgba(52,211,153,0.18)); }
  .service-bank { overflow: visible; }
  .service-bank .desk { position: absolute; left: 12px; right: 12px; height: 30px; margin-top: 0;
    display: flex; align-items: center; justify-content: space-between; gap: 6px; }
  .checkin-counters .desk:nth-of-type(2), .immigration-counters .desk:nth-of-type(2) { top: 84px; }
  .checkin-counters .desk:nth-of-type(3), .immigration-counters .desk:nth-of-type(3) { top: 129px; }
  .checkin-counters .desk:nth-of-type(4), .immigration-counters .desk:nth-of-type(4) { top: 174px; }
  .security-lanes .desk:nth-of-type(2) { top: 94px; }
  .security-lanes .desk:nth-of-type(3) { top: 156px; }
  .service-bank .desk::before { content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: var(--neon); box-shadow: 0 0 9px rgba(59,215,255,0.7); flex: 0 0 auto; }
  .service-bank .desk .progress { left: 22px; width: 0%; max-width: calc(100% - 22px); }
  .desk { position: relative; border: 1px solid rgba(59,215,255,0.42); background: rgba(59,215,255,0.11);
    border-radius: 7px; padding: 7px; margin-top: 8px; color: #e8f8ff; font-size: 11px; font-weight: 800; overflow: hidden; }
  .desk .progress { position: absolute; left: 0; bottom: 0; height: 3px; width: 0%;
    background: linear-gradient(90deg, var(--green), var(--neon)); transition: width 0.28s ease; }
  .route { position: absolute; height: 3px; background: linear-gradient(90deg, transparent, var(--neon), transparent);
    opacity: 0.78; transform-origin: left center; }
  .route.main { left: 142px; top: 370px; width: 1110px; }
  .route.down { left: 460px; top: 514px; width: 320px; transform: rotate(18deg); }
  /* Aircraft stands: one parking stand per flight, each holding exactly one plane. */
  .plane-stand { position: absolute; width: 154px; height: 78px; transform: translate(-50%, -50%);
    border: 1px dashed rgba(59,215,255,0.35); border-radius: 12px; background: rgba(3,9,20,0.42); z-index: 6; }
  .plane-stand .stand-gate { position: absolute; top: 6px; left: 9px; color: var(--muted);
    font-size: 10px; font-weight: 900; letter-spacing: 0.05em; text-transform: uppercase; }
  .plane-stand .stand-flight { position: absolute; left: 9px; bottom: 7px; color: #eefaff;
    font-size: 10px; font-weight: 900; max-width: 132px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .plane-stand.delayed { border-color: rgba(248,113,113,0.6); background: rgba(248,113,113,0.08); }
  .plane-stand.departed { border-color: rgba(100,116,139,0.38); opacity: 0.5; }
  .plane-stand.turnaround { border-color: rgba(52,211,153,0.45); background: rgba(52,211,153,0.08); opacity: 1; }
  .plane { position: absolute; transform: translate(-50%, -50%); z-index: 15;
    display: flex; flex-direction: column; align-items: center; gap: 3px; transition: opacity 0.7s ease; }
  .plane-icon { font-size: 34px; line-height: 1; color: var(--neon);
    transition: transform 0.9s ease; filter: drop-shadow(0 0 7px rgba(59,215,255,0.5)); }
  .plane-label { font-size: 11px; font-weight: 900; color: #eefaff; text-shadow: 0 1px 3px #000; white-space: nowrap; }
  .plane-tag { font-size: 9px; font-weight: 900; padding: 1px 7px; border-radius: 999px;
    background: rgba(52,211,153,0.18); color: var(--green); }
  .plane.delayed .plane-icon { color: #f87171 !important;
    filter: drop-shadow(0 0 9px rgba(248,113,113,0.75)); animation: planePulse 0.8s infinite alternate; }
  .plane.delayed .plane-tag { background: rgba(248,113,113,0.2); color: var(--red); }
  .plane.boarding .plane-tag { background: rgba(59,130,246,0.2); color: #93c5fd; }
  .plane.turnaround .plane-icon { color: #34d399 !important; filter: drop-shadow(0 0 8px rgba(52,211,153,0.55)); }
  .plane.turnaround .plane-tag { background: rgba(52,211,153,0.18); color: var(--green); }
  .plane.departed { opacity: 0; }
  .plane.departed .plane-icon { transform: translate(210px, -280px) rotate(8deg); }
  .gate-connector { position: absolute; width: 3px; border-radius: 999px;
    background: linear-gradient(180deg, rgba(59,215,255,0.15), rgba(59,215,255,0.9), rgba(52,211,153,0.76));
    box-shadow: 0 0 12px rgba(59,215,255,0.35); opacity: 0.82; z-index: 5; }
  .service-link { position: absolute; height: 2px; transform-origin: left center; border-radius: 999px;
    background: linear-gradient(90deg, rgba(59,215,255,0.08), rgba(59,215,255,0.95), rgba(52,211,153,0.6));
    box-shadow: 0 0 10px rgba(59,215,255,0.34); opacity: 0.7; z-index: 10; }
  @keyframes planePulse { from { filter: drop-shadow(0 0 6px rgba(248,113,113,0.5)); }
    to { filter: drop-shadow(0 0 15px rgba(248,113,113,0.95)); } }
  .legend { display: flex; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 12px; }
  .legend span { border: 1px solid rgba(59,215,255,0.28); background: rgba(7,22,36,0.88);
    border-radius: 999px; padding: 6px 10px; }
  .passenger { position: absolute; left: 0; top: 0; width: 18px; height: 18px; border-radius: 50%;
    background: var(--yellow); border: 2px solid #fff0b8; box-shadow: 0 0 15px rgba(255,209,102,0.64);
    transform: translate(-40px, -40px); transition: opacity 0.2s ease; z-index: 20; }
  .passenger::after { content: attr(data-label); position: absolute; top: 21px; left: 50%;
    transform: translate(-50%, -4px); font-size: 9px; font-weight: 900; color: #eefaff;
    text-shadow: 0 1px 4px #000; white-space: nowrap; opacity: 0; visibility: hidden; pointer-events: none;
    transition: opacity 0.16s ease, transform 0.16s ease, visibility 0.16s ease; }
  .passenger:hover::after { opacity: 1; visibility: visible; transform: translate(-50%, 0); }
  .passenger.boarded { background: var(--green); border-color: #d9ffec; box-shadow: 0 0 15px rgba(52,211,153,0.68); }
  /* Passengers belonging to a delayed flight get a pulsing red ring so the delayed flight is recognisable in the crowd. */
  .passenger.delayed-flight { border-color: #fecaca; box-shadow: 0 0 16px rgba(248,113,113,0.85); }
  .passenger.delayed-flight::before { content: ""; position: absolute; inset: -4px; border-radius: 50%;
    border: 2px solid rgba(248,113,113,0.9); animation: ringPulse 0.9s infinite alternate; }
  @keyframes ringPulse { from { opacity: 0.45; } to { opacity: 1; } }
  .passenger-timer { position: absolute; top: 34px; left: 50%; transform: translate(-50%, -8px); min-width: 86px;
    border: 1px solid rgba(59,215,255,0.28); border-radius: 6px; background: rgba(3,9,20,0.82);
    color: #d8f6ff; font-size: 9px; font-weight: 900; padding: 2px 4px; text-align: center; white-space: nowrap;
    opacity: 0; visibility: hidden; pointer-events: none; z-index: 40;
    transition: opacity 0.16s ease, transform 0.16s ease, visibility 0.16s ease; }
  .passenger:hover .passenger-timer { opacity: 1; visibility: visible; transform: translate(-50%, 0); }
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
  .resource-group:last-child .resource-stack { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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
  .flight-status-badge.cleared { background: rgba(56,189,248,0.18); color: var(--neon); }
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
        <button class="sim-button fullscreen-button" id="fullscreen-button" type="button">Full Screen</button>
      </div>
    </div>
    <div class="airport-map" id="airport-map">
      <div class="route main"></div>
      <div class="route down"></div>
      <section class="zone entrance"><div class="zone-title">Entrance</div><div class="zone-note">Arrivals enter terminal</div></section>
      <section class="zone checkin-queue queue-zone"><div class="zone-title">Check-in Queue</div><div class="zone-note">Split by counter</div><div class="queue-lanes lanes-3"><span class="queue-lane" data-label="C1"></span><span class="queue-lane" data-label="C2"></span><span class="queue-lane" data-label="C3"></span></div></section>
      <section class="zone checkin-counters service-bank"><div class="zone-title">Check-in Counters</div><div class="desk service-desk">Counter 1<div class="progress"></div></div><div class="desk service-desk">Counter 2<div class="progress"></div></div><div class="desk service-desk">Counter 3<div class="progress"></div></div></section>
      <section class="zone security-queue queue-zone"><div class="zone-title">Security Queue</div><div class="zone-note">Split by lane</div><div class="queue-lanes lanes-2"><span class="queue-lane" data-label="L1"></span><span class="queue-lane" data-label="L2"></span></div></section>
      <section class="zone security-lanes service-bank"><div class="zone-title">Security Lanes</div><div class="desk service-desk">Lane 1<div class="progress"></div></div><div class="desk service-desk">Lane 2<div class="progress"></div></div></section>
      <section class="zone immigration-queue queue-zone"><div class="zone-title">Immigration Queue</div><div class="zone-note">Split by counter</div><div class="queue-lanes lanes-3"><span class="queue-lane" data-label="I1"></span><span class="queue-lane" data-label="I2"></span><span class="queue-lane" data-label="I3"></span></div></section>
      <section class="zone immigration-counters service-bank"><div class="zone-title">Immigration Counters</div><div class="desk service-desk">Immigration 1<div class="progress"></div></div><div class="desk service-desk">Immigration 2<div class="progress"></div></div><div class="desk service-desk">Immigration 3<div class="progress"></div></div></section>
      <section class="zone lounge"><div class="zone-title">Waiting Lounge</div><div class="zone-note">Passengers hold here until flight clearance</div></section>
      <section class="zone boarding-queue"><div class="zone-title">Boarding Queue</div><div class="zone-note">Released by flight clearance</div></section>
      <section class="zone boarding-gates"><div class="zone-title">Boarding Gates</div><div class="zone-note">One active flight per gate</div><div class="desk service-desk">Gate A1<div class="progress"></div></div><div class="desk service-desk">Gate A2<div class="progress"></div></div><div class="desk service-desk">Gate B1<div class="progress"></div></div><div class="desk service-desk">Gate B2<div class="progress"></div></div><div class="desk service-desk">Gate C1<div class="progress"></div></div><div class="desk service-desk">Gate C2<div class="progress"></div></div></section>
      <section class="zone apron"><div class="zone-title">Aircraft Stands &amp; Boarding Lanes</div><div class="zone-note">Six gates: A1, A2, B1, B2, C1, C2 | departed planes do not return; next flight occupies the stand</div></section>
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
              <div class="resource-card" data-resource="boarding-2"><div class="resource-name">Gate B1 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="boarding-3"><div class="resource-name">Gate B2 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="boarding-4"><div class="resource-name">Gate C1 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
              <div class="resource-card" data-resource="boarding-5"><div class="resource-name">Gate C2 <span class="status-pill" data-field="status">OPEN</span></div><div class="resource-row"><span>Current passenger</span><span class="resource-value" data-field="passenger">None</span></div><div class="resource-row"><span>Queue length</span><span class="resource-value" data-field="queue">0</span></div><div class="resource-row"><span>Average wait</span><span class="resource-value" data-field="wait">0.0 min</span></div><div class="resource-row"><span>Remaining service</span><span class="resource-value" data-field="remaining">0.0 min</span></div></div>
            </div>
          </div>
        </div>
      </section>
      <section class="control-panel flight-panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Airport Flight Information Board</div>
            <div class="panel-subtitle">Model-assisted status driven by live passenger readiness, terminal queues, and gate availability</div>
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
  const fullscreenButton = document.getElementById("fullscreen-button");
  const fullscreenTarget = document.querySelector(".control-shell");
  let speed = 0.65;
  let simulationMode = "auto";
  let stepIndex = 0;
  let lastFlightTick = -1;
  const GATES = ["A1", "A2", "B1", "B2", "C1", "C2"];
  const GATE_INDEX = Object.fromEntries(GATES.map((gate, index) => [gate, index]));
  const SERVICE_COUNTS = { checkin: 3, security: 2, immigration: 3 };
  const SERVICE_STAGE_GROUP = {
    checkinCounter: "checkin",
    securityLane: "security",
    immigrationCounter: "immigration",
  };
  const resources = [
    { id: "checkin-0", group: "checkin", label: "Check-in Counter 1", index: 0, count: 3, serviceStage: "checkinCounter", queue: "checkin", baseWait: 2.4, waitFactor: 0.85, serviceMinutes: 3.4, busyAt: 2, overloadAt: 5 },
    { id: "checkin-1", group: "checkin", label: "Check-in Counter 2", index: 1, count: 3, serviceStage: "checkinCounter", queue: "checkin", baseWait: 2.4, waitFactor: 0.85, serviceMinutes: 3.4, busyAt: 2, overloadAt: 5 },
    { id: "checkin-2", group: "checkin", label: "Check-in Counter 3", index: 2, count: 3, serviceStage: "checkinCounter", queue: "checkin", baseWait: 2.4, waitFactor: 0.85, serviceMinutes: 3.4, busyAt: 2, overloadAt: 5 },
    { id: "security-0", group: "security", label: "Security Lane 1", index: 0, count: 2, serviceStage: "securityLane", queue: "security", baseWait: 3.1, waitFactor: 1.15, serviceMinutes: 3.2, busyAt: 2, overloadAt: 4 },
    { id: "security-1", group: "security", label: "Security Lane 2", index: 1, count: 2, serviceStage: "securityLane", queue: "security", baseWait: 3.1, waitFactor: 1.15, serviceMinutes: 3.2, busyAt: 2, overloadAt: 4 },
    { id: "immigration-0", group: "immigration", label: "Immigration Counter 1", index: 0, count: 3, serviceStage: "immigrationCounter", queue: "immigration", baseWait: 3.8, waitFactor: 1.05, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5 },
    { id: "immigration-1", group: "immigration", label: "Immigration Counter 2", index: 1, count: 3, serviceStage: "immigrationCounter", queue: "immigration", baseWait: 3.8, waitFactor: 1.05, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5 },
    { id: "immigration-2", group: "immigration", label: "Immigration Counter 3", index: 2, count: 3, serviceStage: "immigrationCounter", queue: "immigration", baseWait: 3.8, waitFactor: 1.05, serviceMinutes: 3.8, busyAt: 2, overloadAt: 5, closesAfter: 62 },
    ...GATES.map((gate, index) => ({
      id: `boarding-${index}`, group: "boarding", label: `Boarding Gate ${gate}`,
      index, gateIndex: index, gate, count: GATES.length, serviceStage: "boardingGate",
      queue: "boarding", baseWait: 1.6, waitFactor: 0.72, serviceMinutes: 3.2,
      busyAt: 2, overloadAt: 5
    }))
  ];
  const flights = __FLIGHT_DATA__;
  const scenarioRushMultiplier = Math.max(0.7, Math.min(1.8,
    flights.reduce((max, flight) => Math.max(max, flight.scenarioRushMultiplier || 1), 1)
  ));
  // Base is ~8 passengers per flight; rush scenarios intentionally crowd the terminal.
  const passengerBatchSize = Math.max(8, Math.round(8 * scenarioRushMultiplier));
  const passengerCount = Math.max(12, flights.length * passengerBatchSize);
  const passengerArrivalSpacing = 0.45 / scenarioRushMultiplier;
  const boardingReleaseSpacing = 0.55 / Math.max(0.85, scenarioRushMultiplier);
  const points = {
    entrance: { x: 82, y: 370 }, checkinQueue: { x: 238, y: 370 }, checkinCounter: { x: 444, y: 370 },
    securityQueue: { x: 616, y: 370 }, securityLane: { x: 804, y: 370 }, immigrationQueue: { x: 976, y: 370 },
    immigrationCounter: { x: 1168, y: 370 }, lounge: { x: 398, y: 648 }, boardingQueue: { x: 682, y: 648 },
    boardingGate: { x: 642, y: 792 }, aircraft: { x: 642, y: 930 }
  };
  const preboardingStages = [
    { name: "entrance", from: "entrance", to: "checkinQueue", duration: 2.0 },
    { name: "checkinQueue", from: "checkinQueue", to: "checkinQueue", duration: 3.0, queue: "checkin" },
    { name: "checkinCounter", from: "checkinQueue", to: "checkinCounter", duration: 3.4, progress: [0, 1, 2] },
    { name: "securityQueue", from: "checkinCounter", to: "securityQueue", duration: 2.5, queue: "security" },
    { name: "securityLane", from: "securityQueue", to: "securityLane", duration: 3.2, progress: [3, 4] },
    { name: "immigrationQueue", from: "securityLane", to: "immigrationQueue", duration: 2.4, queue: "immigration" },
    { name: "immigrationCounter", from: "immigrationQueue", to: "immigrationCounter", duration: 3.8, progress: [5, 6, 7] },
    { name: "lounge", from: "immigrationCounter", to: "lounge", duration: 4.5 }
  ];
  const boardingStages = [
    { name: "boardingApproach", from: "lounge", to: "boardingQueue", duration: 2.4, perFlight: true },
    { name: "boardingLane", from: "boardingQueue", to: "boardingQueue", duration: 3.2, queue: "boarding", perFlight: true },
    { name: "boardingGate", from: "boardingQueue", to: "gate", duration: 3.2, progress: [8, 9, 10, 11, 12, 13], perFlight: true },
    { name: "walkToPlane", from: "gate", to: "planeStand", duration: 2.8, perFlight: true },
    { name: "boarded", from: "planeStand", to: "planeStand", duration: 999, perFlight: true }
  ];
  const preboardingDuration = preboardingStages.reduce((total, stage) => total + stage.duration, 0);
  const flightOps = new Map();
  const stepStages = [...preboardingStages, { name: "loungeHold", duration: 5.0 }, ...boardingStages.slice(0, -1)];
  const stepMoments = stepStages.reduce((moments, stage, index) => {
    const previous = index === 0 ? 0 : moments[index - 1];
    moments.push(previous + Math.min(stage.duration, 5));
    return moments;
  }, [0]);
  function ease(value) { return value < 0.5 ? 2 * value * value : 1 - Math.pow(-2 * value + 2, 2) / 2; }
  function interpolate(start, end, progress) {
    const curved = ease(Math.max(0, Math.min(progress, 1)));
    return { x: start.x + (end.x - start.x) * curved, y: start.y + (end.y - start.y) * curved };
  }
  function serviceIndexFor(passenger, group) {
    if (group === "checkin") return passenger.checkinIndex ?? passenger.index % SERVICE_COUNTS.checkin;
    if (group === "security") return passenger.securityIndex ?? passenger.index % SERVICE_COUNTS.security;
    if (group === "immigration") return passenger.immigrationIndex ?? passenger.index % SERVICE_COUNTS.immigration;
    return 0;
  }
  function queueLinePoint(queue, resourceIndex, slot, drift = 0) {
    const layouts = {
      checkin: { frontX: 300, y: 300, gapY: 45, gapX: 20 },
      security: { frontX: 660, y: 310, gapY: 62, gapX: 20 },
      immigration: { frontX: 1030, y: 300, gapY: 45, gapX: 20 },
    };
    const layout = layouts[queue];
    const sideOffset = drift % 2 === 0 ? -5 : 5;
    return { x: layout.frontX - slot * layout.gapX, y: layout.y + resourceIndex * layout.gapY + sideOffset };
  }
  function queuePosition(queue, passenger) {
    if (queue === "boarding") return boardingQueuePosition(passenger);
    if (!SERVICE_COUNTS[queue]) return points[`${queue}Queue`] || points[queue] || { x: 0, y: 0 };
    const resourceIndex = serviceIndexFor(passenger, queue);
    const slot = Math.floor(passenger.index / SERVICE_COUNTS[queue]) % 5;
    const drift = Math.floor(passenger.index / (SERVICE_COUNTS[queue] * 5));
    return queueLinePoint(queue, resourceIndex, slot, drift);
  }
  function queueGuidePosition(queue, index) {
    const resourceIndex = index % SERVICE_COUNTS[queue];
    const slot = Math.floor(index / SERVICE_COUNTS[queue]) % 4;
    return queueLinePoint(queue, resourceIndex, slot, index);
  }
  function servicePoint(stageName, passenger) {
    const group = SERVICE_STAGE_GROUP[stageName];
    const resourceIndex = serviceIndexFor(passenger, group);
    const positions = {
      checkinCounter: { x: 386, y: 300, gapY: 45 },
      securityLane: { x: 746, y: 310, gapY: 62 },
      immigrationCounter: { x: 1114, y: 300, gapY: 45 },
    };
    const base = positions[stageName];
    return { x: base.x, y: base.y + resourceIndex * base.gapY };
  }
  function serviceApproachPoint(stageName, passenger, progress) {
    const group = SERVICE_STAGE_GROUP[stageName];
    const normalized = Math.min(progress / 0.82, 1);
    return interpolate(queuePosition(group, passenger), servicePoint(stageName, passenger), normalized);
  }
  function gateIndexForFlight(flight) {
    return GATE_INDEX[flight.gate] ?? 0;
  }
  function loungePosition(passenger) {
    const column = passenger.flightSlot % 4;
    const row = passenger.flightIndex % 6;
    return { x: 284 + column * 35, y: 618 + row * 13 };
  }
  function boardingQueuePosition(passenger) {
    const column = passenger.flightSlot % 4;
    return { x: 610 + column * 24, y: 606 + passenger.gateIndex * 16 };
  }
  function gatePoint(gateIndex) {
    return { x: planeX(gateIndex), y: 792 };
  }
  function gateLaneEntryPoint(gateIndex) {
    return { x: planeX(gateIndex), y: 824 };
  }
  function gateLanePoint(gateIndex, slot) {
    const gate = gateLaneEntryPoint(gateIndex);
    const plane = { x: planeX(gateIndex), y: planeY(gateIndex) };
    const progress = 0.28 + (slot % 5) * 0.1;
    const rowOffset = (Math.floor(slot / 5) % 2) * 8 - 4;
    return {
      x: gate.x + rowOffset,
      y: gate.y + (plane.y - gate.y) * progress + rowOffset,
    };
  }
  function pointFor(name, passenger) {
    if (name === "checkinQueue") return queuePosition("checkin", passenger);
    if (name === "checkinCounter") return servicePoint("checkinCounter", passenger);
    if (name === "securityQueue") return queuePosition("security", passenger);
    if (name === "securityLane") return servicePoint("securityLane", passenger);
    if (name === "immigrationQueue") return queuePosition("immigration", passenger);
    if (name === "immigrationCounter") return servicePoint("immigrationCounter", passenger);
    if (name === "lounge") return loungePosition(passenger);
    if (name === "boardingQueue") return boardingQueuePosition(passenger);
    if (name === "gate") return gateLaneEntryPoint(passenger.gateIndex);
    if (name === "planeStand") return { x: planeX(passenger.gateIndex), y: planeY(passenger.gateIndex) };
    return points[name];
  }
  function phaseFromStages(stageList, localTime, passenger) {
    let cursor = 0;
    for (const stage of stageList) {
      const next = cursor + stage.duration;
      if (localTime <= next) {
        const progress = (localTime - cursor) / stage.duration;
        if (stage.queue === "boarding") return { stage, progress, position: boardingQueuePosition(passenger) };
        if (stage.name === "walkToPlane" && progress < 0.62) return { stage, progress, position: gateLanePoint(passenger.gateIndex, passenger.flightSlot) };
        if (SERVICE_STAGE_GROUP[stage.name]) {
          return { stage, progress, position: serviceApproachPoint(stage.name, passenger, progress) };
        }
        if (stage.queue && progress < 0.34) {
          return { stage, progress, position: interpolate(pointFor(stage.from, passenger), queuePosition(stage.queue, passenger), progress / 0.34) };
        }
        if (stage.queue) return { stage, progress, position: queuePosition(stage.queue, passenger) };
        return { stage, progress, position: interpolate(pointFor(stage.from, passenger), pointFor(stage.to, passenger), progress) };
      }
      cursor = next;
    }
    const last = stageList[stageList.length - 1];
    return { stage: last, progress: 1, position: pointFor(last.to, passenger) };
  }
  function terminalPressureSnapshot(elapsed) {
    const queues = { checkin: 0, security: 0, immigration: 0 };
    passengers.forEach((passenger) => {
      const localTime = elapsed - passenger.delay;
      if (localTime < 0 || localTime > preboardingDuration) return;
      const state = phaseFromStages(preboardingStages, localTime, passenger);
      if (state.stage.queue && Object.prototype.hasOwnProperty.call(queues, state.stage.queue)) {
        queues[state.stage.queue] += 1;
      }
    });
    const entries = [
      { key: "security", label: "Security", value: queues.security, weight: 1.25 },
      { key: "immigration", label: "Immigration", value: queues.immigration, weight: 1.05 },
      { key: "checkin", label: "Check-in", value: queues.checkin, weight: 0.75 },
    ];
    const worst = entries.slice().sort((left, right) => right.value * right.weight - left.value * left.weight)[0];
    const score = entries.reduce((total, item) => total + item.value * item.weight, 0) * scenarioRushMultiplier;
    return { queues, score, bottleneck: worst.value > 0 ? worst.label : "None" };
  }
  function initializeFlightOperations(allPassengers) {
    flights.forEach((flight) => {
      const totalPassengers = allPassengers.filter((passenger) => passenger.flightCode === flight.flight).length;
      flightOps.set(flight.flight, {
        totalPassengers,
        readyPassengers: 0,
        readyRatio: 0,
        clearanceAt: null,
        boardingAt: null,
        gateClosedAt: null,
        departedAt: null,
        turnaroundAt: null,
        estimatedDelayMinutes: flight.delayMinutes || 0,
        reason: flight.delayReason || "On schedule",
        bottleneck: flight.bottleneckStage || "None",
      });
    });
  }
  function updateFlightOperations(elapsed) {
    const pressure = terminalPressureSnapshot(elapsed);
    flights.forEach((flight) => {
      const ops = flightOps.get(flight.flight);
      if (!ops) return;
      const plannedClearance = flight.plannedClearanceStart ?? flight.clearanceStart ?? 34;
      const plannedDeparture = flight.plannedDepartedStart ?? flight.departedStart ?? plannedClearance + 30;
      const inboundReadyAt = plannedClearance + (flight.inboundHoldMinutes || 0);
      const forcedHoldUntil = flight.forcedDelay
        ? plannedClearance + Math.max(5, flight.forcedDelayMinutes || flight.delayMinutes || 0)
        : null;
      const minimumReadyRatio = flight.minimumReadyRatio ?? 0.72;
      const pressureLimit = flight.pressureHoldLimit ?? 10;
      const readyPassengers = passengers.filter((passenger) => (
        passenger.flightCode === flight.flight && passenger.delay + preboardingDuration <= elapsed
      )).length;
      const readyRatio = ops.totalPassengers ? readyPassengers / ops.totalPassengers : 1;
      ops.readyPassengers = readyPassengers;
      ops.readyRatio = readyRatio;

      if (ops.clearanceAt == null) {
        const lateEnoughToRelax = elapsed >= plannedClearance + 18;
        const readyOk = readyRatio >= minimumReadyRatio || lateEnoughToRelax;
        const inboundOk = elapsed >= inboundReadyAt;
        const pressureOk = pressure.score <= pressureLimit || elapsed >= plannedClearance + 16;
        const forcedOk = forcedHoldUntil == null || elapsed >= forcedHoldUntil;

        if (elapsed >= plannedClearance && readyOk && inboundOk && pressureOk && forcedOk) {
          ops.clearanceAt = elapsed;
          ops.boardingAt = elapsed + 2;
          ops.gateClosedAt = ops.boardingAt + 12 + Math.ceil(ops.totalPassengers * 0.25);
          ops.departedAt = ops.gateClosedAt + 8;
          ops.turnaroundAt = ops.departedAt + 12;
          ops.reason = elapsed > plannedClearance + 1 ? "Cleared after terminal recovery" : "Cleared for boarding";
          ops.bottleneck = "None";
        } else if (forcedHoldUntil != null && elapsed < forcedHoldUntil) {
          ops.reason = flight.delayReason || "Manual scenario hold";
          ops.bottleneck = flight.bottleneckStage || "Operations";
        } else if (elapsed < plannedClearance) {
          ops.reason = "On schedule";
          ops.bottleneck = "None";
        } else if (!inboundOk) {
          ops.reason = "Inbound Aircraft Not Ready";
          ops.bottleneck = flight.bottleneckStage || "Aircraft";
        } else if (!readyOk) {
          ops.reason = "Passengers Still in Terminal";
          ops.bottleneck = pressure.bottleneck;
        } else if (!pressureOk) {
          ops.reason = `${pressure.bottleneck} Congestion`;
          ops.bottleneck = pressure.bottleneck;
        } else {
          ops.reason = "Awaiting Operations Clearance";
          ops.bottleneck = pressure.bottleneck;
        }
      }

      const projectedDeparture = ops.departedAt ?? Math.max(plannedDeparture, elapsed + 22);
      const runtimeDelay = Math.max(0, Math.round(projectedDeparture - plannedDeparture));
      ops.estimatedDelayMinutes = flight.forcedDelay
        ? Math.max(flight.delayMinutes || 0, flight.forcedDelayMinutes || 0, runtimeDelay)
        : runtimeDelay;
    });
  }
  function phaseFor(localTime, passenger, elapsed) {
    if (localTime <= preboardingDuration) {
      return phaseFromStages(preboardingStages, localTime, passenger);
    }
    const readyAt = passenger.delay + preboardingDuration;
    const ops = flightOps.get(passenger.flightCode);
    const boardingAt = ops?.boardingAt;
    const releaseAt = boardingAt == null ? Infinity : Math.max(readyAt, boardingAt + passenger.flightSlot * boardingReleaseSpacing);
    if (elapsed < releaseAt) {
      return {
        stage: { name: "loungeHold", queue: "lounge", duration: Math.max(1, releaseAt - readyAt) },
        progress: Math.max(0, Math.min((elapsed - readyAt) / Math.max(1, releaseAt - readyAt), 1)),
        position: loungePosition(passenger),
        releaseAt
      };
    }
    return phaseFromStages(boardingStages, elapsed - releaseAt, passenger);
  }
  function timerText(state, passenger) {
    if (state.stage.name === "boarded") return "Boarded ✓";
    if (state.stage.name === "loungeHold") return `Waiting for ${passenger.flightCode} clearance`;
    if (state.stage.name === "boardingLane") return `Gate ${passenger.gate} queue`;
    if (state.stage.name === "boardingApproach") return `Released to Gate ${passenger.gate}`;
    if (state.stage.name === "walkToPlane") return `To ${passenger.gate} plane`;
    if (state.stage.queue || state.stage.name === "lounge")
      return `Waiting: ${(state.progress * state.stage.duration).toFixed(1)} min`;
    if (state.stage.progress) {
      const remaining = Math.max(0, state.stage.duration * (1 - state.progress));
      return `Service Remaining: ${remaining.toFixed(1)} min`;
    }
    return "";
  }
  function createPassenger(index, assignedFlight = null, assignedSlot = null, startDelay = null) {
    const element = document.createElement("div");
    const flight = assignedFlight || flights[index % flights.length];
    const flightIndex = flights.indexOf(flight);
    const flightSlot = assignedSlot ?? Math.floor(index / flights.length);
    const gateIndex = gateIndexForFlight(flight);
    const passengerId = `P${String(index + 1).padStart(3, "0")}`;
    const flightCode = flight.flight;
    const gate = flight.gate;
    const checkinIndex = index % SERVICE_COUNTS.checkin;
    const securityIndex = (flightIndex + flightSlot + index) % SERVICE_COUNTS.security;
    const immigrationIndex = (flightIndex * 2 + flightSlot + index) % SERVICE_COUNTS.immigration;
    element.className = "passenger";
    element.dataset.id = passengerId; element.dataset.flight = flightCode;
    element.dataset.label = `${passengerId} / ${flightCode}`;
    element.style.background = flight.color || "var(--yellow)";
    element.style.borderColor = flight.color || "#fff0b8";
    if (flight.delayMinutes > 0 || flight.delayProbability >= 0.55) element.classList.add("delayed-flight");
    const timer = document.createElement("span");
    timer.className = "passenger-timer"; element.appendChild(timer); map.appendChild(element);
    return {
      index, element, timer, delay: startDelay ?? index * passengerArrivalSpacing,
      flight, flightCode, gate, flightIndex, flightSlot, gateIndex,
      checkinIndex, securityIndex, immigrationIndex,
    };
  }
  function renderFlightGroups() {
    flightGroupStrip.innerHTML = flights.map((flight) => `
      <span class="flight-group-badge" style="--flight-color:${flight.color || "#3bd7ff"}">
        ${flight.flight} / Gate ${flight.gate} / ${flight.destination}
      </span>`).join("");
  }
  function registerFlightOperation(flight, flightPassengers) {
    flightOps.set(flight.flight, {
      totalPassengers: flightPassengers.length,
      readyPassengers: 0,
      readyRatio: 0,
      clearanceAt: null,
      boardingAt: null,
      gateClosedAt: null,
      departedAt: null,
      turnaroundAt: null,
      estimatedDelayMinutes: flight.delayMinutes || 0,
      reason: flight.delayReason || "On schedule",
      bottleneck: flight.bottleneckStage || "None",
    });
  }
  function nextDestinationFor(index, cycle) {
    const destinations = ["Dubai", "Doha", "Istanbul", "London", "Singapore", "Karachi", "Lahore", "Bangkok"];
    return destinations[(index + cycle) % destinations.length];
  }
  function rotateGateFlight(flight, index, elapsed) {
    const gate = flight.gate;
    const gateIndex = gateIndexForFlight(flight);
    const cycle = (flight.cycle || 0) + 1;
    const newCode = flight.nextFlight || `NX-${720 + index * 37 + cycle * 111}`;
    const oldCode = flight.flight;
    passengers.forEach((passenger) => {
      if (passenger.flightCode === oldCode) {
        passenger.retired = true;
        passenger.element.style.opacity = "0";
      }
    });
    flightOps.delete(flight.flight);

    flight.flight = newCode;
    flight.nextFlight = `NX-${720 + index * 37 + (cycle + 1) * 111}`;
    flight.destination = nextDestinationFor(index, cycle);
    flight.scheduledMinutes = 9 * 60 + Math.round(elapsed + 34 + index * 4);
    flight.delayMinutes = 0;
    flight.forcedDelay = false;
    flight.forcedDelayMinutes = 0;
    flight.delayProbability = Math.max(0.18, Math.min(0.74, (flight.delayProbability || 0.36) * 0.72 + 0.12 + (cycle % 3) * 0.04));
    flight.riskLevel = flight.delayProbability >= 0.55 ? "High" : flight.delayProbability >= 0.35 ? "Medium" : "Low";
    flight.delayReason = "On schedule";
    flight.bottleneckStage = "None";
    flight.plannedClearanceStart = elapsed + 19 + index * 2;
    flight.plannedBoardingStart = flight.plannedClearanceStart + 5;
    flight.plannedGateClosedStart = flight.plannedBoardingStart + 14;
    flight.plannedDepartedStart = flight.plannedGateClosedStart + 9;
    flight.clearanceStart = flight.plannedClearanceStart;
    flight.boardingStart = flight.plannedBoardingStart;
    flight.gateClosedStart = flight.plannedGateClosedStart;
    flight.departedStart = flight.plannedDepartedStart;
    flight.turnaroundStart = flight.plannedDepartedStart + 12;
    flight.inboundHoldMinutes = cycle % 2 === 0 ? 6 : 2;
    flight.minimumReadyRatio = 0.72;
    flight.pressureHoldLimit = Math.max(8, 14 - index);
    flight.cycle = cycle;

    const newPassengers = Array.from({ length: passengerBatchSize }, (_, slot) => (
      createPassenger(nextPassengerIndex++, flight, slot, elapsed + slot * passengerArrivalSpacing)
    ));
    passengers.push(...newPassengers);
    registerFlightOperation(flight, newPassengers);
    renderFlightGroups();

    const plane = planeEls[index];
    const stand = standEls[gateIndex];
    if (plane) {
      plane.classList.remove("departed", "turnaround", "delayed", "boarding");
      plane.querySelector(".plane-label").textContent = flight.flight;
      plane.querySelector(".plane-tag").textContent = "ON TIME";
      plane.querySelector(".plane-icon").style.color = flight.color || "#3bd7ff";
    }
    if (stand) {
      stand.classList.remove("departed", "turnaround", "delayed");
      const standFlight = stand.querySelector("[data-stand-flight]");
      if (standFlight) standFlight.textContent = `${flight.flight} active`;
    }
  }
  function createQueueDots() {
    [["checkin", 12], ["security", 8], ["immigration", 12]].forEach(([queue, count]) => {
      for (let index = 0; index < count; index += 1) {
        const dot = document.createElement("div"); const position = queueGuidePosition(queue, index);
        dot.className = "queue-dot"; dot.style.left = `${position.x + 7}px`; dot.style.top = `${position.y + 7}px`;
        map.appendChild(dot);
      }
    });
  }
  function drawServiceLink(group, stageName, resourceIndex) {
    const fakePassenger = {
      index: resourceIndex,
      checkinIndex: group === "checkin" ? resourceIndex : 0,
      securityIndex: group === "security" ? resourceIndex : 0,
      immigrationIndex: group === "immigration" ? resourceIndex : 0,
    };
    const start = queueLinePoint(group, resourceIndex, 0, 0);
    const end = servicePoint(stageName, fakePassenger);
    const from = { x: start.x + 18, y: start.y + 9 };
    const to = { x: end.x + 9, y: end.y + 9 };
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const link = document.createElement("div");
    link.className = "service-link";
    link.style.left = `${from.x}px`;
    link.style.top = `${from.y}px`;
    link.style.width = `${Math.hypot(dx, dy)}px`;
    link.style.transform = `rotate(${Math.atan2(dy, dx)}rad)`;
    map.appendChild(link);
  }
  function createServiceLinks() {
    [["checkin", "checkinCounter"], ["security", "securityLane"], ["immigration", "immigrationCounter"]]
      .forEach(([group, stageName]) => {
        for (let index = 0; index < SERVICE_COUNTS[group]; index += 1) {
          drawServiceLink(group, stageName, index);
        }
      });
  }
  // ----- Aircraft stands: one plane per flight (one gate = one plane) -----
  const planeEls = [];
  const standEls = [];
  const gateLineEls = [];
  const apronLeft = 24;
  const apronWidth = 1238;
  function planeX(index) { return apronLeft + (apronWidth / GATES.length) * (index + 0.5); }
  function planeY(index) { return 932; }
  function drawGateConnector(gate, gateIndex) {
    const start = gateLaneEntryPoint(gateIndex);
    const end = { x: planeX(gateIndex), y: planeY(gateIndex) };
    const line = document.createElement("div");
    line.className = "gate-connector";
    line.dataset.gate = gate;
    line.style.left = `${start.x}px`;
    line.style.top = `${start.y}px`;
    line.style.height = `${Math.max(0, end.y - start.y - 42)}px`;
    map.appendChild(line);
    gateLineEls[gateIndex] = line;
  }
  function setupPlanes() {
    GATES.forEach((gate, gateIndex) => {
      const x = planeX(gateIndex);
      const y = planeY(gateIndex);
      drawGateConnector(gate, gateIndex);
      const stand = document.createElement("div");
      stand.className = "plane-stand";
      stand.style.left = `${x}px`;
      stand.style.top = `${y}px`;
      stand.innerHTML = `<div class="stand-gate">Gate ${gate}</div><div class="stand-flight" data-stand-flight>Available</div>`;
      map.appendChild(stand);
      standEls[gateIndex] = stand;
    });

    flights.forEach((flight, index) => {
      const x = planeX(gateIndexForFlight(flight));
      const y = planeY(gateIndexForFlight(flight));
      const plane = document.createElement("div");
      plane.className = "plane";
      plane.style.left = `${x}px`;
      plane.style.top = `${y}px`;
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
      const stand = standEls[gateIndexForFlight(flight)];
      if (!plane || !stand) return;
      const state = flightStatusFor(flight, elapsed);
      const isDelayed = state.status === "DELAYED" || state.delayMinutes > 0;
      const departed = state.status === "DEPARTED";
      const boarding = state.status === "BOARDING" || state.status === "GATE CLOSED";
      const replacementDue = departed && elapsed >= (state.turnaroundAt || (flight.turnaroundStart || flight.departedStart + 12));
      const label = plane.querySelector(".plane-label");
      const tag = plane.querySelector(".plane-tag");
      const standFlight = stand.querySelector("[data-stand-flight]");

      plane.classList.remove("departed", "delayed", "boarding", "turnaround");
      stand.classList.remove("departed", "delayed", "turnaround");

      if (replacementDue) {
        rotateGateFlight(flight, index, elapsed);
        return;
      } else if (departed) {
        plane.classList.add("departed");
        stand.classList.add("departed");
        label.textContent = flight.flight;
        tag.textContent = "DEPARTED";
        if (standFlight) standFlight.textContent = `${flight.flight} departed`;
      } else {
        plane.classList.toggle("delayed", isDelayed);
        plane.classList.toggle("boarding", boarding);
        stand.classList.toggle("delayed", isDelayed);
        label.textContent = flight.flight;
        tag.textContent = state.status;
        if (standFlight) standFlight.textContent = `${flight.flight} active`;
      }
    });
  }
  function resetProgressBars() { progressBars.forEach((bar) => { bar.style.width = "0%"; }); }
  function updateProgress(stage, progress, passenger) {
    if (!stage.progress) return;
    const barIndex = stage.name === "boardingGate"
      ? stage.progress[passenger.gateIndex % stage.progress.length]
      : stage.progress[serviceIndexFor(passenger, SERVICE_STAGE_GROUP[stage.name]) % stage.progress.length];
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
  function passengerMatchesResource(item, resource, mode) {
    if (resource.group === "boarding") {
      const inQueue = mode === "queue" && item.state.stage.queue === resource.queue;
      const inService = mode === "service" && item.state.stage.name === resource.serviceStage;
      return (inQueue || inService) && item.passenger.gateIndex === resource.gateIndex;
    }
    if (mode === "queue") {
      return item.state.stage.queue === resource.queue && serviceIndexFor(item.passenger, resource.group) === resource.index;
    }
    return item.state.stage.name === resource.serviceStage && serviceIndexFor(item.passenger, resource.group) === resource.index;
  }
  function buildResourceState(resource, passengerStates, elapsed) {
    const card = resourceCards.find((item) => item.dataset.resource === resource.id);
    const closed = isClosed(resource, elapsed);
    const queuedPassengers = passengerStates.filter((item) => passengerMatchesResource(item, resource, "queue"));
    const activePassenger = passengerStates.find((item) => passengerMatchesResource(item, resource, "service"));
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
    const ops = flightOps.get(flight.flight);
    const plannedClearance = flight.plannedClearanceStart ?? flight.clearanceStart ?? 34;
    let status = "ON TIME";
    if (ops?.departedAt != null && elapsed >= ops.departedAt) status = "DEPARTED";
    else if (ops?.gateClosedAt != null && elapsed >= ops.gateClosedAt) status = "GATE CLOSED";
    else if (ops?.boardingAt != null && elapsed >= ops.boardingAt) status = "BOARDING";
    else if (ops?.clearanceAt != null && elapsed >= ops.clearanceAt) status = "CLEARED";
    else if (flight.forcedDelay && (ops?.estimatedDelayMinutes || flight.delayMinutes || 0) > 0) status = "DELAYED";
    else if (elapsed >= plannedClearance) status = "DELAYED";

    const delayMinutes = ops?.estimatedDelayMinutes ?? flight.delayMinutes ?? 0;
    return {
      status,
      estimatedMinutes: flight.scheduledMinutes + delayMinutes,
      delayMinutes,
      reason: ops?.reason || flight.delayReason || "None",
      bottleneck: ops?.bottleneck || flight.bottleneckStage || "None",
      turnaroundAt: ops?.turnaroundAt ?? flight.turnaroundStart,
      readyPassengers: ops?.readyPassengers ?? 0,
      totalPassengers: ops?.totalPassengers ?? 0,
    };
  }
  function flightStatusClass(status) { return status.toLowerCase().replaceAll(" ", "-"); }
  function updateFlightBoard(elapsed, passengerStates) {
    const tick = Math.floor(elapsed * 2);
    if (tick === lastFlightTick) return;
    lastFlightTick = tick;
    flightClock.textContent = formatTime(9 * 60 + elapsed);
    flightBoardBody.innerHTML = flights.map((flight) => {
      const state = flightStatusFor(flight, elapsed);
      const className = flightStatusClass(state.status);
      const waiting = passengerStates.filter((item) => (
        item.passenger.flightCode === flight.flight
        && ["loungeHold", "boardingLane", "boardingGate"].includes(item.state.stage.name)
      )).length;
      return `
        <tr class="flight-row ${className}">
          <td>${flight.flight}</td><td>${flight.destination}</td><td>${flight.gate}</td>
          <td>${formatTime(flight.scheduledMinutes)}</td><td>${formatTime(state.estimatedMinutes)}</td>
          <td><span class="flight-status-badge ${className}">${state.status}</span></td>
          <td>${state.reason}</td><td>${waiting} / ${state.totalPassengers}</td>
          <td>${state.bottleneck}</td>
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
  function fullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement;
  }
  function updateFullscreenButton() {
    const active = Boolean(fullscreenElement()) || fullscreenTarget.classList.contains("fullscreen-fallback");
    fullscreenButton.classList.toggle("active", active);
    fullscreenButton.textContent = active
      ? (fullscreenTarget.classList.contains("fullscreen-fallback") ? "Exit Theater" : "Exit Full Screen")
      : "Full Screen";
  }
  async function enterNativeFullscreen() {
    if (fullscreenTarget.requestFullscreen) return fullscreenTarget.requestFullscreen();
    if (fullscreenTarget.webkitRequestFullscreen) return fullscreenTarget.webkitRequestFullscreen();
    if (fullscreenTarget.msRequestFullscreen) return fullscreenTarget.msRequestFullscreen();
    throw new Error("Native fullscreen is not available");
  }
  async function exitNativeFullscreen() {
    if (document.exitFullscreen) return document.exitFullscreen();
    if (document.webkitExitFullscreen) return document.webkitExitFullscreen();
    if (document.msExitFullscreen) return document.msExitFullscreen();
  }
  fullscreenButton.addEventListener("click", async () => {
    try {
      if (fullscreenElement()) {
        await exitNativeFullscreen();
      } else if (fullscreenTarget.classList.contains("fullscreen-fallback")) {
        fullscreenTarget.classList.remove("fullscreen-fallback");
      } else {
        await enterNativeFullscreen();
      }
    } catch (error) {
      fullscreenTarget.classList.toggle("fullscreen-fallback");
    }
    updateFullscreenButton();
  });
  document.addEventListener("fullscreenchange", updateFullscreenButton);
  document.addEventListener("webkitfullscreenchange", updateFullscreenButton);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && fullscreenTarget.classList.contains("fullscreen-fallback")) {
      fullscreenTarget.classList.remove("fullscreen-fallback");
      updateFullscreenButton();
    }
  });
  createServiceLinks();
  createQueueDots();
  renderFlightGroups();
  setupPlanes();
  const passengers = Array.from({ length: passengerCount }, (_, index) => createPassenger(index));
  let nextPassengerIndex = passengerCount;
  initializeFlightOperations(passengers);
  let simulationElapsed = 0;
  let lastFrameTime = performance.now();
  function animate(now) {
    const frameDelta = Math.min((now - lastFrameTime) / 1000, 0.08);
    lastFrameTime = now;
    if (simulationMode === "auto") simulationElapsed += frameDelta * speed;
    const elapsed = simulationElapsed;
    const passengerStates = [];
    updateFlightOperations(elapsed);
    resetProgressBars();
    passengers.forEach((passenger) => {
      if (passenger.retired) { passenger.element.style.opacity = "0"; return; }
      const localTime = elapsed - passenger.delay;
      if (localTime < 0) { passenger.element.style.opacity = "0"; return; }
      const state = phaseFor(localTime, passenger, elapsed);
      const boarded = state.stage.name === "boarded";
      // Once boarded, the passenger fades into their aircraft and leaves the floor.
      passenger.element.style.opacity = boarded ? "0" : "1";
      passenger.element.dataset.stage = state.stage.name;
      passenger.element.style.transform = `translate(${state.position.x}px, ${state.position.y}px)`;
      passenger.element.classList.toggle("boarded", state.stage.name === "walkToPlane" || boarded);
      passenger.timer.textContent = timerText(state, passenger);
      passengerStates.push({ passenger, state });
      updateProgress(state.stage, state.progress, passenger);
    });
    updateResourceCards(passengerStates, elapsed);
    updateFlightBoard(elapsed, passengerStates);
    updatePlanes(elapsed);
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
})();
</script>
</body>
</html>
"""
