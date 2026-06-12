"""Advanced animated airport simulation page for the Streamlit dashboard."""

import streamlit as st
import streamlit.components.v1 as components


def build_advanced_airport_html() -> str:
    """Build the custom HTML/CSS airport map component."""

    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  :root {
    --bg: #050b12;
    --panel: #071520;
    --panel-2: #0b1f2c;
    --line: #1e6b99;
    --neon: #38c5ff;
    --muted: #8ca6b9;
    --text: #edf8ff;
    --green: #39d98a;
    --red: #ff5c6c;
    --blue: #3aa7ff;
    --yellow: #ffd166;
    --gray: #8b98a6;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at top left, #102b3d 0, var(--bg) 36%);
    color: var(--text);
    font-family: Inter, Segoe UI, Arial, sans-serif;
  }
  .control-room {
    min-height: 860px;
    padding: 18px;
    background:
      linear-gradient(rgba(56, 197, 255, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56, 197, 255, 0.04) 1px, transparent 1px),
      var(--bg);
    background-size: 36px 36px;
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 16px;
  }
  .title {
    font-size: 34px;
    font-weight: 900;
    color: var(--text);
    letter-spacing: 0;
  }
  .subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-top: 4px;
  }
  .system-badge {
    border: 1px solid var(--line);
    color: var(--neon);
    border-radius: 8px;
    padding: 10px 14px;
    background: rgba(7, 21, 32, 0.86);
    box-shadow: 0 0 22px rgba(56, 197, 255, 0.14);
    font-weight: 800;
  }
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 16px;
  }
  .metric-card {
    background: rgba(7, 21, 32, 0.94);
    border: 1px solid rgba(56, 197, 255, 0.28);
    border-radius: 8px;
    padding: 12px;
    min-height: 82px;
  }
  .metric-label {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    font-weight: 800;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 900;
    margin-top: 8px;
    color: var(--text);
  }
  .main-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.8fr);
    gap: 14px;
  }
  .airport-map {
    position: relative;
    height: 620px;
    border: 1px solid rgba(56, 197, 255, 0.32);
    border-radius: 12px;
    background:
      linear-gradient(135deg, rgba(56, 197, 255, 0.06), transparent 36%),
      rgba(7, 21, 32, 0.92);
    overflow: hidden;
    box-shadow: inset 0 0 45px rgba(56, 197, 255, 0.07);
  }
  .zone {
    position: absolute;
    border: 1px solid rgba(56, 197, 255, 0.45);
    border-radius: 10px;
    background: rgba(11, 31, 44, 0.82);
    padding: 10px;
    box-shadow: inset 0 0 20px rgba(56, 197, 255, 0.06);
  }
  .zone-label {
    font-size: 13px;
    color: var(--neon);
    font-weight: 900;
    text-transform: uppercase;
  }
  .zone-detail {
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
  }
  .entrance { left: 24px; top: 248px; width: 135px; height: 120px; }
  .checkin-queue { left: 195px; top: 220px; width: 155px; height: 180px; }
  .counters { left: 386px; top: 176px; width: 175px; height: 270px; }
  .security-queue { left: 596px; top: 220px; width: 150px; height: 180px; }
  .security-lanes { left: 782px; top: 176px; width: 170px; height: 270px; }
  .lounge { left: 230px; top: 470px; width: 360px; height: 120px; }
  .gates { left: 640px; top: 470px; width: 230px; height: 120px; }
  .aircraft { left: 898px; top: 480px; width: 170px; height: 100px; }
  .path-line {
    position: absolute;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent);
    opacity: 0.7;
    transform-origin: left center;
  }
  .path-1 { left: 152px; top: 306px; width: 792px; }
  .path-2 { left: 500px; top: 428px; width: 410px; transform: rotate(22deg); }
  .counter-desk,
  .lane-desk,
  .gate-desk {
    background: rgba(56, 197, 255, 0.12);
    border: 1px solid rgba(56, 197, 255, 0.42);
    border-radius: 7px;
    padding: 8px;
    margin-top: 9px;
    color: #dff6ff;
    font-weight: 800;
    font-size: 12px;
  }
  .aircraft-shape {
    position: absolute;
    right: 18px;
    bottom: 18px;
    width: 96px;
    height: 38px;
    border-radius: 50% 12px 12px 50%;
    background: linear-gradient(90deg, #c7e8ff, #6fbaf0);
    box-shadow: 0 0 24px rgba(111, 186, 240, 0.28);
  }
  .aircraft-shape::before,
  .aircraft-shape::after {
    content: "";
    position: absolute;
    background: #6fbaf0;
    width: 42px;
    height: 10px;
    left: 38px;
  }
  .aircraft-shape::before { top: -10px; transform: rotate(-22deg); }
  .aircraft-shape::after { bottom: -10px; transform: rotate(22deg); }
  .side-panel {
    display: grid;
    grid-template-rows: 1fr 1fr;
    gap: 14px;
  }
  .panel {
    background: rgba(7, 21, 32, 0.94);
    border: 1px solid rgba(56, 197, 255, 0.28);
    border-radius: 10px;
    padding: 12px;
    min-height: 0;
  }
  .panel-title {
    color: var(--neon);
    font-weight: 900;
    margin-bottom: 10px;
    text-transform: uppercase;
    font-size: 13px;
  }
  .placeholder-text {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
  }
</style>
</head>
<body>
  <div class="control-room">
    <div class="header">
      <div>
        <div class="title">Advanced Animated Airport</div>
        <div class="subtitle">Operations control room | passenger flow and flight readiness monitor</div>
      </div>
      <div class="system-badge">LIVE OPS READY</div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">Total Passengers</div><div class="metric-value" id="metric-total">0</div></div>
      <div class="metric-card"><div class="metric-label">Check-in</div><div class="metric-value" id="metric-checkin">0</div></div>
      <div class="metric-card"><div class="metric-label">Security</div><div class="metric-value" id="metric-security">0</div></div>
      <div class="metric-card"><div class="metric-label">Lounge</div><div class="metric-value" id="metric-lounge">0</div></div>
      <div class="metric-card"><div class="metric-label">Boarded</div><div class="metric-value" id="metric-boarded">0</div></div>
      <div class="metric-card"><div class="metric-label">Avg Wait</div><div class="metric-value" id="metric-wait">0m</div></div>
    </div>

    <div class="main-grid">
      <div class="airport-map" id="airport-map">
        <div class="path-line path-1"></div>
        <div class="path-line path-2"></div>
        <section class="zone entrance"><div class="zone-label">Entrance</div><div class="zone-detail">Passenger arrivals</div></section>
        <section class="zone checkin-queue"><div class="zone-label">Check-in Queue</div><div class="zone-detail">Queue visualization</div></section>
        <section class="zone counters"><div class="zone-label">Check-in Counters</div><div class="counter-desk">Counter 1</div><div class="counter-desk">Counter 2</div><div class="counter-desk">Counter 3</div></section>
        <section class="zone security-queue"><div class="zone-label">Security Queue</div><div class="zone-detail">Screening line</div></section>
        <section class="zone security-lanes"><div class="zone-label">Security Lanes</div><div class="lane-desk">Lane 1</div><div class="lane-desk">Lane 2</div></section>
        <section class="zone lounge"><div class="zone-label">Waiting Lounge</div><div class="zone-detail">Passengers cleared security</div></section>
        <section class="zone gates"><div class="zone-label">Boarding Gates</div><div class="gate-desk">Gate A1</div><div class="gate-desk">Gate A2</div></section>
        <section class="zone aircraft"><div class="zone-label">Aircraft</div><div class="zone-detail">Boarded area</div><div class="aircraft-shape"></div></section>
      </div>

      <aside class="side-panel">
        <section class="panel">
          <div class="panel-title">Digital Flight Board</div>
          <div class="placeholder-text">Flight board will activate with simulation data.</div>
        </section>
        <section class="panel">
          <div class="panel-title">Live Event Log</div>
          <div class="placeholder-text">Passenger events will stream here during animation.</div>
        </section>
      </aside>
    </div>
  </div>
</body>
</html>
"""


def render_advanced_animated_airport_page() -> None:
    """Render the advanced animated airport simulation page."""

    components.html(build_advanced_airport_html(), height=900, scrolling=True)
