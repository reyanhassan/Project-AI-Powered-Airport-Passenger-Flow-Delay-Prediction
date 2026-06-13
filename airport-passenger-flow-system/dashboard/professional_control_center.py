"""Professional Airport Control Center page for the Streamlit dashboard."""

import streamlit as st
import streamlit.components.v1 as components


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
    border: 1px solid rgba(59, 215, 255, 0.42);
    background: rgba(59, 215, 255, 0.11);
    border-radius: 7px;
    padding: 7px;
    margin-top: 8px;
    color: #e8f8ff;
    font-size: 11px;
    font-weight: 800;
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
      <section class="zone checkin-counters"><div class="zone-title">Check-in Counters</div><div class="desk">Counter 1</div><div class="desk">Counter 2</div><div class="desk">Counter 3</div></section>
      <section class="zone security-queue"><div class="zone-title">Security Queue</div><div class="zone-note">Screening line</div></section>
      <section class="zone security-lanes"><div class="zone-title">Security Lanes</div><div class="desk">Lane 1</div><div class="desk">Lane 2</div></section>
      <section class="zone immigration-queue"><div class="zone-title">Immigration Queue</div><div class="zone-note">Passport control line</div></section>
      <section class="zone immigration-counters"><div class="zone-title">Immigration Counters</div><div class="desk">Immigration 1</div><div class="desk">Immigration 2</div><div class="desk">Immigration 3</div></section>
      <section class="zone lounge"><div class="zone-title">Waiting Lounge</div><div class="zone-note">Cleared passengers</div></section>
      <section class="zone boarding-queue"><div class="zone-title">Boarding Queue</div><div class="zone-note">Gate line-up</div></section>
      <section class="zone boarding-gates"><div class="zone-title">Boarding Gates</div><div class="desk">Gate A1</div><div class="desk">Gate A2</div></section>
      <section class="zone aircraft"><div class="zone-title">Aircraft</div><div class="zone-note">Boarded area</div><div class="aircraft-body"></div></section>
    </div>
    <div class="legend">
      <span>Blue route: active passenger flow</span>
      <span>Yellow dots: passengers</span>
      <span>Green dots: boarded</span>
      <span>Progress bars activate at counters</span>
    </div>
  </div>
</body>
</html>
"""


def render_professional_airport_control_center_page() -> None:
    """Render the professional airport control center page."""

    components.html(build_professional_control_center_html(), height=980, scrolling=True)
