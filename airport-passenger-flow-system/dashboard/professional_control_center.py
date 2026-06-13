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
    { flight: "PK-302", destination: "Karachi", gate: "A1", scheduledMinutes: 9 * 60 + 10, delayMinutes: 0, delayStart: 12, boardingStart: 24, gateClosedStart: 43, departedStart: 58 },
    { flight: "AP-144", destination: "Lahore", gate: "A2", scheduledMinutes: 9 * 60 + 25, delayMinutes: 18, delayStart: 10, boardingStart: 54, gateClosedStart: 70, departedStart: 82 },
    { flight: "SA-781", destination: "Dubai", gate: "B3", scheduledMinutes: 9 * 60 + 40, delayMinutes: 0, delayStart: 16, boardingStart: 36, gateClosedStart: 55, departedStart: 72 },
    { flight: "ER-219", destination: "Islamabad", gate: "C1", scheduledMinutes: 10 * 60 + 5, delayMinutes: 24, delayStart: 18, boardingStart: 66, gateClosedStart: 80, departedStart: 92 },
    { flight: "GB-508", destination: "Doha", gate: "A4", scheduledMinutes: 10 * 60 + 20, delayMinutes: 0, delayStart: 24, boardingStart: 52, gateClosedStart: 75, departedStart: 90 },
    { flight: "PK-419", destination: "Jeddah", gate: "D2", scheduledMinutes: 10 * 60 + 35, delayMinutes: 35, delayStart: 22, boardingStart: 78, gateClosedStart: 94, departedStart: 108 }
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
    const cycle = elapsed % 116;
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

    components.html(build_professional_control_center_html(), height=1380, scrolling=True)
