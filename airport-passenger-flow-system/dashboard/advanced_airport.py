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
    grid-template-columns: repeat(7, minmax(0, 1fr));
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
  .flight-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .flight-table th {
    color: var(--neon);
    background: rgba(56, 197, 255, 0.08);
    text-align: left;
    padding: 8px 6px;
    border-bottom: 1px solid rgba(56, 197, 255, 0.24);
  }
  .flight-table td {
    padding: 8px 6px;
    border-bottom: 1px solid rgba(56, 197, 255, 0.12);
    color: var(--text);
  }
  .flight-badge {
    display: inline-block;
    border-radius: 5px;
    padding: 3px 7px;
    min-width: 78px;
    text-align: center;
    font-size: 10px;
    font-weight: 900;
  }
  .status-on-time { background: var(--green); color: #02120a; }
  .status-delayed { background: var(--red); color: #fff; }
  .status-boarding { background: var(--blue); color: #fff; }
  .status-gate-closed { background: var(--yellow); color: #071018; }
  .status-departed { background: var(--gray); color: #fff; }
  .event-log {
    height: 254px;
    overflow: hidden;
    display: flex;
    flex-direction: column-reverse;
    gap: 6px;
    padding-right: 4px;
  }
  .event-item {
    border-left: 3px solid var(--neon);
    background: rgba(56, 197, 255, 0.08);
    border-radius: 6px;
    padding: 7px 8px;
    color: #dff6ff;
    font-size: 12px;
    line-height: 1.35;
  }
  .event-time {
    color: var(--yellow);
    font-weight: 900;
    margin-right: 6px;
  }
  .passenger {
    position: absolute;
    left: 0;
    top: 0;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--yellow);
    border: 2px solid #fff3bf;
    box-shadow: 0 0 14px rgba(255, 209, 102, 0.65);
    transform: translate(-40px, -40px);
    transition: opacity 0.2s ease;
    z-index: 10;
  }
  .passenger::after {
    content: attr(data-id);
    position: absolute;
    left: 50%;
    top: 20px;
    transform: translateX(-50%);
    color: #dff6ff;
    font-size: 9px;
    font-weight: 900;
    white-space: nowrap;
    text-shadow: 0 1px 4px #000;
  }
  .passenger.boarded {
    background: var(--green);
    border-color: #d6ffe9;
    box-shadow: 0 0 14px rgba(57, 217, 138, 0.58);
  }
  .queue-dot {
    position: absolute;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(56, 197, 255, 0.7);
    box-shadow: 0 0 10px rgba(56, 197, 255, 0.65);
    z-index: 8;
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
      <div class="metric-card"><div class="metric-label">Delayed Flights</div><div class="metric-value" id="metric-delayed">0</div></div>
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
          <table class="flight-table">
            <thead>
              <tr>
                <th>Flight</th>
                <th>Destination</th>
                <th>Gate</th>
                <th>Sched</th>
                <th>Est</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="flight-board-body"></tbody>
          </table>
        </section>
        <section class="panel">
          <div class="panel-title">Live Event Log</div>
          <div class="event-log" id="event-log"></div>
        </section>
      </aside>
    </div>
  </div>
<script>
(() => {
  const config = {
    passengerCount: 36,
    speed: 1.0,
    delayProbability: 0.28,
    autoStart: true
  };
  const flights = [
    { flight: "PK-302", destination: "Karachi", gate: "A1", scheduled: "09:20", estimate: "09:32", status: "DELAYED", delay: 12 },
    { flight: "PA-117", destination: "Lahore", gate: "A2", scheduled: "09:45", estimate: "09:45", status: "BOARDING", delay: 0 },
    { flight: "ER-640", destination: "Dubai", gate: "B1", scheduled: "10:05", estimate: "10:05", status: "ON TIME", delay: 0 },
    { flight: "QR-811", destination: "Doha", gate: "B2", scheduled: "10:25", estimate: "10:25", status: "GATE CLOSED", delay: 0 },
    { flight: "TK-905", destination: "Istanbul", gate: "C1", scheduled: "10:50", estimate: "11:08", status: "DELAYED", delay: 18 },
    { flight: "EK-615", destination: "Dubai", gate: "C2", scheduled: "11:10", estimate: "11:10", status: "DEPARTED", delay: 0 }
  ];

  const map = document.getElementById("airport-map");
  const flightBoardBody = document.getElementById("flight-board-body");
  const eventLog = document.getElementById("event-log");
  const metrics = {
    total: document.getElementById("metric-total"),
    checkin: document.getElementById("metric-checkin"),
    security: document.getElementById("metric-security"),
    lounge: document.getElementById("metric-lounge"),
    boarded: document.getElementById("metric-boarded"),
    wait: document.getElementById("metric-wait"),
    delayed: document.getElementById("metric-delayed")
  };
  const points = {
    entrance: { x: 82, y: 306 },
    checkQueue: { x: 252, y: 306 },
    counter: { x: 472, y: 306 },
    securityQueue: { x: 660, y: 306 },
    securityLane: { x: 864, y: 306 },
    lounge: { x: 416, y: 532 },
    gate: { x: 752, y: 532 },
    aircraft: { x: 986, y: 532 }
  };
  const phases = [
    { name: "entrance", from: "entrance", to: "checkQueue", duration: 2.2 },
    { name: "checkin", from: "checkQueue", to: "counter", duration: 5.0, queue: "checkin" },
    { name: "counter", from: "counter", to: "securityQueue", duration: 3.2 },
    { name: "security", from: "securityQueue", to: "securityLane", duration: 4.6, queue: "security" },
    { name: "lane", from: "securityLane", to: "lounge", duration: 3.8 },
    { name: "lounge", from: "lounge", to: "gate", duration: 5.0 },
    { name: "gate", from: "gate", to: "aircraft", duration: 4.8, queue: "gate" },
    { name: "boarded", from: "aircraft", to: "aircraft", duration: 999 }
  ];
  const totalDuration = phases.slice(0, -1).reduce((sum, phase) => sum + phase.duration, 0);

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

  function queuePosition(queueName, passengerIndex) {
    const column = passengerIndex % 5;
    const row = Math.floor(passengerIndex / 5) % 6;
    if (queueName === "checkin") return { x: 218 + column * 24, y: 262 + row * 22 };
    if (queueName === "security") return { x: 618 + column * 22, y: 262 + row * 22 };
    return { x: 668 + column * 24, y: 510 + row * 17 };
  }

  function getPhase(localTime, passengerIndex) {
    let cursor = 0;
    for (const phase of phases) {
      const next = cursor + phase.duration;
      if (localTime <= next) {
        const progress = (localTime - cursor) / phase.duration;
        if (phase.queue && progress < 0.55) {
          return {
            phase,
            position: queuePosition(phase.queue, passengerIndex),
            progress,
            queued: true
          };
        }
        return {
          phase,
          position: interpolate(points[phase.from], points[phase.to], progress),
          progress,
          queued: false
        };
      }
      cursor = next;
    }
    return {
      phase: phases[phases.length - 1],
      position: points.aircraft,
      progress: 1,
      queued: false
    };
  }

  function createPassengers() {
    const passengers = [];
    for (let index = 0; index < config.passengerCount; index += 1) {
      const element = document.createElement("div");
      const id = `P${String(index + 1).padStart(3, "0")}`;
      element.className = "passenger";
      element.dataset.id = id;
      element.title = id;
      map.appendChild(element);
      passengers.push({
        id,
        index,
        delay: index * 0.55,
        waitOffset: 2 + (index % 7) * 0.42,
        element,
        lastPhase: ""
      });
    }
    return passengers;
  }

  function statusClass(status) {
    return `status-${status.toLowerCase().replaceAll(" ", "-")}`;
  }

  function renderFlightBoard() {
    flightBoardBody.innerHTML = flights.map((flight) => `
      <tr>
        <td>${flight.flight}</td>
        <td>${flight.destination}</td>
        <td>${flight.gate}</td>
        <td>${flight.scheduled}</td>
        <td>${flight.estimate}</td>
        <td><span class="flight-badge ${statusClass(flight.status)}">${flight.status}</span></td>
      </tr>
    `).join("");
    metrics.delayed.textContent = String(flights.filter((flight) => flight.status === "DELAYED").length);
  }

  function addEvent(message, elapsed) {
    const item = document.createElement("div");
    item.className = "event-item";
    item.innerHTML = `<span class="event-time">${Math.max(elapsed, 0).toFixed(1)}m</span>${message}`;
    eventLog.prepend(item);
    while (eventLog.children.length > 22) eventLog.removeChild(eventLog.lastChild);
  }

  function eventForPhase(passengerId, phaseName) {
    const counterNumber = (Number(passengerId.slice(1)) % 3) + 1;
    const laneNumber = (Number(passengerId.slice(1)) % 2) + 1;
    const gateName = Number(passengerId.slice(1)) % 2 === 0 ? "A2" : "A1";
    const messages = {
      entrance: `Passenger ${passengerId} entered airport`,
      checkin: `Passenger ${passengerId} reached check-in queue`,
      counter: `Counter ${counterNumber} serving ${passengerId}`,
      security: `Passenger ${passengerId} reached security queue`,
      lane: `Passenger ${passengerId} cleared security at Lane ${laneNumber}`,
      lounge: `Passenger ${passengerId} entered waiting lounge`,
      gate: `Passenger ${passengerId} waiting at Gate ${gateName}`,
      boarded: `Passenger ${passengerId} boarded flight`
    };
    return messages[phaseName] || `Passenger ${passengerId} moving`;
  }

  function createStaticQueueDots() {
    const queues = [
      ["checkin", 12],
      ["security", 10],
      ["gate", 8]
    ];
    for (const [queueName, count] of queues) {
      for (let index = 0; index < count; index += 1) {
        const dot = document.createElement("div");
        const point = queuePosition(queueName, index);
        dot.className = "queue-dot";
        dot.style.left = `${point.x + 6}px`;
        dot.style.top = `${point.y + 6}px`;
        map.appendChild(dot);
      }
    }
  }

  function updateMetrics(counts, averageWait) {
    metrics.total.textContent = String(config.passengerCount);
    metrics.checkin.textContent = String(counts.checkin);
    metrics.security.textContent = String(counts.security);
    metrics.lounge.textContent = String(counts.lounge);
    metrics.boarded.textContent = String(counts.boarded);
    metrics.wait.textContent = `${averageWait.toFixed(1)}m`;
  }

  function startAnimation() {
    const passengers = createPassengers();
    createStaticQueueDots();
    renderFlightBoard();
    flights.filter((flight) => flight.status === "DELAYED").forEach((flight) => {
      addEvent(`Flight ${flight.flight} delayed by ${flight.delay} minutes`, 0);
    });
    const startTime = performance.now();

    function tick(now) {
      const elapsed = ((now - startTime) / 1000) * config.speed;
      const counts = { checkin: 0, security: 0, lounge: 0, boarded: 0 };
      let totalWait = 0;

      passengers.forEach((passenger) => {
        const localTime = elapsed - passenger.delay;
        if (localTime < 0) {
          passenger.element.style.opacity = "0";
          return;
        }

        passenger.element.style.opacity = "1";
        const state = getPhase(localTime, passenger.index);
        const { phase, position } = state;
        passenger.element.style.transform = `translate(${position.x}px, ${position.y}px)`;
        passenger.element.classList.toggle("boarded", phase.name === "boarded");

        if (phase.name !== passenger.lastPhase) {
          passenger.lastPhase = phase.name;
          addEvent(eventForPhase(passenger.id, phase.name), elapsed);
        }

        if (["checkin", "counter"].includes(phase.name)) counts.checkin += 1;
        if (["security", "lane"].includes(phase.name)) counts.security += 1;
        if (["lounge", "gate"].includes(phase.name)) counts.lounge += 1;
        if (phase.name === "boarded") counts.boarded += 1;
        totalWait += Math.min(localTime, totalDuration) * 0.18 + passenger.waitOffset;
      });

      updateMetrics(counts, totalWait / Math.max(passengers.length, 1));
      requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }

  if (config.autoStart) startAnimation();
})();
</script>
</body>
</html>
"""


def render_advanced_animated_airport_page() -> None:
    """Render the advanced animated airport simulation page."""

    components.html(build_advanced_airport_html(), height=900, scrolling=True)
