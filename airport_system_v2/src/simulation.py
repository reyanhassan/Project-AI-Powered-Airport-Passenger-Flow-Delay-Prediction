"""SimPy passenger-flow simulation for a three-stage airport.

Passengers arrive over time and flow through three resource-constrained stages:
check-in → security → boarding. Each stage is a :class:`simpy.Resource` with a
configurable number of servers. The module records a full event timeline per
passenger, from which it derives:

* aggregate statistics (waits per stage, busiest stage, congestion level),
* a queue-length timeline for the analytics charts, and
* frame-by-frame passenger positions for the live animated map.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

# Stage geometry for the live animation map (x-centre of each zone).
ZONE_X = {"check_in": 2.5, "security": 5.5, "boarding": 8.5, "done": 10.8}
_WAIT_OFFSET = 0.7  # waiting passengers sit just before a zone's service point.


@dataclass
class SimulationConfig:
    """Tunable parameters for a single simulation run."""

    num_passengers: int = 80
    arrival_interval: float = 1.5
    check_in_counters: int = 3
    security_lanes: int = 3
    boarding_gates: int = 2
    random_seed: int = 42
    check_in_time: tuple[float, float] = (2.0, 6.0)
    security_time: tuple[float, float] = (1.5, 5.0)
    boarding_time: tuple[float, float] = (1.0, 3.0)


@dataclass
class Passenger:
    """A single simulated passenger and the timestamps of their journey."""

    pid: int
    lane: int
    arrival_time: float = 0.0
    # Per-stage timestamps: queue-join, service-start, service-end.
    times: dict[str, float] = field(default_factory=dict)
    check_in_wait: float = 0.0
    security_wait: float = 0.0
    boarding_wait: float = 0.0
    completion_time: float = 0.0

    @property
    def total_wait(self) -> float:
        """Total time spent queueing across all three stages."""
        return self.check_in_wait + self.security_wait + self.boarding_wait

    @property
    def journey_time(self) -> float:
        """End-to-end time from arrival to completed boarding."""
        return self.completion_time - self.arrival_time if self.completion_time else 0.0

    def record(self) -> dict[str, float]:
        """Flatten the passenger into a summary row for a DataFrame."""
        return {
            "passenger_id": self.pid,
            "arrival_time": round(self.arrival_time, 2),
            "check_in_wait": round(self.check_in_wait, 2),
            "security_wait": round(self.security_wait, 2),
            "boarding_wait": round(self.boarding_wait, 2),
            "total_wait": round(self.total_wait, 2),
            "journey_time": round(self.journey_time, 2),
            "completion_time": round(self.completion_time, 2),
        }


def _serve(env, resource, passenger, stage, wait_attr, service_range, rng):
    """Generic SimPy process: queue for ``resource`` then receive service.

    Records queue-join, service-start and service-end timestamps on the passenger
    and stores the measured queueing time on ``wait_attr``.
    """
    passenger.times[f"{stage}_join"] = env.now
    join = env.now
    with resource.request() as req:
        yield req
        setattr(passenger, wait_attr, env.now - join)
        passenger.times[f"{stage}_start"] = env.now
        yield env.timeout(rng.uniform(*service_range))
        passenger.times[f"{stage}_end"] = env.now


def _journey(env, resources, passenger, config, rng):
    """Run one passenger through check-in, security and boarding in sequence."""
    yield env.process(_serve(env, resources["check_in"], passenger, "check_in",
                             "check_in_wait", config.check_in_time, rng))
    yield env.process(_serve(env, resources["security"], passenger, "security",
                             "security_wait", config.security_time, rng))
    yield env.process(_serve(env, resources["boarding"], passenger, "boarding",
                             "boarding_wait", config.boarding_time, rng))
    passenger.completion_time = env.now


def _arrivals(env, resources, passengers, config, rng):
    """Generate passengers with exponential inter-arrival times."""
    for pid in range(1, config.num_passengers + 1):
        passenger = Passenger(pid=pid, lane=pid % 14, arrival_time=env.now)
        passengers.append(passenger)
        env.process(_journey(env, resources, passenger, config, rng))
        yield env.timeout(rng.expovariate(1 / config.arrival_interval))


def run_simulation(config: SimulationConfig) -> list[Passenger]:
    """Execute the full simulation and return the completed passenger list.

    Args:
        config: The simulation configuration.

    Returns:
        A list of :class:`Passenger` objects with all timestamps populated.
    """
    rng = random.Random(config.random_seed)
    env = simpy.Environment()
    resources = {
        "check_in": simpy.Resource(env, capacity=config.check_in_counters),
        "security": simpy.Resource(env, capacity=config.security_lanes),
        "boarding": simpy.Resource(env, capacity=config.boarding_gates),
    }
    passengers: list[Passenger] = []
    env.process(_arrivals(env, resources, passengers, config, rng))
    env.run()
    return passengers


def passengers_dataframe(passengers: list[Passenger]) -> pd.DataFrame:
    """Convert passengers into a tidy summary DataFrame (one row each)."""
    return pd.DataFrame([p.record() for p in passengers])


def compute_statistics(passengers: list[Passenger]) -> dict[str, object]:
    """Summarise a finished simulation into headline metrics.

    Args:
        passengers: Completed passenger list.

    Returns:
        Dictionary of totals, per-stage average waits, the busiest stage and a
        coarse congestion label.
    """
    if not passengers:
        return {"total_passengers": 0, "avg_check_in_wait": 0.0, "avg_security_wait": 0.0,
                "avg_boarding_wait": 0.0, "avg_total_wait": 0.0, "avg_journey_time": 0.0,
                "max_journey_time": 0.0, "busiest_stage": "None", "congestion_level": "Low"}

    frame = passengers_dataframe(passengers)
    stage_waits = {
        "Check-in": frame["check_in_wait"].mean(),
        "Security": frame["security_wait"].mean(),
        "Boarding": frame["boarding_wait"].mean(),
    }
    avg_total = float(frame["total_wait"].mean())
    congestion = "Low" if avg_total < 5 else "Medium" if avg_total < 12 else "High"

    return {
        "total_passengers": int(len(frame)),
        "avg_check_in_wait": round(stage_waits["Check-in"], 2),
        "avg_security_wait": round(stage_waits["Security"], 2),
        "avg_boarding_wait": round(stage_waits["Boarding"], 2),
        "avg_total_wait": round(avg_total, 2),
        "avg_journey_time": round(float(frame["journey_time"].mean()), 2),
        "max_journey_time": round(float(frame["journey_time"].max()), 2),
        "busiest_stage": max(stage_waits, key=stage_waits.get),
        "congestion_level": congestion,
    }


def stage_wait_dataframe(passengers: list[Passenger]) -> pd.DataFrame:
    """Return a tidy ``stage``/``avg_wait`` DataFrame for the per-stage bar chart."""
    stats = compute_statistics(passengers)
    return pd.DataFrame([
        {"stage": "Check-in", "avg_wait": stats["avg_check_in_wait"]},
        {"stage": "Security", "avg_wait": stats["avg_security_wait"]},
        {"stage": "Boarding", "avg_wait": stats["avg_boarding_wait"]},
    ])


def _time_grid(passengers: list[Passenger], frames: int) -> np.ndarray:
    """Build an evenly spaced array of sample times spanning the simulation."""
    end = max((p.completion_time for p in passengers), default=1.0) or 1.0
    return np.linspace(0, end, frames)


def queue_timeline(passengers: list[Passenger], frames: int = 40) -> pd.DataFrame:
    """Sample queue lengths at each stage across evenly spaced time points.

    A passenger counts toward a stage's queue when they have joined that stage's
    queue but service has not yet started.

    Args:
        passengers: Completed passenger list.
        frames: Number of time samples.

    Returns:
        Long-form DataFrame with columns ``time``, ``stage`` and ``queue_length``
        (suited to an animated/standard Plotly line chart).
    """
    rows = []
    for t in _time_grid(passengers, frames):
        for stage, label in (("check_in", "Check-in"), ("security", "Security"), ("boarding", "Boarding")):
            count = sum(
                1 for p in passengers
                if p.times.get(f"{stage}_join", np.inf) <= t < p.times.get(f"{stage}_start", -np.inf)
            )
            rows.append({"time": round(float(t), 2), "stage": label, "queue_length": count})
    return pd.DataFrame(rows)


def _passenger_state(p: Passenger, t: float) -> tuple[str, str] | None:
    """Return ``(zone, status)`` for passenger ``p`` at time ``t``, or ``None``.

    ``None`` means the passenger has not yet arrived. Status is one of
    ``waiting``, ``serving`` or ``done``.
    """
    if t < p.arrival_time:
        return None
    for stage, zone in (("check_in", "check_in"), ("security", "security"), ("boarding", "boarding")):
        join = p.times.get(f"{stage}_join", np.inf)
        start = p.times.get(f"{stage}_start", np.inf)
        end = p.times.get(f"{stage}_end", np.inf)
        if join <= t < start:
            return zone, "waiting"
        if start <= t < end:
            return zone, "serving"
    if p.completion_time and t >= p.completion_time:
        return "done", "done"
    # Between finishing one stage and joining the next — treat as waiting at next zone.
    return "boarding", "waiting"


def animation_frames(passengers: list[Passenger], frames: int = 40) -> pd.DataFrame:
    """Build per-frame passenger positions for the live animated map.

    Each present passenger is placed at the x-centre of their current zone (with a
    backward offset while queueing) and a fixed vertical lane, coloured by status.

    Args:
        passengers: Completed passenger list.
        frames: Number of animation frames to render.

    Returns:
        DataFrame with columns ``frame``, ``time``, ``passenger_id``, ``x``, ``y``,
        ``status`` and ``zone``.
    """
    rows = []
    for frame_idx, t in enumerate(_time_grid(passengers, frames)):
        for p in passengers:
            state = _passenger_state(p, t)
            if state is None:
                continue
            zone, status = state
            x = ZONE_X[zone] - (_WAIT_OFFSET if status == "waiting" else 0.0)
            jitter = ((p.pid * 37) % 100) / 100.0 - 0.5  # deterministic horizontal scatter
            rows.append({
                "frame": frame_idx,
                "time": round(float(t), 1),
                "passenger_id": p.pid,
                "x": round(x + jitter * 0.5, 3),
                "y": 1.0 + p.lane * 0.62,
                "status": status,
                "zone": zone,
            })
    return pd.DataFrame(rows)
