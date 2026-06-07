"""Main SimPy simulation logic for airport passenger flow."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import simpy

from .airport import Airport
from .passenger import Passenger


@dataclass
class SimulationConfig:
    """Store all simulation settings in one easy-to-change object."""

    num_passengers: int = 50
    average_arrival_interval: float = 2.0
    check_in_counters: int = 3
    security_lanes: int = 2
    boarding_gates: int = 1
    random_seed: int = 42
    check_in_service_time: tuple[float, float] = (2.0, 6.0)
    security_service_time: tuple[float, float] = (1.5, 5.0)
    boarding_service_time: tuple[float, float] = (1.0, 3.0)

    def validate(self) -> None:
        """Raise a clear error when simulation settings are not realistic."""

        if self.num_passengers < 0:
            raise ValueError("num_passengers must be 0 or greater.")
        if self.average_arrival_interval <= 0:
            raise ValueError("average_arrival_interval must be greater than 0.")
        if self.check_in_counters <= 0:
            raise ValueError("check_in_counters must be greater than 0.")
        if self.security_lanes <= 0:
            raise ValueError("security_lanes must be greater than 0.")
        if self.boarding_gates <= 0:
            raise ValueError("boarding_gates must be greater than 0.")

        self._validate_time_range("check_in_service_time", self.check_in_service_time)
        self._validate_time_range("security_service_time", self.security_service_time)
        self._validate_time_range("boarding_service_time", self.boarding_service_time)

    @staticmethod
    def _validate_time_range(name: str, time_range: tuple[float, float]) -> None:
        """Check that a service-time range has a valid lower and upper limit."""

        minimum, maximum = time_range
        if minimum <= 0 or maximum <= 0:
            raise ValueError(f"{name} values must be greater than 0.")
        if minimum > maximum:
            raise ValueError(f"{name} minimum cannot be greater than maximum.")


def _service_time(rng: random.Random, minimum: float, maximum: float) -> float:
    """Return a random service time between two values."""

    return rng.uniform(minimum, maximum)


def passenger_journey(
    env: simpy.Environment,
    airport: Airport,
    passenger: Passenger,
    rng: random.Random,
    config: SimulationConfig,
):
    """Move one passenger through all airport stages."""

    passenger.add_event("arrived_at_airport", env.now)

    yield env.process(
        airport.check_in_passenger(
            passenger,
            service_time=_service_time(rng, *config.check_in_service_time),
        )
    )
    yield env.process(
        airport.pass_security(
            passenger,
            service_time=_service_time(rng, *config.security_service_time),
        )
    )
    yield env.process(
        airport.board_flight(
            passenger,
            service_time=_service_time(rng, *config.boarding_service_time),
        )
    )


def _passenger_arrivals(
    env: simpy.Environment,
    airport: Airport,
    passengers: List[Passenger],
    rng: random.Random,
    config: SimulationConfig,
):
    """Create passengers over time instead of placing everyone at once."""

    for passenger_id in range(1, config.num_passengers + 1):
        passenger = Passenger(
            passenger_id=passenger_id,
            arrival_time=env.now,
            flight_id=f"FL-{rng.randint(100, 999)}",
        )
        passengers.append(passenger)
        env.process(passenger_journey(env, airport, passenger, rng, config))

        # Exponential arrivals are common in simple queueing simulations.
        time_until_next_arrival = rng.expovariate(1 / config.average_arrival_interval)
        yield env.timeout(time_until_next_arrival)


def run_simulation(
    num_passengers: int = 50,
    average_arrival_interval: float = 2.0,
    check_in_counters: int = 3,
    security_lanes: int = 2,
    boarding_gates: int = 1,
    random_seed: int = 42,
    config: SimulationConfig | None = None,
) -> List[Passenger]:
    """Run the complete passenger flow simulation and return passengers."""

    if config is None:
        config = SimulationConfig(
            num_passengers=num_passengers,
            average_arrival_interval=average_arrival_interval,
            check_in_counters=check_in_counters,
            security_lanes=security_lanes,
            boarding_gates=boarding_gates,
            random_seed=random_seed,
        )

    config.validate()

    rng = random.Random(config.random_seed)
    env = simpy.Environment()
    airport = Airport(
        env,
        check_in_counters=config.check_in_counters,
        security_lanes=config.security_lanes,
        boarding_gates=config.boarding_gates,
    )
    passengers: List[Passenger] = []

    env.process(
        _passenger_arrivals(
            env,
            airport,
            passengers,
            rng,
            config,
        )
    )
    env.run()

    return passengers


def passengers_to_dataframe(passengers: List[Passenger]) -> pd.DataFrame:
    """Convert passenger objects into a Pandas DataFrame for analysis."""

    return pd.DataFrame([passenger.to_record() for passenger in passengers])


def event_log_to_dataframe(passengers: List[Passenger]) -> pd.DataFrame:
    """Create a detailed event log from every passenger journey."""

    event_rows = []
    for passenger in passengers:
        event_rows.extend(passenger.event_log)

    return pd.DataFrame(event_rows)


def _congestion_level(average_total_wait: float) -> str:
    """Convert average waiting time into a simple congestion label."""

    if average_total_wait < 5:
        return "Low"
    if average_total_wait < 10:
        return "Medium"

    return "High"


def calculate_statistics(passengers: List[Passenger]) -> Dict[str, object]:
    """Calculate beginner-friendly summary statistics from the simulation."""

    if not passengers:
        return {
            "total_passengers": 0,
            "average_check_in_wait": 0.0,
            "average_security_wait": 0.0,
            "average_boarding_wait": 0.0,
            "average_total_wait": 0.0,
            "maximum_total_wait": 0.0,
            "average_journey_time": 0.0,
            "maximum_journey_time": 0.0,
            "busiest_stage": "None",
            "congestion_level": "Low",
        }

    passenger_data = passengers_to_dataframe(passengers)
    stage_waits = {
        "Check-in": passenger_data["check_in_wait"].mean(),
        "Security": passenger_data["security_wait"].mean(),
        "Boarding": passenger_data["boarding_wait"].mean(),
    }
    average_total_wait = round(passenger_data["total_wait"].mean(), 2)

    return {
        "total_passengers": len(passenger_data),
        "average_check_in_wait": round(passenger_data["check_in_wait"].mean(), 2),
        "average_security_wait": round(passenger_data["security_wait"].mean(), 2),
        "average_boarding_wait": round(passenger_data["boarding_wait"].mean(), 2),
        "average_total_wait": average_total_wait,
        "maximum_total_wait": round(passenger_data["total_wait"].max(), 2),
        "average_journey_time": round(passenger_data["journey_time"].mean(), 2),
        "maximum_journey_time": round(passenger_data["journey_time"].max(), 2),
        "busiest_stage": max(stage_waits, key=stage_waits.get),
        "congestion_level": _congestion_level(average_total_wait),
    }


def print_statistics(statistics: Dict[str, object]) -> None:
    """Display simulation statistics in the terminal."""

    print("\nAirport Passenger Flow Simulation Statistics")
    print("-" * 52)
    print(f"Total passengers: {int(statistics['total_passengers'])}")
    print(f"Average check-in wait: {statistics['average_check_in_wait']} minutes")
    print(f"Average security wait: {statistics['average_security_wait']} minutes")
    print(f"Average boarding wait: {statistics['average_boarding_wait']} minutes")
    print(f"Average total wait: {statistics['average_total_wait']} minutes")
    print(f"Maximum total wait: {statistics['maximum_total_wait']} minutes")
    print(f"Average journey time: {statistics['average_journey_time']} minutes")
    print(f"Maximum journey time: {statistics['maximum_journey_time']} minutes")
    print(f"Busiest stage: {statistics['busiest_stage']}")
    print(f"Congestion level: {statistics['congestion_level']}")


def save_simulation_outputs(
    passengers: List[Passenger],
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Save passenger summaries and event logs as CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    passenger_summary_path = output_path / "simulation_passenger_summary.csv"
    event_log_path = output_path / "simulation_event_log.csv"

    passengers_to_dataframe(passengers).to_csv(passenger_summary_path, index=False)
    event_log_to_dataframe(passengers).to_csv(event_log_path, index=False)

    return {
        "passenger_summary": passenger_summary_path,
        "event_log": event_log_path,
    }
