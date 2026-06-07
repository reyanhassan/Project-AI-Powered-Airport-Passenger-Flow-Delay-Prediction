"""Main SimPy simulation logic for airport passenger flow."""

import random
from typing import List

import simpy

from .airport import Airport
from .passenger import Passenger


def _service_time(rng: random.Random, minimum: float, maximum: float) -> float:
    """Return a random service time between two values."""

    return rng.uniform(minimum, maximum)


def passenger_journey(
    env: simpy.Environment,
    airport: Airport,
    passenger: Passenger,
    rng: random.Random,
):
    """Move one passenger through all airport stages."""

    passenger.add_event("arrived_at_airport", env.now)

    yield env.process(
        airport.check_in_passenger(
            passenger,
            service_time=_service_time(rng, 2.0, 6.0),
        )
    )
    yield env.process(
        airport.pass_security(
            passenger,
            service_time=_service_time(rng, 1.5, 5.0),
        )
    )
    yield env.process(
        airport.board_flight(
            passenger,
            service_time=_service_time(rng, 1.0, 3.0),
        )
    )


def _passenger_arrivals(
    env: simpy.Environment,
    airport: Airport,
    passengers: List[Passenger],
    num_passengers: int,
    average_arrival_interval: float,
    rng: random.Random,
):
    """Create passengers over time instead of placing everyone at once."""

    for passenger_id in range(1, num_passengers + 1):
        passenger = Passenger(
            passenger_id=passenger_id,
            arrival_time=env.now,
            flight_id=f"FL-{rng.randint(100, 999)}",
        )
        passengers.append(passenger)
        env.process(passenger_journey(env, airport, passenger, rng))

        # Exponential arrivals are common in simple queueing simulations.
        time_until_next_arrival = rng.expovariate(1 / average_arrival_interval)
        yield env.timeout(time_until_next_arrival)


def run_simulation(
    num_passengers: int = 50,
    average_arrival_interval: float = 2.0,
    check_in_counters: int = 3,
    security_lanes: int = 2,
    boarding_gates: int = 1,
    random_seed: int = 42,
) -> List[Passenger]:
    """Run the complete passenger flow simulation and return passengers."""

    rng = random.Random(random_seed)
    env = simpy.Environment()
    airport = Airport(
        env,
        check_in_counters=check_in_counters,
        security_lanes=security_lanes,
        boarding_gates=boarding_gates,
    )
    passengers: List[Passenger] = []

    env.process(
        _passenger_arrivals(
            env,
            airport,
            passengers,
            num_passengers,
            average_arrival_interval,
            rng,
        )
    )
    env.run()

    return passengers
