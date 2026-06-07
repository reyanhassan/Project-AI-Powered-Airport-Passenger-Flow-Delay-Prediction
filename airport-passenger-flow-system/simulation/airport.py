"""Airport resources such as check-in counters, security, and boarding gates."""

import simpy

from .passenger import Passenger


class Airport:
    """Represent shared airport service points in a SimPy environment."""

    def __init__(
        self,
        env: simpy.Environment,
        check_in_counters: int = 3,
        security_lanes: int = 2,
        boarding_gates: int = 1,
    ) -> None:
        self.env = env

        # A Resource is like a limited number of staff desks or service lanes.
        self.check_in_counters = simpy.Resource(env, capacity=check_in_counters)
        self.security_lanes = simpy.Resource(env, capacity=security_lanes)
        self.boarding_gates = simpy.Resource(env, capacity=boarding_gates)

    def resource_summary(self) -> dict[str, int]:
        """Return capacities so the dashboard or reports can display them."""

        return {
            "check_in_counters": self.check_in_counters.capacity,
            "security_lanes": self.security_lanes.capacity,
            "boarding_gates": self.boarding_gates.capacity,
        }

    def check_in_passenger(self, passenger: Passenger, service_time: float):
        """Simulate one passenger waiting for and completing check-in."""

        passenger.add_event("joined_check_in_queue", self.env.now)
        queue_start = self.env.now

        with self.check_in_counters.request() as request:
            yield request
            passenger.check_in_wait = self.env.now - queue_start
            passenger.add_event("started_check_in", self.env.now)

            # The timeout represents the staff member serving this passenger.
            yield self.env.timeout(service_time)
            passenger.add_event("finished_check_in", self.env.now)

    def pass_security(self, passenger: Passenger, service_time: float):
        """Simulate one passenger waiting for and passing security."""

        passenger.add_event("joined_security_queue", self.env.now)
        queue_start = self.env.now

        with self.security_lanes.request() as request:
            yield request
            passenger.security_wait = self.env.now - queue_start
            passenger.add_event("started_security", self.env.now)

            yield self.env.timeout(service_time)
            passenger.add_event("finished_security", self.env.now)
