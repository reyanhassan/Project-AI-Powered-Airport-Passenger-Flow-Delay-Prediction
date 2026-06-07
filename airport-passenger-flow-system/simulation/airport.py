"""Airport resources such as check-in counters, security, and boarding gates."""

import simpy


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
