"""Passenger entity for the airport passenger flow simulation.

The Passenger class is intentionally small and beginner friendly. It stores
the basic identity of one passenger and the waiting times collected while the
simulation runs.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Passenger:
    """Store information about one simulated airport passenger."""

    passenger_id: int
    arrival_time: float
    flight_id: str
    check_in_wait: float = 0.0
    security_wait: float = 0.0
    boarding_wait: float = 0.0
    event_log: List[Dict[str, object]] = field(default_factory=list)

    def add_event(self, event_name: str, current_time: float) -> None:
        """Add a timestamped event to this passenger's journey log."""

        self.event_log.append(
            {
                "time": round(current_time, 2),
                "passenger_id": self.passenger_id,
                "flight_id": self.flight_id,
                "event": event_name,
            }
        )

    def total_wait(self) -> float:
        """Return the total time spent waiting in queues."""

        return self.check_in_wait + self.security_wait + self.boarding_wait

    def to_record(self) -> Dict[str, object]:
        """Convert the passenger data into a row that Pandas can save."""

        return {
            "passenger_id": self.passenger_id,
            "flight_id": self.flight_id,
            "arrival_time": round(self.arrival_time, 2),
            "check_in_wait": round(self.check_in_wait, 2),
            "security_wait": round(self.security_wait, 2),
            "boarding_wait": round(self.boarding_wait, 2),
            "total_wait": round(self.total_wait(), 2),
        }
