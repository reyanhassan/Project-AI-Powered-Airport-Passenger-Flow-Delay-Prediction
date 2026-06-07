"""Basic smoke tests for the simulation module."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation import SimulationConfig, calculate_statistics, run_simulation


class SimulationSmokeTest(unittest.TestCase):
    """Check that the main simulation path works with a small scenario."""

    def test_run_simulation_returns_expected_passengers(self) -> None:
        config = SimulationConfig(
            num_passengers=10,
            average_arrival_interval=1.5,
            check_in_counters=2,
            security_lanes=2,
            boarding_gates=1,
            random_seed=7,
        )

        passengers = run_simulation(config=config)
        statistics = calculate_statistics(passengers)

        self.assertEqual(len(passengers), 10)
        self.assertEqual(statistics["total_passengers"], 10)
        self.assertIn(statistics["congestion_level"], {"Low", "Medium", "High"})

        for passenger in passengers:
            self.assertGreaterEqual(passenger.total_wait(), 0)
            self.assertGreaterEqual(passenger.completion_time, passenger.arrival_time)

    def test_invalid_configuration_raises_clear_error(self) -> None:
        config = SimulationConfig(num_passengers=-1)

        with self.assertRaises(ValueError):
            run_simulation(config=config)


if __name__ == "__main__":
    unittest.main()
