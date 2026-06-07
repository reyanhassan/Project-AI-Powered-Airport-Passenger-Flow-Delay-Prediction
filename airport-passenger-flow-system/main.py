"""Entry point for running the airport passenger flow project."""

from pathlib import Path

from simulation import calculate_statistics, print_statistics, run_simulation
from simulation import save_simulation_outputs


def main() -> None:
    """Run a sample airport passenger flow simulation."""

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "data" / "processed"

    passengers = run_simulation(
        num_passengers=50,
        average_arrival_interval=2.0,
        check_in_counters=3,
        security_lanes=2,
        boarding_gates=1,
        random_seed=42,
    )
    statistics = calculate_statistics(passengers)

    print_statistics(statistics)
    saved_files = save_simulation_outputs(passengers, output_dir)

    print("\nSaved simulation output files:")
    for file_label, file_path in saved_files.items():
        print(f"- {file_label}: {file_path}")


if __name__ == "__main__":
    main()
