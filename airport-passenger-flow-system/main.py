"""Entry point for running the airport passenger flow project."""

import argparse
from pathlib import Path

from simulation import SimulationConfig, calculate_statistics, print_statistics, run_simulation
from simulation import save_simulation_outputs


def parse_arguments() -> argparse.Namespace:
    """Read simulation options from the command line."""

    parser = argparse.ArgumentParser(description="Run airport passenger simulation.")
    parser.add_argument("--passengers", type=int, default=50)
    parser.add_argument("--arrival-interval", type=float, default=2.0)
    parser.add_argument("--check-in-counters", type=int, default=3)
    parser.add_argument("--security-lanes", type=int, default=2)
    parser.add_argument("--boarding-gates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="")

    return parser.parse_args()


def main() -> None:
    """Run a sample airport passenger flow simulation."""

    args = parse_arguments()
    project_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "processed"

    config = SimulationConfig(
        num_passengers=args.passengers,
        average_arrival_interval=args.arrival_interval,
        check_in_counters=args.check_in_counters,
        security_lanes=args.security_lanes,
        boarding_gates=args.boarding_gates,
        random_seed=args.seed,
    )
    passengers = run_simulation(config=config)
    statistics = calculate_statistics(passengers)

    print_statistics(statistics)
    saved_files = save_simulation_outputs(passengers, output_dir)

    print("\nSaved simulation output files:")
    for file_label, file_path in saved_files.items():
        print(f"- {file_label}: {file_path}")


if __name__ == "__main__":
    main()
