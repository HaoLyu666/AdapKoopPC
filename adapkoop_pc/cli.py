from __future__ import annotations

import argparse
from pathlib import Path

from .config import ModelConfig, PROJECT_ROOT, SimulationConfig, resolve_project_path
from .simulation import Simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AdapKoopPC mixed-platoon control simulation.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/true_new_64", help="Directory with epoch*_*.tar files.")
    parser.add_argument("--epoch", default="8", help="Checkpoint epoch number.")
    parser.add_argument("--output-dir", default="outputs/default", help="Directory for metrics and simulation result files.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--vehicles", type=int, default=50, help="Number of vehicles in the platoon.")
    parser.add_argument("--steps", type=int, default=1550, help="Simulation steps.")
    parser.add_argument("--horizon", type=int, default=10, help="MPC prediction horizon.")
    parser.add_argument("--cav-penetration", type=float, default=0.3, help="CAV penetration rate.")
    parser.add_argument("--truck-penetration", type=float, default=0.2, help="Truck penetration rate.")
    parser.add_argument("--noise-std", type=float, default=0.4, help="Observation noise standard deviation.")
    parser.add_argument("--seed", type=int, default=72, help="NumPy/Torch random seed.")
    parser.add_argument("--arrangement-seed", type=int, default=30, help="Python random seed for vehicle arrangement.")
    parser.add_argument("--optimizer-maxiter", type=int, default=None, help="Optional SLSQP maximum iterations.")
    parser.add_argument("--progress", action="store_true", help="Print simulation progress.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    output_dir = resolve_project_path(args.output_dir)
    config = SimulationConfig(
        checkpoint_dir=checkpoint_dir,
        checkpoint_epoch=args.epoch,
        output_dir=output_dir,
        device=args.device,
        seed=args.seed,
        arrangement_seed=args.arrangement_seed,
        horizon=args.horizon,
        vehicle_count=args.vehicles,
        simulation_steps=args.steps,
        cav_penetration=args.cav_penetration,
        truck_penetration=args.truck_penetration,
        observation_noise_std=args.noise_std,
        optimizer_max_iterations=args.optimizer_maxiter,
        progress=args.progress,
    )
    model = ModelConfig(device=args.device)
    result = Simulator(config, model).run()

    try:
        display_output = result.output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        display_output = result.output_dir
    print(f"Results written to {display_output}")
    for key, value in result.metrics.items():
        print(f"{key}: {value:.8g}")
