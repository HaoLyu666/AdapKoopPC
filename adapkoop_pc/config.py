from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    """Neural model settings that must match the checkpoint architecture."""

    hidden_size: int = 64
    n_head: int = 4
    attention_output_size: int = 32
    input_length: int = 31
    output_length: int = 15
    feature_length: int = 5
    trajectory_hidden_size: int | None = None
    use_elu: bool = True
    dropout: float = 0.2
    leaky_relu_slope: float = 0.1
    linear_decoder: bool = False
    train_mode: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.trajectory_hidden_size is None:
            self.trajectory_hidden_size = self.hidden_size

    def to_legacy_args(self) -> dict[str, Any]:
        """Return the key names used by the original model implementation."""

        return {
            "device": torch.device(self.device),
            "lstm_encoder_size": self.hidden_size,
            "n_head": self.n_head,
            "att_out": self.attention_output_size,
            "in_length": self.input_length,
            "out_length": self.output_length,
            "f_length": self.feature_length,
            "traj_linear_hidden": self.trajectory_hidden_size,
            "use_elu": self.use_elu,
            "dropout": self.dropout,
            "relu": self.leaky_relu_slope,
            "liner_dec": int(self.linear_decoder),
            "train_flag": self.train_mode,
        }


@dataclass
class SimulationConfig:
    """Simulation and MPC settings."""

    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "true_new_64"
    checkpoint_epoch: str = "8"
    output_dir: Path = PROJECT_ROOT / "outputs" / "default"
    device: str = "cpu"
    seed: int = 72
    arrangement_seed: int = 30

    horizon: int = 10
    vehicle_count: int = 50
    time_step: float = 0.12
    simulation_steps: int = 1550

    velocity_max: float = 33.0
    velocity_min: float = 0.0
    acceleration_max: float = 6.0
    acceleration_min: float = -6.0
    headway_max: float = 150.0
    headway_min: float = 0.0

    position_weight: float = 10.0
    velocity_weight: float = 10.0
    control_weight: float = 1.0
    velocity_difference_weight: float = 100.0

    idm_acceleration: list[float] = field(
        default_factory=lambda: [1.1258, 1.500000017959308]
    )
    idm_desired_velocity: list[float] = field(
        default_factory=lambda: [35.9551, 54.246295638178570]
    )
    idm_comfortable_deceleration: list[float] = field(default_factory=lambda: [4.0, 6.0])
    idm_min_spacing: list[float] = field(default_factory=lambda: [5.1645, 9.657302537298452])
    idm_time_headway: list[float] = field(default_factory=lambda: [1.0318, 1.714745062310631])
    idm_delta: float = 4.0
    vehicle_length: list[float] = field(default_factory=lambda: [4.24, 11.82])

    cav_headway_time: float = 1.0
    cav_penetration: float = 0.3
    truck_penetration: float = 0.2

    lead_base_velocity: float = 25.0
    lead_sine_amplitude: float = 5.0
    lead_sine_frequency: float = 0.02
    lead_sine_start_step: int = 40
    observation_noise_std: float = 0.4
    optimizer_max_iterations: int | None = None

    progress: bool = False

    def apply_reproducibility(self) -> None:
        import random

        random.seed(self.arrangement_seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def to_legacy_args(self, model_config: ModelConfig) -> dict[str, Any]:
        args = model_config.to_legacy_args()
        args.update(
            {
                "path": str(self.checkpoint_dir),
                "np": self.horizon,
                "veh_num": self.vehicle_count,
                "time_step": self.time_step,
                "sim_step": self.simulation_steps,
                "v_max": self.velocity_max,
                "v_min": self.velocity_min,
                "a_max": self.acceleration_max,
                "a_min": self.acceleration_min,
                "h_max": self.headway_max,
                "h_min": self.headway_min,
                "x_weight": self.position_weight,
                "v_weight": self.velocity_weight,
                "u_weight": self.control_weight,
                "dv_weight": self.velocity_difference_weight,
                "a_idm": self.idm_acceleration,
                "ve": self.idm_desired_velocity,
                "b_idm": self.idm_comfortable_deceleration,
                "s0": self.idm_min_spacing,
                "T0": self.idm_time_headway,
                "delta": [self.idm_delta],
                "veh_length": self.vehicle_length,
                "cav_th": self.cav_headway_time,
                "cav_permeability": self.cav_penetration,
                "truck_permeability": self.truck_penetration,
            }
        )
        return args


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
