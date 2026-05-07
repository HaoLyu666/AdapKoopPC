from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from .checkpoints import checkpoint_set, infer_hidden_size, load_state
from .config import ModelConfig, SimulationConfig
from .kmpc import KoopmanMatrixBuilder, KoopmanOptimizer
from .models import DSEncoder, Encoder, StateEncoder
from .vehicle import VehicleStateModel


@dataclass
class SimulationResult:
    platoon: np.ndarray
    states: np.ndarray
    lead_reference: np.ndarray
    lead_state: np.ndarray
    control_inputs: np.ndarray
    computation_time: np.ndarray
    metrics: dict[str, float]
    output_dir: Path


class Simulator:
    """Run the AdapKoopPC mixed-platoon control simulation."""

    def __init__(self, sim_config: SimulationConfig, model_config: ModelConfig | None = None):
        self.sim_config = sim_config
        if model_config is None:
            hidden_size = infer_hidden_size(sim_config.checkpoint_dir, sim_config.checkpoint_epoch)
            model_config = ModelConfig(hidden_size=hidden_size, device=sim_config.device)
        self.model_config = model_config
        self.model_config.device = sim_config.device
        self.model_config.train_mode = False
        self.args = sim_config.to_legacy_args(self.model_config)

        self.device = torch.device(sim_config.device)
        self.dt = sim_config.time_step
        self.vehicle_count = sim_config.vehicle_count
        self.history_length = self.model_config.input_length
        self.horizon = sim_config.horizon
        self.steps = sim_config.simulation_steps
        self.vehicle_lengths: np.ndarray | None = None

    def run(self) -> SimulationResult:
        self.sim_config.apply_reproducibility()
        self.sim_config.output_dir.mkdir(parents=True, exist_ok=True)

        lead_velocity = self._lead_velocity_profile()
        matrix_builder = KoopmanMatrixBuilder(self.sim_config, self.model_config)
        (
            A,
            B,
            H,
            F_S,
            F_R,
            platoon,
            cav_count,
            _hdv_count,
            _truck_count,
        ) = matrix_builder.build()

        hdv_index = np.where(platoon != 0)
        cav_index = np.where(platoon == 0)
        optimizer = KoopmanOptimizer(self.sim_config, cav_index)
        vehicle_model = VehicleStateModel(self.sim_config, platoon)
        self.vehicle_lengths = vehicle_model.hdv_vehicle_lengths()

        ds_encoder, state_encoder, encoder = self._load_encoders()
        states, lead_state = self._initial_state(lead_velocity, platoon, vehicle_model)
        observed_states = states.copy()

        warm_start_control = np.random.uniform(-2, 2, self.horizon * cav_count)
        control_inputs = np.zeros([lead_velocity.shape[0], cav_count])
        computation_time = np.zeros_like(lead_velocity)

        noise_std = self.sim_config.observation_noise_std
        optimizer_options = {}
        if self.sim_config.optimizer_max_iterations is not None:
            optimizer_options["maxiter"] = self.sim_config.optimizer_max_iterations

        with torch.no_grad():
            for i in range(self.steps - self.horizon):
                if self.sim_config.progress and (i < 5 or i % 50 == 0):
                    print(f"step {i}/{self.steps - self.horizon - 1}")

                hdv_acceleration = vehicle_model.idm_acceleration(
                    states[hdv_index[0], i, 1],
                    states[hdv_index[0], i, 2],
                    states[hdv_index[0], i, 3],
                )
                states[hdv_index[0], i, 4] = hdv_acceleration

                if i >= self.history_length - 1:
                    observed_states[hdv_index[0], i, 4] = hdv_acceleration
                    history = torch.from_numpy(
                        observed_states[hdv_index[0], i - self.history_length + 1 : i + 1, :]
                    ).to(dtype=torch.float32, device=self.device)

                    reference = self._mpc_reference(i, lead_velocity, lead_state, observed_states, cav_index)
                    ds = ds_encoder(history)
                    state_embedding = state_encoder(history[:, :, 1:-1])
                    koopman_state = encoder(state_embedding, ds)
                    hdv_koopman_state = torch.squeeze(koopman_state).detach().cpu().numpy()

                    mixed_state = {k: v for k, v in zip(hdv_index[0], hdv_koopman_state)}
                    cav_states = observed_states[cav_index[0], i : i + 1, 2:].reshape(-1, 3)
                    mixed_state.update({k: v for k, v in zip(cav_index[0], cav_states)})

                    lifted_initial_state = np.concatenate([mixed_state[j] for j in range(self.vehicle_count)])
                    lifted_initial_state[0] = lead_state[i, 0]

                    objective = optimizer.objective(lifted_initial_state, reference, H, F_S, F_R)
                    bounds = optimizer.bounds(cav_count)

                    start_time = time.time()
                    result = minimize(
                        objective,
                        warm_start_control,
                        method="SLSQP",
                        bounds=bounds,
                        options=optimizer_options,
                    )
                    computation_time[i] = time.time() - start_time

                    optimal_control = result.x
                    control_inputs[i] = optimal_control[:cav_count]
                    warm_start_control = optimal_control

                    states[cav_index[0], i + 1, 4] = (
                        states[cav_index[0], i, 4] + self.dt * control_inputs[i]
                    )
                    observed_states[cav_index[0], i + 1, 4] = (
                        states[cav_index[0], i, 4] + self.dt * control_inputs[i]
                    )
                    lead_state[i + 1, 1], lead_state[i + 1, 0] = vehicle_model.next_lead_state(
                        states[0, i, 4],
                        control_inputs[i, 0],
                        lead_state[i, 1],
                        lead_state[i, 0],
                    )

                next_velocity, next_delta_velocity, next_headway = vehicle_model.next_vehicle_state(
                    states[:, i, 3],
                    states[:, i, 2],
                    lead_velocity[i + 1],
                    states[:, i, -1],
                )
                states[:, i + 1, 3] = next_velocity
                states[:, i + 1, 1] = next_delta_velocity
                states[:, i + 1, 2] = next_headway

                observed_velocity = np.random.normal(0, noise_std, next_velocity.shape) + next_velocity
                observed_headway = np.random.normal(0, noise_std, next_velocity.shape) + next_headway
                observed_states[:, i + 1, 3] = observed_velocity
                observed_states[:, i + 1, 1] = next_delta_velocity
                observed_states[:, i + 1, 2] = observed_headway

                if np.any(next_velocity < 0):
                    raise RuntimeError(f"Negative vehicle velocity at simulation step {i + 1}")

        states = self._append_positions(states, lead_state)
        metrics = self._metrics(states, computation_time)
        result = SimulationResult(
            platoon=platoon,
            states=states,
            lead_reference=lead_velocity,
            lead_state=lead_state,
            control_inputs=control_inputs,
            computation_time=computation_time,
            metrics=metrics,
            output_dir=self.sim_config.output_dir,
        )
        self.save_result(result)
        return result

    def _load_encoders(self) -> tuple[DSEncoder, StateEncoder, Encoder]:
        paths = checkpoint_set(self.sim_config.checkpoint_dir, self.sim_config.checkpoint_epoch)
        ds_encoder = DSEncoder(self.args)
        state_encoder = StateEncoder(self.args)
        encoder = Encoder(self.args)
        load_state(ds_encoder, paths.ds_encoder, self.device)
        load_state(state_encoder, paths.state_encoder, self.device)
        load_state(encoder, paths.encoder, self.device)
        ds_encoder.to(self.device).eval()
        state_encoder.to(self.device).eval()
        encoder.to(self.device).eval()
        return ds_encoder, state_encoder, encoder

    def _lead_velocity_profile(self) -> np.ndarray:
        index = np.arange(self.steps)
        velocity = np.ones(self.steps) * self.sim_config.lead_base_velocity
        start = self.sim_config.lead_sine_start_step
        velocity[start:] = self.sim_config.lead_base_velocity - self.sim_config.lead_sine_amplitude * np.sin(
            self.sim_config.lead_sine_frequency * (index[start:] - start)
        )
        return velocity

    def _initial_state(
        self,
        lead_velocity: np.ndarray,
        platoon: np.ndarray,
        vehicle_model: VehicleStateModel,
    ) -> tuple[np.ndarray, np.ndarray]:
        lead_velocity = lead_velocity[: self.steps]
        lead_state = np.tile(lead_velocity, (2, 1)).T
        lead_state[0, 0] = 0
        headways = []
        for i in range(len(lead_velocity) - 1):
            lead_state[i + 1, 0] = lead_state[i, 0] + lead_velocity[i] * self.dt
        for i in range(self.history_length):
            headways.append(vehicle_model.idm_equilibrium_headway(lead_velocity[i])[0])

        lead_state[0 : self.history_length, :] = lead_state[0 : self.history_length, :]
        lead_state[0 : self.history_length, 0] = lead_state[0 : self.history_length, 0] - headways

        states = np.zeros((self.vehicle_count, self.steps, 5))
        equilibrium_headway, vehicle_lengths = vehicle_model.initial_state(lead_velocity[0])
        states[:, :, 0] = vehicle_lengths.reshape(-1, 1) * np.ones((self.vehicle_count, self.steps))
        states[:, 0, 2] = equilibrium_headway
        states[:, 0, 3] = lead_velocity[0]
        return states, lead_state

    def _mpc_reference(
        self,
        i: int,
        lead_velocity: np.ndarray,
        lead_state: np.ndarray,
        observed_states: np.ndarray,
        cav_index: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        reference = np.zeros([self.vehicle_count, self.horizon, 3])
        reference[:, :, 1] = np.mean(lead_velocity[i - self.horizon : i])
        for cav_position in range(1, len(cav_index[0])):
            upstream_index = cav_index[0][cav_position] - 1
            reference[cav_index[0][cav_position] :, :, 1] = np.mean(
                observed_states[upstream_index, i - self.horizon : i, 3]
            )
        reference[0, :, 0] = lead_state[i + 1 : i + self.horizon + 1, 0]
        return reference.swapaxes(0, 1).flatten()

    def _append_positions(self, states: np.ndarray, lead_state: np.ndarray) -> np.ndarray:
        positions = np.zeros([states.shape[0], states.shape[1], 1])
        positions[0, :, 0] = lead_state[:, 0]
        for i in range(1, self.vehicle_count):
            positions[i, :, 0] = positions[i - 1, :, 0] - states[i, :, 2]
        return np.concatenate([states, positions - positions[-1, 0, 0]], axis=-1)

    def _metrics(self, states: np.ndarray, computation_time: np.ndarray) -> dict[str, float]:
        active = computation_time[computation_time > 0]
        variance_start = min(100, states.shape[1] - 1)
        return {
            "mean_computation_time_all_steps": float(np.mean(computation_time)),
            "mean_computation_time_active_steps": float(np.mean(active)) if active.size else 0.0,
            "velocity_variance_after_step_100": float(np.var(states[:, variance_start:, 3])),
            "headway_variance_after_step_100": float(np.var(states[1:, variance_start:, 2])),
        }

    def save_result(self, result: SimulationResult) -> None:
        output_dir = result.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "simulation_result.npz",
            platoon=result.platoon,
            states=result.states,
            lead_reference=result.lead_reference,
            lead_state=result.lead_state,
            control_inputs=result.control_inputs,
            computation_time=result.computation_time,
        )
        self._write_metrics(output_dir / "metrics.json", result)
        self._write_vehicle_summary(output_dir / "vehicle_summary.csv", result)

    def _write_metrics(self, path: Path, result: SimulationResult) -> None:
        payload = {
            "metrics": result.metrics,
            "checkpoint_dir": str(self.sim_config.checkpoint_dir),
            "checkpoint_epoch": self.sim_config.checkpoint_epoch,
            "simulation_config": _json_ready(asdict(self.sim_config)),
            "model_config": _json_ready(asdict(self.model_config)),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_vehicle_summary(self, path: Path, result: SimulationResult) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["vehicle", "type", "mean_velocity", "velocity_variance", "mean_headway"])
            for vehicle_id in range(result.states.shape[0]):
                writer.writerow(
                    [
                        vehicle_id,
                        int(result.platoon[vehicle_id]),
                        float(np.mean(result.states[vehicle_id, :, 3])),
                        float(np.var(result.states[vehicle_id, min(100, result.states.shape[1] - 1) :, 3])),
                        float(np.mean(result.states[vehicle_id, :, 2])),
                    ]
                )


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
