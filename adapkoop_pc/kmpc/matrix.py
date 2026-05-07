from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from ..checkpoints import checkpoint_set, load_state
from ..config import ModelConfig, SimulationConfig
from ..models import KoopmanSpace


class KoopmanMatrixBuilder:
    """Build the lifted linear dynamics matrices for the MPC problem."""

    def __init__(self, sim_config: SimulationConfig, model_config: ModelConfig):
        self.sim_config = sim_config
        self.model_config = model_config
        self.args = sim_config.to_legacy_args(model_config)
        self.checkpoint_dir = Path(sim_config.checkpoint_dir)
        self.device = torch.device(sim_config.device)
        self.horizon = sim_config.horizon
        self.vehicle_count = sim_config.vehicle_count
        self.koopman_state_size = model_config.hidden_size
        self.dt = sim_config.time_step
        self.cav_penetration = sim_config.cav_penetration
        self.truck_penetration = sim_config.truck_penetration

        (
            self.platoon,
            self.cav_count,
            self.hdv_count,
            self.truck_count,
        ) = self.generate_arrangement()

        self.a_hdv = np.zeros((self.koopman_state_size, self.koopman_state_size))
        self.b_hdv = np.zeros((self.koopman_state_size, 1))
        self.c_hdv = np.zeros((3, self.koopman_state_size))

        self.a_cav = np.eye(3)
        self.a_cav[0, 1] = -self.dt
        self.a_cav[1, 2] = self.dt
        self.b_cav = np.array([[0.0], [0.5 * self.dt**2], [self.dt]])
        self.c_cav = np.eye(3)

        if self.cav_count < 1:
            raise ValueError("At least one CAV is required because the lead vehicle is controlled.")
        non_cav_count = self.vehicle_count - self.cav_count
        self.state_dimension = self.koopman_state_size * non_cav_count + 3 * self.cav_count
        self.output_dimension = 3 * self.vehicle_count * self.horizon
        self.A = np.zeros((self.state_dimension, self.state_dimension))
        self.A[0:3, 0:3] = self.a_cav
        self.A[0, 1] = self.dt
        self.A[0, 2] = 0.5 * self.dt**2

        self.B = np.zeros((self.state_dimension, self.cav_count))
        self.B[1, 0] = 0.5 * self.dt**2
        self.B[2, 0] = self.dt

        self.C = np.zeros((3 * self.vehicle_count, self.state_dimension))
        self.C[0:3, 0:3] = self.c_cav

        self.R = np.eye(self.horizon * self.cav_count) * sim_config.control_weight
        q_cav = np.diag([0, sim_config.velocity_weight, 0])
        q_hdv = np.diag([0, 0, sim_config.velocity_difference_weight])
        self.Q = np.zeros((self.output_dimension, self.output_dimension))

        for i in np.where(self.platoon == 0)[0]:
            self.Q[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = q_cav
        for i in np.where(self.platoon != 0)[0]:
            self.Q[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = q_hdv
        for j in range(1, self.horizon):
            start = 3 * self.vehicle_count * j
            end = 3 * self.vehicle_count * (j + 1)
            self.Q[start:end, start:end] = self.Q[0 : 3 * self.vehicle_count, 0 : 3 * self.vehicle_count]

    def generate_arrangement(self) -> tuple[np.ndarray, int, int, int]:
        cav_count = int(self.vehicle_count * self.cav_penetration)
        truck_count = int(self.vehicle_count * self.truck_penetration)
        hdv_count = self.vehicle_count - cav_count - truck_count

        arrangement = [0] * cav_count + [1] * hdv_count + [2] * truck_count
        tail = arrangement[1:]
        random.Random(self.sim_config.arrangement_seed).shuffle(tail)
        platoon = np.array([arrangement[0]] + tail)
        return platoon, cav_count, hdv_count, truck_count

    def load_hdv_koopman_block(self) -> None:
        paths = checkpoint_set(self.checkpoint_dir, self.sim_config.checkpoint_epoch)
        decoder = KoopmanSpace(self.args)
        load_state(decoder, paths.koopman, self.device)
        weights = decoder.state_dict()
        self.a_hdv = weights["A.weight"].detach().cpu().numpy()
        self.b_hdv = weights["B.weight"].detach().cpu().numpy()
        self.c_hdv = weights["C.weight"].detach().cpu().numpy()[[1, 2, 0]]

    def generate_index_array(self) -> np.ndarray:
        index_array = []
        current_index = 0
        index_array.append(current_index)
        for element in self.platoon:
            current_index += 3 if element == 0 else self.koopman_state_size
            index_array.append(current_index)
        return np.array(index_array)

    def build_single_step_blocks(self) -> None:
        self.index = self.generate_index_array()
        cav_j = 1

        for i in range(1, self.vehicle_count):
            if self.platoon[i] == 0:
                self.A[self.index[i] : self.index[i + 1], self.index[i] : self.index[i + 1]] = self.a_cav
                if self.platoon[i - 1] == 0:
                    self.A[self.index[i], self.index[i] - 2] = self.dt
                else:
                    self.A[self.index[i], self.index[i - 1] : self.index[i]] = self.dt * self.c_hdv[1:-1]

                self.B[self.index[i] : self.index[i + 1], cav_j : cav_j + 1] = self.b_cav
                self.C[3 * i : 3 * (i + 1), self.index[i] : self.index[i + 1]] = self.c_cav
                cav_j += 1
            else:
                self.A[self.index[i] : self.index[i + 1], self.index[i] : self.index[i + 1]] = self.a_hdv
                self.C[3 * i : 3 * (i + 1), self.index[i] : self.index[i + 1]] = self.c_hdv
                if self.platoon[i - 1] == 0:
                    self.A[self.index[i] : self.index[i + 1], self.index[i] - 2 : self.index[i] - 1] = self.b_hdv
                else:
                    self.A[self.index[i] : self.index[i + 1], self.index[i - 1] : self.index[i]] = np.dot(
                        self.b_hdv,
                        self.c_hdv[1:-1],
                    )

    def lifted_A(self) -> np.ndarray:
        rows = np.empty((0, self.A.shape[1]))
        for i in range(self.horizon):
            rows = np.concatenate((rows, self.C @ np.linalg.matrix_power(self.A, i + 1)), axis=0)
        return rows

    def lifted_B(self) -> np.ndarray:
        result = np.zeros((self.output_dimension, self.horizon * self.cav_count))
        for i in range(self.horizon):
            for j in range(i + 1):
                row = slice(i * self.C.shape[0], (i + 1) * self.C.shape[0])
                col = slice(j * self.cav_count, (j + 1) * self.cav_count)
                result[row, col] = self.C @ np.linalg.matrix_power(self.A, i - j) @ self.B
        return result

    def weight_matrices(self, lifted_A: np.ndarray, lifted_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = 2 * (lifted_B.T @ self.Q @ lifted_B + self.R)
        F_S = 2 * lifted_A.T @ self.Q @ lifted_B
        F_R = 2 * self.Q @ lifted_B
        return H, F_S, F_R

    def build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        self.load_hdv_koopman_block()
        self.build_single_step_blocks()
        lifted_A = self.lifted_A()
        lifted_B = self.lifted_B()
        H, F_S, F_R = self.weight_matrices(lifted_A, lifted_B)
        return (
            lifted_A,
            lifted_B,
            H,
            F_S,
            F_R,
            self.platoon,
            self.cav_count,
            self.hdv_count,
            self.truck_count,
        )
