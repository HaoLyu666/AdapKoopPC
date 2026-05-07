from __future__ import annotations

import numpy as np

from ..config import SimulationConfig


class KoopmanOptimizer:
    def __init__(self, config: SimulationConfig, cav_index: tuple[np.ndarray, ...]):
        self.horizon = config.horizon
        self.vehicle_count = config.vehicle_count
        self.dimension = 3 * self.vehicle_count * self.horizon
        self.velocity_max = config.velocity_max
        self.velocity_min = config.velocity_min
        self.headway_max = config.headway_max
        self.headway_min = config.headway_min
        self.acceleration_max = config.acceleration_max
        self.acceleration_min = config.acceleration_min

        self.v_fitter = np.zeros((self.dimension, self.dimension))
        self.v_fitter[1::3, 1::3] = 1

        self.h_fitter = np.zeros((self.dimension, self.dimension))
        self.h_fitter[3 : 3 * self.vehicle_count : 3, 3 : 3 * self.vehicle_count : 3] = 1
        for i in range(1, self.horizon):
            row = slice(3 * self.vehicle_count * i, 3 * self.vehicle_count * (i + 1))
            self.h_fitter[row, row] = self.h_fitter[0 : 3 * self.vehicle_count, 0 : 3 * self.vehicle_count]

        self.h_lead_fitter = np.zeros((self.dimension, self.dimension))
        self.h_lead_fitter[0 :: 3 * self.vehicle_count, 0 :: 3 * self.vehicle_count] = 1

        self.a_cav_fitter = np.zeros((self.dimension, self.dimension))
        self.a_cav_fitter[2 + cav_index[0] * 3, 2 + cav_index[0] * 3] = 1
        for i in range(1, self.horizon):
            row = slice(3 * self.vehicle_count * i, 3 * self.vehicle_count * (i + 1))
            self.a_cav_fitter[row, row] = self.a_cav_fitter[0 : 3 * self.vehicle_count, 0 : 3 * self.vehicle_count]

    def objective(self, state: np.ndarray, reference: np.ndarray, H: np.ndarray, F_S: np.ndarray, F_R: np.ndarray):
        state = state[:, np.newaxis]
        return lambda x: float(0.5 * x.T @ H @ x + (state.T @ F_S - reference.T @ F_R) @ x)

    def constraints(self, A: np.ndarray, B: np.ndarray, state: np.ndarray, reference: np.ndarray) -> list[dict]:
        return [
            {"type": "ineq", "fun": lambda x: self.v_fitter @ (A @ state + B @ x - self.velocity_min)},
            {"type": "ineq", "fun": lambda x: self.v_fitter @ (-(A @ state + B @ x - self.velocity_max))},
            {"type": "ineq", "fun": lambda x: self.h_fitter @ (A @ state + B @ x - self.headway_min)},
            {"type": "ineq", "fun": lambda x: self.h_fitter @ (-(A @ state + B @ x - self.headway_max))},
            {"type": "ineq", "fun": lambda x: self.a_cav_fitter @ (A @ state + B @ x - self.acceleration_min)},
            {"type": "ineq", "fun": lambda x: self.a_cav_fitter @ (-(A @ state + B @ x - self.acceleration_max))},
        ]

    def bounds(self, cav_count: int) -> list[tuple[float, float]]:
        return [(self.acceleration_min, self.acceleration_max)] * self.horizon * cav_count
