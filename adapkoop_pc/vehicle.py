from __future__ import annotations

import numpy as np

from .config import SimulationConfig


class VehicleStateModel:
    """IDM-based vehicle state reconstruction used in the mixed platoon simulation."""

    def __init__(self, config: SimulationConfig, platoon: np.ndarray):
        self.vehicle_count = config.vehicle_count
        self.dt = config.time_step
        self.cav_penetration = config.cav_penetration
        self.platoon = platoon

        hdv_types = self.platoon[np.where(self.platoon != 0)]
        self.hdv_count = len(hdv_types)
        self.a_idm = np.zeros(self.hdv_count)
        self.ve = np.zeros_like(self.a_idm)
        self.b_idm = np.zeros_like(self.a_idm)
        self.s0 = np.zeros_like(self.a_idm)
        self.T0 = np.zeros_like(self.a_idm)
        self.delta = config.idm_delta
        self.hdv_length = np.zeros_like(self.a_idm)

        for vehicle_type in (1, 2):
            mask = hdv_types == vehicle_type
            idx = vehicle_type - 1
            self.a_idm[mask] = config.idm_acceleration[idx]
            self.ve[mask] = config.idm_desired_velocity[idx]
            self.b_idm[mask] = config.idm_comfortable_deceleration[idx]
            self.s0[mask] = config.idm_min_spacing[idx]
            self.T0[mask] = config.idm_time_headway[idx]
            self.hdv_length[mask] = config.vehicle_length[idx]

    def hdv_vehicle_lengths(self) -> np.ndarray:
        return self.hdv_length

    def idm_acceleration(self, delta_velocity: np.ndarray, headway: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        desired_spacing = self.s0 + np.maximum(
            0,
            self.T0 * velocity
            - (velocity * delta_velocity) / (2 * np.sqrt(self.a_idm * self.b_idm)),
        )
        return self.a_idm * (
            1
            - np.power(velocity / self.ve, self.delta)
            - np.power(desired_spacing / (headway - self.hdv_length), 2)
        )

    def idm_equilibrium_headway(self, velocity: float) -> np.ndarray:
        desired_spacing = self.s0 + np.maximum(0, self.T0 * velocity)
        return desired_spacing / np.sqrt(1 - np.power(velocity / self.ve, self.delta)) + self.hdv_length

    def initial_state(self, velocity: float) -> tuple[np.ndarray, np.ndarray]:
        equilibrium_headway = self.idm_equilibrium_headway(velocity)
        headway = np.zeros(self.vehicle_count)
        headway[self.platoon == 1] = np.min(equilibrium_headway)
        headway[self.platoon == 0] = np.min(equilibrium_headway)
        headway[self.platoon == 2] = np.max(equilibrium_headway)

        vehicle_length = np.zeros(self.vehicle_count)
        vehicle_length[self.platoon != 2] = np.min(self.hdv_length)
        vehicle_length[self.platoon == 2] = np.max(self.hdv_length)
        return headway, vehicle_length

    def next_vehicle_state(
        self,
        velocity: np.ndarray,
        headway: np.ndarray,
        lead_velocity: float,
        acceleration: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_velocity = velocity + acceleration * self.dt
        next_delta_velocity = np.concatenate(([lead_velocity], next_velocity[:-1])) - next_velocity
        next_headway = headway + next_delta_velocity * self.dt
        return next_velocity, next_delta_velocity, next_headway

    def next_lead_state(
        self,
        acceleration: float,
        jerk: float,
        velocity: float,
        position: float,
    ) -> tuple[float, float]:
        next_velocity = velocity + acceleration * self.dt + jerk * 0.5 * self.dt**2
        next_position = position + velocity * self.dt
        return next_velocity, next_position
