"""6-state constant-velocity Kalman filter.

Smooths the noisy per-frame detection into a clean state estimate (position +
velocity in the [x, y, s] space). The detector gives jittery boxes; the KF gives
the stable velocity estimate that the RK4 rollout needs to predict anything sane.

The KF handles *estimation* (where is it, how fast); RK4 handles *prediction*
(where will it be) so a nonlinear motion model can be swapped in without touching
the estimator.
"""
from __future__ import annotations

import numpy as np

from .state import STATE_DIM, MEAS_DIM, X, Y, S, VX, VY, VS


class KalmanCV:
    def __init__(self, meas: np.ndarray, pos_var: float = 10.0, vel_var: float = 1e3,
                 process_var: float = 1.0, meas_var: float = 10.0):
        # state
        self.x = np.zeros(STATE_DIM, dtype=float)
        self.x[X], self.x[Y], self.x[S] = meas

        # covariance: confident on initial position, very unsure on velocity
        self.P = np.eye(STATE_DIM)
        self.P[X, X] = self.P[Y, Y] = self.P[S, S] = pos_var
        self.P[VX, VX] = self.P[VY, VY] = self.P[VS, VS] = vel_var

        # measurement matrix: observe x, y, s
        self.H = np.zeros((MEAS_DIM, STATE_DIM))
        self.H[0, X] = self.H[1, Y] = self.H[2, S] = 1.0

        self.R = np.eye(MEAS_DIM) * meas_var
        self._q = process_var

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(STATE_DIM)
        F[X, VX] = dt
        F[Y, VY] = dt
        F[S, VS] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # white-noise-acceleration process noise, block-diagonal per axis
        q = self._q
        Q = np.zeros((STATE_DIM, STATE_DIM))
        dt2, dt3 = dt * dt, dt * dt * dt
        for p, v in ((X, VX), (Y, VY), (S, VS)):
            Q[p, p] = dt3 / 3.0 * q
            Q[p, v] = Q[v, p] = dt2 / 2.0 * q
            Q[v, v] = dt * q
        return Q

    def predict(self, dt: float) -> None:
        if dt <= 0:
            dt = 1e-3
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, meas: np.ndarray) -> None:
        z = np.asarray(meas, dtype=float)
        y = z - self.H @ self.x
        S_ = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S_)
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K @ self.H) @ self.P

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()
