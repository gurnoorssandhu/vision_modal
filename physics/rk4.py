"""Classical RK4 integrator + trajectory rollout.

This is the literal "RK4 analysis to predict future states" from the brief: given
the current estimated state and a motion model f, integrate the ODE forward over a
horizon and return the predicted trajectory.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

Dynamics = Callable[[np.ndarray], np.ndarray]


def rk4_step(state: np.ndarray, dt: float, f: Dynamics) -> np.ndarray:
    """One classical 4th-order Runge-Kutta step of an autonomous ODE state' = f(state)."""
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(state: np.ndarray, horizon_s: float, steps: int, f: Dynamics) -> List[np.ndarray]:
    """Integrate `state` forward over `horizon_s` in `steps` RK4 substeps.

    Returns the list of predicted states (length == steps), the last being the
    state at the full horizon.
    """
    dt = horizon_s / max(steps, 1)
    traj: List[np.ndarray] = []
    s = state.astype(float).copy()
    for _ in range(steps):
        s = rk4_step(s, dt, f)
        traj.append(s.copy())
    return traj
