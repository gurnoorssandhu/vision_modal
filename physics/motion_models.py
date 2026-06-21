"""Motion models = the ODE right-hand side f(state) the RK4 integrator rolls forward.

Each function maps a 6-D state to its time-derivative. Constant-velocity is the
baseline; the drag model adds a velocity-proportional decel term, which is where
RK4 (vs naive linear extrapolation) actually earns its place — the nonlinear term
makes the closed-form solution awkward but RK4 integrates it for free.
"""
from __future__ import annotations

import numpy as np

from .state import X, Y, S, VX, VY, VS


def constant_velocity(state: np.ndarray) -> np.ndarray:
    """d/dt [x,y,s,vx,vy,vs] = [vx,vy,vs,0,0,0]."""
    d = np.zeros_like(state)
    d[X] = state[VX]
    d[Y] = state[VY]
    d[S] = state[VS]
    return d


def drag(state: np.ndarray, k: float = 0.5) -> np.ndarray:
    """Constant velocity with linear drag on each velocity component (dv = -k v)."""
    d = constant_velocity(state)
    d[VX] = -k * state[VX]
    d[VY] = -k * state[VY]
    d[VS] = -k * state[VS]
    return d


def get_model(name: str, drag_coeff: float = 0.5):
    """Return a callable f(state) -> dstate for the named model."""
    if name == "drag":
        return lambda s: drag(s, drag_coeff)
    return constant_velocity
