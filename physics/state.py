"""State-vector conventions shared across the physics layer.

Each tracked object lives in a 6-D image-plane state:

    [x, y, s, vx, vy, vs]

    x, y   bbox centre in pixels
    s      bbox scale (sqrt(area)) — a monocular proxy for *looming*: bigger = closer
    vx,vy  centre velocity (px/s)
    vs     scale velocity (px/s). vs > 0 means the object is growing == approaching.

Working in (x, y, s) keeps the monocular pipeline honest: we never claim metric
depth, only relative range (1/s) and a physically-grounded looming rate. When a
stereo / depth camera is added, an absolute-depth channel augments this state
without changing its shape (see perception/depth.py).
"""
from __future__ import annotations

import numpy as np

# state indices
X, Y, S, VX, VY, VS = range(6)
STATE_DIM = 6
MEAS_DIM = 3   # we observe (x, y, s)


def bbox_to_measurement(bbox) -> np.ndarray:
    """(x1, y1, x2, y2) -> measurement [x, y, s]."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = float(np.sqrt(w * h))
    return np.array([cx, cy, s], dtype=float)


def state_to_bbox(state: np.ndarray, aspect: float) -> tuple:
    """state -> (x1, y1, x2, y2) using a fixed aspect (w/h) to recover the box."""
    cx, cy, s = state[X], state[Y], max(state[S], 1.0)
    # s = sqrt(w*h), w = aspect*h  ->  h = s/sqrt(aspect)
    h = s / np.sqrt(aspect)
    w = aspect * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
