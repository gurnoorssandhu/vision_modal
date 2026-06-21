"""Collision risk from predicted trajectories.

Monocular, so everything is relative: we use the looming rate (vs) for a
time-to-collision estimate and the central image band as the ego "danger
corridor" (where we're heading). Risk fuses: are we approaching, will the
predicted path cross our corridor, and how soon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .state import X, Y, S, VS


@dataclass
class RiskResult:
    ttc: float                 # seconds (inf if not approaching)
    in_corridor: bool          # predicted path enters the ego heading band
    risk: float                # 0..1
    lateral_offset: float      # signed, +1 = object to the right of centre, -1 = left
    pred_center: tuple         # (x, y) at the prediction horizon


def time_to_collision(state: np.ndarray) -> float:
    """Looming TTC: tau = s / (ds/dt). Only meaningful while growing (approaching)."""
    s, vs = state[S], state[VS]
    if vs <= 1e-3:
        return float("inf")
    return float(s / vs)


def evaluate(current: np.ndarray, traj: List[np.ndarray], img_w: int, img_h: int,
             corridor_frac: float, ttc_warn: float, ttc_stop: float) -> RiskResult:
    cx_img = img_w / 2.0
    half_corr = corridor_frac * img_w / 2.0
    corr_lo, corr_hi = cx_img - half_corr, cx_img + half_corr

    ttc = time_to_collision(current)

    # does the current or any predicted centre fall inside the corridor?
    xs = [current[X]] + [s[X] for s in traj]
    in_corridor = any(corr_lo <= x <= corr_hi for x in xs)

    # closest lateral approach to the corridor edge (0 if it enters)
    nearest = min(abs(x - cx_img) for x in xs)
    corridor_factor = 1.0 if nearest <= half_corr else max(
        0.0, 1.0 - (nearest - half_corr) / max(half_corr, 1.0)
    )

    # ttc -> 0..1 urgency
    if ttc >= ttc_warn:
        ttc_factor = 0.0
    elif ttc <= ttc_stop:
        ttc_factor = 1.0
    else:
        ttc_factor = (ttc_warn - ttc) / max(ttc_warn - ttc_stop, 1e-3)

    risk = float(np.clip(ttc_factor * corridor_factor, 0.0, 1.0))

    end = traj[-1] if traj else current
    lateral_offset = float(np.clip((end[X] - cx_img) / max(cx_img, 1.0), -1.0, 1.0))

    return RiskResult(
        ttc=ttc,
        in_corridor=in_corridor,
        risk=risk,
        lateral_offset=lateral_offset,
        pred_center=(float(end[X]), float(end[Y])),
    )
