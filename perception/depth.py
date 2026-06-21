"""Depth channel — monocular now, pluggable for stereo / depth cameras later.

v1 returns only a *relative* range proxy from bbox scale (range ~ 1/s) plus the
looming rate. No metric depth is claimed. The DepthChannel interface is the seam
where an OAK-D / RealSense / stereo-pair backend drops in: implement `estimate`
to return a metric depth and the physics layer can start consuming absolute range
without any other change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.state import S, VS


@dataclass
class RangeCue:
    relative_range: float   # arbitrary units, larger = farther (proxy = 1/s)
    looming_rate: float     # ds/dt; >0 approaching
    metric: bool = False    # True once a real depth sensor feeds this


class DepthChannel:
    """Base interface. Mono implementation below; stereo/depth-cam subclass later."""

    def estimate(self, state: np.ndarray) -> RangeCue:  # pragma: no cover - interface
        raise NotImplementedError


class MonoLoomingDepth(DepthChannel):
    def __init__(self, scale_const: float = 1.0):
        # range_proxy = scale_const / s ; constant only sets units, not used metrically
        self.scale_const = scale_const

    def estimate(self, state: np.ndarray) -> RangeCue:
        s = max(float(state[S]), 1e-3)
        return RangeCue(
            relative_range=self.scale_const / s,
            looming_rate=float(state[VS]),
            metric=False,
        )
