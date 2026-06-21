"""Avoidance decision: risk assessments -> a steering command.

v1 emits the command (and a continuous steer vector) for logging / HUD only. The
`to_motor` seam is where a Pi motor driver plugs in later — nothing above this
layer needs to change to actuate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from physics.collision import RiskResult


@dataclass
class Command:
    action: str         # "CLEAR" | "LEFT" | "RIGHT" | "STOP"
    steer: float        # -1..1, + = steer right, - = steer left
    speed: float        # 0..1 speed scale
    reason: str


def decide(risks: List[RiskResult], risk_threshold: float, ttc_stop: float) -> Command:
    hazards = [r for r in risks if r.risk >= risk_threshold and r.in_corridor]
    if not hazards:
        return Command("CLEAR", 0.0, 1.0, "no obstacle in corridor")

    worst = max(hazards, key=lambda r: r.risk)
    speed = max(0.0, 1.0 - worst.risk)

    # head-on and very close -> stop
    if worst.ttc <= ttc_stop and abs(worst.lateral_offset) < 0.3:
        return Command("STOP", 0.0, 0.0,
                       f"head-on TTC={worst.ttc:.2f}s risk={worst.risk:.2f}")

    # steer away from the obstacle's side
    if worst.lateral_offset >= 0:        # obstacle on the right -> go left
        steer = -worst.risk
        action = "LEFT"
    else:                                # obstacle on the left -> go right
        steer = worst.risk
        action = "RIGHT"

    return Command(action, steer, speed,
                   f"avoid TTC={worst.ttc:.2f}s risk={worst.risk:.2f} off={worst.lateral_offset:+.2f}")


def to_motor(cmd: Command) -> Tuple[float, float]:
    """Placeholder mapping command -> (left_wheel, right_wheel) in [-1, 1].

    Differential-drive mix: forward speed +/- steer. Wire a GPIO motor driver here
    on the Pi. Returns the values without touching hardware in v1.
    """
    base = cmd.speed
    left = base + cmd.steer
    right = base - cmd.steer
    clamp = lambda v: max(-1.0, min(1.0, v))
    return clamp(left), clamp(right)
