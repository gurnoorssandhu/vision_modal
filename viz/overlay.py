"""All OpenCV drawing for the debug view / annotated stream.

Renders detections, track IDs, velocity arrows, the RK4-predicted "ghost" box and
trajectory, the ego danger corridor, per-object risk colouring, and a HUD with the
avoidance command + the async scene-model line.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from physics.state import X, Y, VX, VY, state_to_bbox


def _risk_color(risk: float):
    """Green (safe) -> red (danger) in BGR."""
    risk = float(np.clip(risk, 0.0, 1.0))
    return (0, int(255 * (1 - risk)), int(255 * risk))


def draw_corridor(frame, corridor_frac: float):
    h, w = frame.shape[:2]
    half = corridor_frac * w / 2.0
    cx = w / 2.0
    lo, hi = int(cx - half), int(cx + half)
    overlay = frame.copy()
    cv2.rectangle(overlay, (lo, 0), (hi, h), (255, 180, 0), -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
    cv2.line(frame, (lo, 0), (lo, h), (255, 180, 0), 1)
    cv2.line(frame, (hi, 0), (hi, h), (255, 180, 0), 1)


def _ibox(bbox):
    return tuple(int(v) for v in bbox)


def draw_item(frame, track, risk, traj):
    color = _risk_color(risk.risk)
    x1, y1, x2, y2 = _ibox(track.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"#{track.id} {track.label} r={risk.risk:.2f}",
                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if risk.ttc != float("inf"):
        cv2.putText(frame, f"TTC {risk.ttc:.1f}s", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    s = track.state
    cx, cy = int(s[X]), int(s[Y])
    # velocity arrow (~0.4s lookahead, just for visibility)
    vx, vy = s[VX], s[VY]
    cv2.arrowedLine(frame, (cx, cy), (int(cx + vx * 0.4), int(cy + vy * 0.4)),
                    (0, 255, 255), 2, tipLength=0.3)

    # RK4 predicted trajectory + ghost box at horizon
    pts = [(int(p[X]), int(p[Y])) for p in traj]
    for a, b in zip([(cx, cy)] + pts, pts):
        cv2.line(frame, a, b, (255, 0, 255), 1)
    if traj:
        gx1, gy1, gx2, gy2 = _ibox(state_to_bbox(traj[-1], track.aspect))
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 255), 1)


def draw_hud(frame, command, scene, fps: float, det_fps: float = 0.0):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.45, frame, 0.55, 0, frame)

    act_color = {"CLEAR": (0, 255, 0), "STOP": (0, 0, 255)}.get(command.action, (0, 200, 255))
    cv2.putText(frame, f"CMD: {command.action}  steer={command.steer:+.2f} "
                       f"speed={command.speed:.2f}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, act_color, 2)
    cv2.putText(frame, command.reason, (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    stale = "" if (scene.ok and scene.age() < 5) else " [stale]"
    sc = {"low": (0, 255, 0), "medium": (0, 200, 255), "high": (0, 0, 255)}.get(
        scene.caution_level, (200, 200, 200))
    scene_txt = f"SCENE({scene.caution_level}{stale}): {scene.scene}"
    cv2.putText(frame, scene_txt[:90], (8, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)

    if fps:
        cv2.putText(frame, f"{fps:4.1f} FPS", (w - 96, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if det_fps:
        cv2.putText(frame, f"det {det_fps:4.1f}", (w - 96, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def annotate(frame, items: List[dict], command, scene, fps: float, corridor_frac: float,
             det_fps: float = 0.0):
    draw_corridor(frame, corridor_frac)
    for it in items:
        draw_item(frame, it["track"], it["risk"], it["traj"])
    draw_hud(frame, command, scene, fps, det_fps)
    return frame
