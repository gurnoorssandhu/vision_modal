"""Offline smoke test: drives the full physics/planning/render path with a
synthetic approaching object (no camera, no network). Verifies the detector loads
and that a looming, centred object eventually triggers a non-CLEAR command.

Run: python tests/smoke_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg_mod
from perception.detector import ObjectDetector, Detection
from perception.tracker import Tracker
from physics import rk4, collision
from physics.motion_models import get_model
from planning import avoidance
from viz import overlay
from reasoning.scene_llm import SceneResult


def main() -> int:
    cfg = cfg_mod.load()
    W, H = cfg.width, cfg.height
    model_f = get_model(cfg.motion_model, cfg.drag_coeff)

    # 1) detector loads and runs on a synthetic frame without error
    if os.path.exists(cfg.model_path):
        det = ObjectDetector(cfg.model_path, cfg.score_threshold, cfg.max_results,
                             cfg.detect_size, cfg.allowed_labels)
        frame = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        out = det.detect(frame)
        print(f"detector OK ({len(out)} detections on noise frame)")
        det.close()
    else:
        print("detector model missing -> skipping live detector check")

    # 2) synthetic approaching object, centred, growing box (looming)
    tracker = Tracker(cfg.iou_match_threshold, cfg.max_age, cfg.min_hits)
    cx, cy = W / 2.0, H / 2.0
    dt = 1.0 / 30.0
    actions = []
    last = None
    for i in range(40):
        size = 40 + i * 6                      # box grows -> approaching
        x1, y1 = cx - size / 2, cy - size / 2
        x2, y2 = cx + size / 2, cy + size / 2
        dets = [Detection(bbox=(x1, y1, x2, y2), label="person", score=0.9)]
        tracks = tracker.update(dets, dt)

        risks = []
        items = []
        for t in tracks:
            traj = rk4.rollout(t.state, cfg.predict_horizon_s, cfg.predict_steps, model_f)
            r = collision.evaluate(t.state, traj, W, H, cfg.corridor_frac,
                                   cfg.ttc_warn_s, cfg.ttc_stop_s)
            risks.append(r)
            items.append({"track": t, "risk": r, "traj": traj})
        cmd = avoidance.decide(risks, cfg.risk_threshold, cfg.ttc_stop_s)

        # exercise the renderer on a blank frame
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        overlay.annotate(frame, items, cmd, SceneResult(scene="test", ok=True), 30.0,
                         cfg.corridor_frac)

        if cmd.action != last:
            print(f"  frame {i:2d}: {cmd.action:5s} | {cmd.reason}")
            last = cmd.action
        actions.append(cmd.action)

    assert any(a != "CLEAR" for a in actions), "expected a non-CLEAR command for approaching object"
    print("PIPELINE OK: approaching object raised a command ->",
          sorted(set(actions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
