"""Lightweight multi-object tracker (SORT-style).

Greedy IoU association of detections to existing tracks; each track carries its
own Kalman filter so it keeps a smoothed state (and survives short detector
dropouts). This is what gives objects persistent IDs and the velocity estimate
the physics layer predicts from.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

from physics.kalman import KalmanCV
from physics.state import bbox_to_measurement, state_to_bbox, X, Y
from .detector import Detection


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next_id = 0

    def __init__(self, det: Detection):
        Track._next_id += 1
        self.id = Track._next_id
        self.label = det.label
        self.score = det.score
        x1, y1, x2, y2 = det.bbox
        self.aspect = max(x2 - x1, 1.0) / max(y2 - y1, 1.0)
        self.kf = KalmanCV(bbox_to_measurement(det.bbox))
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history = deque(maxlen=30)
        self._record()

    def _record(self) -> None:
        s = self.kf.state
        self.history.append((float(s[X]), float(s[Y])))

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection) -> None:
        self.label = det.label
        self.score = det.score
        x1, y1, x2, y2 = det.bbox
        self.aspect = max(x2 - x1, 1.0) / max(y2 - y1, 1.0)
        self.kf.update(bbox_to_measurement(det.bbox))
        self.hits += 1
        self.time_since_update = 0
        self._record()

    @property
    def state(self) -> np.ndarray:
        return self.kf.state

    @property
    def bbox(self) -> tuple:
        return state_to_bbox(self.kf.state, self.aspect)


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []

    def update(self, detections: List[Detection], dt: float) -> List[Track]:
        for t in self.tracks:
            t.predict(dt)

        unmatched = set(range(len(detections)))
        # greedy: highest IoU pairs first
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, det in enumerate(detections):
                v = iou(t.bbox, det.bbox)
                if v >= self.iou_threshold:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(detections[di])
            used_t.add(ti)
            used_d.add(di)
            unmatched.discard(di)

        for di in unmatched:
            self.tracks.append(Track(detections[di]))

        # cull stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.hits >= self.min_hits or t.time_since_update == 0]
