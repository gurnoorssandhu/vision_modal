"""Run any detector on its own thread.

The detector (a tflite invoke) is the slowest stage. Inline, it caps the whole
loop at detection FPS. Here the worker thread chews on the newest frame
continuously while the main loop renders/tracks every frame and coasts on the
Kalman prediction between fresh detections — so the displayed FPS decouples from
(and far exceeds) the detection rate.

Only this one thread ever calls `detector.detect()`, so the underlying tflite
interpreter is never touched concurrently.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from .types import Detection


class AsyncDetector:
    def __init__(self, detector):
        self.detector = detector
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._dets: List[Detection] = []
        self._seq = 0
        self.infer_ms = 0.0
        self.fps = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Tuple[List[Detection], int]:
        """Return (latest detections, sequence number). A new seq == fresh results."""
        with self._lock:
            return self._dets, self._seq

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                frame = self._frame
                self._frame = None        # consume; always work on the freshest
            if frame is None:
                time.sleep(0.003)
                continue
            t0 = time.time()
            dets = self.detector.detect(frame)
            dt = time.time() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            with self._lock:
                self._dets = dets
                self._seq += 1
                self.infer_ms = dt * 1000.0
                self.fps = inst if self.fps == 0 else 0.8 * self.fps + 0.2 * inst

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.detector.close()
