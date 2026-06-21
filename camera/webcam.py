"""Threaded webcam source.

A background thread continuously grabs frames and keeps only the latest one, so
the hot loop never blocks on I/O and never processes a stale frame. This is the
single biggest latency win on a laptop webcam.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CameraSource


class WebcamSource(CameraSource):
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, flip: bool = False):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # small internal buffer -> fresher frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")

        self.flip = flip
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._ts = time.time()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
