"""Raspberry Pi camera source (Picamera2 / OV5647), threaded.

A background thread continuously pulls frames so the hot loop never blocks on
`capture_array`, and always gets the freshest frame. Imported lazily so the module
is harmless on a laptop that has no picamera2. Same interface as WebcamSource.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CameraSource


class PiCameraSource(CameraSource):
    def __init__(self, width: int = 640, height: int = 480, flip: bool = False):
        try:
            from picamera2 import Picamera2
        except ImportError as e:  # pragma: no cover - Pi-only path
            raise RuntimeError(
                "picamera2 not available. Install with: "
                "sudo apt install -y python3-picamera2"
            ) from e

        self.flip = flip
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.5)  # let AE/AWB settle

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        fails = 0
        while self._running:
            try:
                rgb = self.picam2.capture_array()    # RGB (blocks ~1/fps)
            except Exception as e:  # noqa: BLE001 - sensor/ribbon blip: don't kill thread
                fails += 1
                if fails in (1, 30):
                    print(f"camera capture error ({fails}x): {e} "
                          "-- check CSI ribbon / power", flush=True)
                time.sleep(0.1)
                continue
            fails = 0
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
        self.picam2.stop()
