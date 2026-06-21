"""Raspberry Pi camera source (Picamera2 / OV5647).

Imported lazily so the module is harmless on a laptop that has no picamera2.
Same interface as WebcamSource — the rest of the pipeline can't tell them apart.
"""
from __future__ import annotations

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

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        rgb = self.picam2.capture_array()       # RGB
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.flip:
            frame = cv2.flip(frame, 1)
        return frame, time.time()

    def release(self) -> None:
        self.picam2.stop()
