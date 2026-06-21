"""MediaPipe Tasks object detector wrapper (EfficientDet-Lite, tflite + XNNPACK).

Runs detection on a downscaled copy of the frame for speed and scales boxes back
to full resolution. Returns plain dicts so nothing downstream depends on the
MediaPipe types.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


@dataclass
class Detection:
    bbox: tuple          # (x1, y1, x2, y2) in full-frame pixels
    label: str
    score: float


class ObjectDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.4,
                 max_results: int = 10, detect_size: int = 320,
                 allowed_labels: Optional[List[str]] = None):
        options = mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            score_threshold=score_threshold,
            max_results=max_results,
            category_allowlist=allowed_labels or None,
        )
        self._detector = mp_vision.ObjectDetector.create_from_options(options)
        self.detect_size = detect_size

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        scale = self.detect_size / max(h, w)
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            scale, small = 1.0, frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        inv = 1.0 / scale
        out: List[Detection] = []
        for det in result.detections:
            bb = det.bounding_box
            x1 = bb.origin_x * inv
            y1 = bb.origin_y * inv
            x2 = (bb.origin_x + bb.width) * inv
            y2 = (bb.origin_y + bb.height) * inv
            cat = det.categories[0] if det.categories else None
            out.append(Detection(
                bbox=(x1, y1, x2, y2),
                label=cat.category_name if cat else "object",
                score=float(cat.score) if cat else 0.0,
            ))
        return out

    def close(self) -> None:
        self._detector.close()
