"""LiteRT (TFLite) object detector — mediapipe-free.

Runs an EfficientDet-Lite model that has the detection post-processing baked in
(TFLite_Detection_PostProcess), so a plain interpreter gives us
boxes / classes / scores / count directly — no anchor decoding. This is the
backend used on the Raspberry Pi (where mediapipe has no Python 3.13 wheel) and is
the default everywhere so the laptop and the Pi run identical code.

Labels are read from the model's embedded metadata (the .tflite is a zip).
"""
from __future__ import annotations

import zipfile
from typing import List, Optional

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # fall back to tflite_runtime if that's what's installed
    from tflite_runtime.interpreter import Interpreter  # type: ignore

from .types import Detection


def _intness(v: np.ndarray) -> float:
    """Mean fractional part of the non-zero entries; ~0 for an integer (class) array."""
    nz = v[v > 1e-6]
    if nz.size == 0:
        return 1.0
    return float(np.mean(np.abs(nz - np.round(nz))))


class LiteRTDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.4,
                 max_results: int = 10, allowed_labels: Optional[List[str]] = None,
                 num_threads: int = 4):
        self.interp = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interp.allocate_tensors()

        inp = self.interp.get_input_details()[0]
        self.in_index = inp["index"]
        self.in_h, self.in_w = int(inp["shape"][1]), int(inp["shape"][2])
        self.in_dtype = inp["dtype"]

        self.out_details = sorted(self.interp.get_output_details(),
                                  key=lambda d: d["index"])
        self.score_threshold = score_threshold
        self.max_results = max_results
        self.allowed = set(allowed_labels) if allowed_labels else None
        self.labels = self._load_labels(model_path)

    @staticmethod
    def _load_labels(path: str) -> List[str]:
        try:
            z = zipfile.ZipFile(path)
            for n in z.namelist():
                if n.endswith(".txt"):
                    return z.read(n).decode("utf-8").splitlines()
        except Exception:  # noqa: BLE001
            pass
        return [str(i) for i in range(90)]

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        if self.in_dtype == np.uint8:
            inp = resized.astype(np.uint8)[None]
        else:  # float model expects [-1, 1]
            inp = ((resized.astype(np.float32) - 127.5) / 127.5)[None]

        self.interp.set_tensor(self.in_index, inp)
        self.interp.invoke()
        tensors = [self.interp.get_tensor(d["index"]) for d in self.out_details]

        # identify outputs by shape/value, independent of tensor order
        boxes = next(t for t in tensors if t.ndim == 3 and t.shape[-1] == 4)[0]  # (N,4)
        per_det = [t[0] for t in tensors if t.ndim == 2]
        if len(per_det) < 2:                       # degenerate model
            return []
        a, b = per_det[0], per_det[1]
        if _intness(a) < _intness(b):
            classes, scores = a, b
        else:
            classes, scores = b, a

        out: List[Detection] = []
        for i in np.argsort(-scores):
            s = float(scores[i])
            if s < self.score_threshold:
                break
            cls = int(round(float(classes[i])))
            label = self.labels[cls] if 0 <= cls < len(self.labels) else str(cls)
            if self.allowed and label not in self.allowed:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            out.append(Detection(
                bbox=(float(xmin) * w, float(ymin) * h,
                      float(xmax) * w, float(ymax) * h),
                label=label, score=s))
            if len(out) >= self.max_results:
                break
        return out

    def close(self) -> None:
        pass
