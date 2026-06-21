"""Async scene reasoning via the OpenAI API (vision).

Runs on its OWN background thread, off the reactive hot path. Every
`interval_s` it grabs the newest frame, JPEG-encodes it, and asks the model for a
structured scene assessment. The reactive loop never waits on this — it reads the
last cached result, which simply goes stale if the network/API is unavailable.

Model defaults to gpt-4o-mini (fast, cheap, vision). Structured output via
response_format json_schema (strict) guarantees a parseable result.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "description": "one-sentence description of the scene"},
        "hazards": {
            "type": "array",
            "items": {"type": "string"},
            "description": "navigation hazards a wheelchair/robot should watch",
        },
        "caution_level": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["scene", "hazards", "caution_level"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are the high-level scene-understanding module of a wheelchair navigation "
    "robot. You receive a single camera frame. Describe the scene in one sentence "
    "and list concrete navigation hazards (curbs, stairs, people, vehicles, poles, "
    "uneven ground). Be terse and practical."
)

_PROMPT = "Assess this frame for navigation. Return scene, hazards, caution_level."


@dataclass
class SceneResult:
    scene: str = "(waiting for scene model...)"
    hazards: List[str] = field(default_factory=list)
    caution_level: str = "low"
    ts: float = 0.0
    ok: bool = False
    error: str = ""

    def age(self) -> float:
        return time.time() - self.ts if self.ts else float("inf")


class SceneReasoner:
    def __init__(self, model: str, interval_s: float = 1.5,
                 jpeg_quality: int = 60, max_tokens: int = 512):
        self.model = model
        self.interval_s = interval_s
        self.jpeg_quality = jpeg_quality
        self.max_tokens = max_tokens

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._result = SceneResult()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._client = None
        self.available = False

        try:
            from openai import OpenAI  # lazy: pipeline still runs if SDK/key missing
            self._client = OpenAI()    # reads OPENAI_API_KEY
            self.available = True
        except Exception as e:  # noqa: BLE001
            self._result = SceneResult(scene=f"(scene model disabled: {e})")

    def set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> SceneResult:
        with self._lock:
            return self._result

    def start(self) -> None:
        if not self.available:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            t0 = time.time()
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is not None:
                self._query(frame)
            # pace to interval, regardless of API latency
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.interval_s - elapsed))

    def _query(self, frame: np.ndarray) -> None:
        try:
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return
            b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                    ]},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "scene_assessment",
                        "strict": True,
                        "schema": _SCHEMA,
                    },
                },
            )
            text = resp.choices[0].message.content or "{}"
            data = json.loads(text)
            result = SceneResult(
                scene=data.get("scene", ""),
                hazards=data.get("hazards", []),
                caution_level=data.get("caution_level", "low"),
                ts=time.time(),
                ok=True,
            )
        except Exception as e:  # noqa: BLE001 - never let the API kill the thread
            with self._lock:
                prev = self._result
            result = SceneResult(
                scene=prev.scene, hazards=prev.hazards,
                caution_level=prev.caution_level, ts=prev.ts,
                ok=False, error=str(e)[:120],
            )
        with self._lock:
            self._result = result
