"""Shared perception types — kept dependency-free so importing them never drags
in mediapipe / litert (the Pi has no mediapipe).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    bbox: tuple          # (x1, y1, x2, y2) in full-frame pixels
    label: str
    score: float
