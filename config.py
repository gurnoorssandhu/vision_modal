"""Central configuration for the vision_modal pipeline.

One dataclass holds every tunable so the hot loop, physics, and viz stay in sync.
Values are deliberately conservative for laptop-webcam testing; the Pi port tunes
resolution / frame-skip down from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # --- camera ---
    source: str = "webcam"            # "webcam" | "picamera"
    cam_index: int = 0                # AVFoundation index 0 on macOS
    width: int = 640
    height: int = 480
    flip: bool = False

    # --- detector ---
    model_path: str = "models/efficientdet_lite0.tflite"
    score_threshold: float = 0.4
    max_results: int = 10
    detect_size: int = 320            # downscale long edge fed to the detector
    detect_every: int = 1            # run detector every N frames (>1 = interpolate)
    # category allow-list (None = keep all COCO classes the model emits)
    allowed_labels: Optional[list] = None

    # --- tracker ---
    iou_match_threshold: float = 0.3
    max_age: int = 15                 # frames a track survives unmatched
    min_hits: int = 2                 # hits before a track is "confirmed"

    # --- physics / prediction ---
    motion_model: str = "constant_velocity"   # "constant_velocity" | "drag"
    drag_coeff: float = 0.5           # used only by the drag model
    predict_horizon_s: float = 1.0    # how far ahead RK4 rolls the state
    predict_steps: int = 10           # RK4 substeps across the horizon

    # --- collision / risk ---
    corridor_frac: float = 0.4        # central band width (fraction of image) = ego heading
    ttc_warn_s: float = 2.0           # time-to-collision below this starts raising risk
    ttc_stop_s: float = 0.8           # below this = hard STOP
    risk_threshold: float = 0.5       # risk above which an avoidance command fires

    # --- scene reasoning (Claude) ---
    llm_enabled: bool = True          # auto-disabled if ANTHROPIC_API_KEY unset
    llm_model: str = "claude-haiku-4-5"   # fast/cheap scene labels
    llm_interval_s: float = 1.5       # min seconds between API calls
    llm_jpeg_quality: int = 60
    llm_max_tokens: int = 512

    # --- runtime / output ---
    headless: bool = False            # True on Pi: serve annotated MJPEG instead of a window
    stream_port: int = 8000
    show_fps: bool = True


def load() -> Config:
    """Hook for env / CLI overrides later; returns defaults for now."""
    return Config()
