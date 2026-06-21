"""Camera source abstraction.

Every backend returns the *newest* frame as a BGR numpy array plus a capture
timestamp. Keeping this interface tiny is what lets a stereo / depth camera drop
in later without the physics or planning layers noticing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraSource(ABC):
    @abstractmethod
    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (frame_bgr, timestamp_seconds). frame is None if no frame yet."""

    @abstractmethod
    def release(self) -> None:
        ...

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
