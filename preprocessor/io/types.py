"""Typed frame packet emitted by frame sources."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FramePacket:
    """Normalized frame payload consumed by downstream preprocessing."""

    frame_index: int
    timestamp_ms: int
    frame_rgb: np.ndarray
    source_id: str
