"""Response/result dataclasses for preprocessor outputs."""

from dataclasses import dataclass, field

import numpy as np

from preprocessor.types.enums import ResultStatus


@dataclass(slots=True)
class HandCandidateFrame:
    """Normalized image crop for one detected hand candidate."""

    frame_rgb: np.ndarray
    timestamp_ms: int
    source_frame_index: int
    source_id: str
    candidate_index: int
    bbox_xyxy_px: tuple[int, int, int, int]


@dataclass(slots=True)
class HandFrameResult:
    """Output shape for current-frame hand features."""

    status: ResultStatus
    timestamp_ms: int
    candidates: list[HandCandidateFrame]
    error_message: str | None = None


@dataclass(slots=True)
class MotionWindowResult:
    """Output shape for current-window motion features."""

    status: ResultStatus
    timestamp_ms: int
    window_size: int
    trajectory_xy_norm: list[tuple[float, float]] = field(default_factory=list)
    delta_x_px: float = 0.0
    delta_y_px: float = 0.0
    path_length_px: float = 0.0
    motion_confidence: float | None = None
    error_message: str | None = None
