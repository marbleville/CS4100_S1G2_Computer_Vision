"""Response/result dataclasses for preprocessor outputs."""

from dataclasses import dataclass, field

from preprocessor.types.enums import ResultStatus


@dataclass(slots=True)
class HandFrameResult:
    """Output shape for current-frame hand features."""

    status: ResultStatus
    timestamp_ms: int
    candidates_bbox_px: list[tuple[int, int, int, int]]
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
