"""Types for preprocessing pipeline outputs."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class PipelineFrameResult:
    """Intermediate result for one preprocessed frame."""

    timestamp_ms: int
    frame_index: int
    mask: np.ndarray
    selected_label: int | None
    selected_bbox_xyxy_px: tuple[int, int, int, int] | None
    selected_centroid_xy_px: tuple[float, float] | None
    selected_area_px: int | None
    candidate_count: int
    quality_score: float
    debug: dict[str, float | int | str] = field(default_factory=dict)
