"""Types for preprocessing pipeline outputs."""

from dataclasses import dataclass, field
from preprocessor.pipeline.components import ComponentStats

import numpy as np


@dataclass(slots=True)
class PipelineFrameResult:
    """Intermediate result for one preprocessed frame."""

    timestamp_ms: int
    frame_index: int
    mask: np.ndarray
    candidates: list[ComponentStats]
    debug: dict[str, float | int | str] = field(default_factory=dict)
