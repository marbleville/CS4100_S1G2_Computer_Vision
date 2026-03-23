"""Types for preprocessing pipeline outputs."""

from dataclasses import dataclass, field
import numpy as np

from preprocessor.pipeline.components import ComponentStats
from preprocessor.types.results import HandCandidateFrame


@dataclass(slots=True)
class PipelineFrameResult:
    """Intermediate result for one preprocessed frame."""

    timestamp_ms: int
    frame_index: int
    mask: np.ndarray
    candidates: list[ComponentStats]
    candidate_frames: list[HandCandidateFrame]
    debug: dict[str, float | int | str] = field(default_factory=dict)
