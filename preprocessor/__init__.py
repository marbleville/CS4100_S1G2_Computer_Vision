"""Public API for the preprocessor package."""

from __future__ import annotations

from preprocessor.config.types import (
    LightingSwitchConfig,
    PreprocessorConfig,
    SkinFusionProfile,
)
from preprocessor.io.base import FrameSource
from preprocessor.types import HandCandidateFrame, HandFrameResult, ResultStatus


class Preprocessor:
    """Default preprocessor composition of source + pipeline."""

    def __init__(self, config: PreprocessorConfig) -> None:
        from preprocessor.io.factory import build_frame_source
        from preprocessor.pipeline.processor import PreprocessingPipeline

        self._source: FrameSource = build_frame_source(config)
        self._pipeline: PreprocessingPipeline = PreprocessingPipeline(config)

    def get_current_hand_candidates(self) -> HandFrameResult:
        from preprocessor.pipeline.processor import pipeline_result_to_hand_result

        packet = self._source.read()
        if packet is None:
            return HandFrameResult(
                status=ResultStatus.NO_HAND,
                timestamp_ms=0,
                candidates=[],
                error_message="End of stream.",
            )
        pipeline_result = self._pipeline.process(packet)
        return pipeline_result_to_hand_result(pipeline_result)

    def next(self) -> HandCandidateFrame | None:
        """Return the next buffered candidate crop, reading more frames as needed."""
        candidate = self._pipeline._pop_next_candidate()
        if candidate is not None:
            return candidate

        while True:
            packet = self._source.read()
            if packet is None:
                return self._pipeline._pop_next_candidate()
            self._pipeline.process(packet)
            candidate = self._pipeline._pop_next_candidate()
            if candidate is not None:
                return candidate


def init_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    """Initialize and return the default preprocessor implementation."""
    return Preprocessor(config)


__all__ = [
    "Preprocessor",
    "PreprocessorConfig",
    "SkinFusionProfile",
    "LightingSwitchConfig",
    "ResultStatus",
    "HandCandidateFrame",
    "HandFrameResult",
    "init_preprocessor",
]
