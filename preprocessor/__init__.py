"""Public API for the preprocessor package."""

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.base import FrameSource
from preprocessor.io.factory import build_frame_source
from preprocessor.pipeline.processor import (
    PreprocessingPipeline,
    pipeline_result_to_hand_result,
)
from preprocessor.types import HandFrameResult, ResultStatus


class Preprocessor:
    """Default preprocessor composition of source + pipeline."""

    def __init__(self, config: PreprocessorConfig) -> None:
        self._source: FrameSource = build_frame_source(config)
        self._pipeline = PreprocessingPipeline(config)

    def get_current_hand_candidates(self) -> HandFrameResult:
        packet = self._source.read()
        if packet is None:
            return HandFrameResult(
                status=ResultStatus.NO_HAND,
                timestamp_ms=0,
                candidates_bbox_px=[],
                error_message="End of stream.",
            )
        pipeline_result = self._pipeline.process(packet)
        return pipeline_result_to_hand_result(pipeline_result)


def init_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    """Initialize and return the default preprocessor implementation."""
    return Preprocessor(config)


__all__ = [
    "Preprocessor",
    "PreprocessorConfig",
    "ResultStatus",
    "HandFrameResult",
    "init_preprocessor",
]
