"""Preprocessing pipeline public exports."""

from __future__ import annotations

from preprocessor.pipeline.types import PipelineFrameResult


def __getattr__(name: str) -> object:
    if name in {"PreprocessingPipeline", "pipeline_result_to_hand_result"}:
        from preprocessor.pipeline.processor import (
            PreprocessingPipeline,
            pipeline_result_to_hand_result,
        )

        exports = {
            "PreprocessingPipeline": PreprocessingPipeline,
            "pipeline_result_to_hand_result": pipeline_result_to_hand_result,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineFrameResult",
    "PreprocessingPipeline",
    "pipeline_result_to_hand_result",
]
