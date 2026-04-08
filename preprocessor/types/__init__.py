"""Stable public types for preprocessor interfaces."""

from preprocessor.config.types import PreprocessorConfig
from preprocessor.types.enums import ResultStatus
from preprocessor.types.results import (
    HandCandidateFrame,
    HandFrameResult,
    MotionWindowResult,
)

__all__ = [
    "PreprocessorConfig",
    "ResultStatus",
    "HandCandidateFrame",
    "HandFrameResult",
    "MotionWindowResult",
]
