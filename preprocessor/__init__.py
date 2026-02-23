"""Public API for the preprocessor package."""

from typing import Protocol

from preprocessor.config.types import PreprocessorConfig
from preprocessor.types import HandFrameResult, MotionWindowResult, ResultStatus


class Preprocessor(Protocol):
    """Public preprocessor interface used by downstream modules."""

    def get_current_hand(self) -> HandFrameResult:
        """Return hand features for the latest frame."""

    def get_current_motion(self) -> MotionWindowResult:
        """Return motion features for the current temporal window."""


def init_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    """Initialize and return a preprocessor implementation.

    Concrete implementation is intentionally out of scope for Phase 1.
    """
    raise NotImplementedError("Phase 1 defines interfaces/types only.")


__all__ = [
    "Preprocessor",
    "PreprocessorConfig",
    "ResultStatus",
    "HandFrameResult",
    "MotionWindowResult",
    "init_preprocessor",
]
