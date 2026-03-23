"""Typed configuration objects for preprocessor initialization."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PreprocessorConfig:
    """Configuration for `init_preprocessor`."""

    input_mode: Literal["webcam", "local_video"]
    video_path: str | None = None
    frame_size: tuple[int, int] = (640, 480)
    threshold_profile: str = "default"
    candidate_frame_size_px: int = 128
    candidate_buffer_size: int = 32

    def __post_init__(self) -> None:
        frame_width, frame_height = self.frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("`frame_size` dimensions must be positive.")
        if self.candidate_frame_size_px <= 0:
            raise ValueError("`candidate_frame_size_px` must be positive.")
        if self.candidate_buffer_size <= 0:
            raise ValueError("`candidate_buffer_size` must be positive.")
