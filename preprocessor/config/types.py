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
