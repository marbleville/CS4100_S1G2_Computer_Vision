"""Typed configuration objects for preprocessor initialization."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PreprocessorConfig:
    """Configuration for `init_preprocessor`."""

    buffer_size: int
    async_process: bool
    input_mode: Literal["webcam", "local_video"]
    camera_device: int = 0
    image_dir: str | None = None
    video_path: str | None = None
    frame_size: tuple[int, int] = (640, 480)
    fps_hint: int = 30
    threshold_profile: str = "default"
