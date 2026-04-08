"""Factory for building frame sources from preprocessor config."""

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.base import FrameSource
from preprocessor.io.video_file_source import VideoFileFrameSource
from preprocessor.io.webcam_source import WebcamFrameSource


def build_frame_source(config: PreprocessorConfig) -> FrameSource:
    """Build a frame source for the provided config."""
    if config.input_mode == "webcam":
        return WebcamFrameSource(config)
    if config.input_mode == "local_video":
        return VideoFileFrameSource(config)
    raise ValueError(f"Unsupported input mode: {config.input_mode}.")
