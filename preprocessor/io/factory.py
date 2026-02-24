"""Factory for building frame sources from preprocessor config."""

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.base import FrameSource
from preprocessor.io.video_file_source import VideoFileFrameSource


def build_frame_source(config: PreprocessorConfig) -> FrameSource:
    """Build a frame source for the provided config.

    Local video source takes precedence when `video_path` is provided.
    """
    if config.video_path and config.input_mode == "local_video":
        return VideoFileFrameSource(config)
    raise NotImplementedError("Only `video_path` is supported in Phase 2.")
