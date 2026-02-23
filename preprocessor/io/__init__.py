"""Frame input adapters and source factory."""

from preprocessor.io.base import FrameSource
from preprocessor.io.factory import build_frame_source
from preprocessor.io.types import FramePacket
from preprocessor.io.video_file_source import VideoFileFrameSource

__all__ = [
    "FramePacket",
    "FrameSource",
    "VideoFileFrameSource",
    "build_frame_source",
]
