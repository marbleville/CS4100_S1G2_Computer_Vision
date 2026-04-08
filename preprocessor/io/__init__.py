"""Frame input adapters, sinks, and source factory."""

from preprocessor.io.base import FrameSource
from preprocessor.io.factory import build_frame_source
from preprocessor.io.frame_packet_writer import DiskFramePacketWriter
from preprocessor.io.types import FramePacket
from preprocessor.io.video_file_source import VideoFileFrameSource
from preprocessor.io.webcam_source import WebcamFrameSource

__all__ = [
    "FramePacket",
    "FrameSource",
    "VideoFileFrameSource",
    "WebcamFrameSource",
    "DiskFramePacketWriter",
    "build_frame_source",
]
