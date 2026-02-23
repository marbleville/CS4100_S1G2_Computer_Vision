"""Local video file frame source."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np
from PIL import Image

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.types import FramePacket


class VideoFileFrameSource:
    """Pull-based frame source that streams frames from a local video file."""

    def __init__(self, config: PreprocessorConfig) -> None:
        if not config.video_path:
            raise ValueError(
                "`video_path` is required for VideoFileFrameSource.")
        self._config = config
        self._video_path = config.video_path
        self._fps: float | None = None
        self._frame_iter: Iterator[np.ndarray] | None = None
        self._frame_index = 0
        self._is_open = False
        self._exhausted = False
        self._source_id = Path(self._video_path).name

    def open(self) -> None:
        if self._is_open:
            return
        metadata = iio.immeta(self._video_path)
        fps = metadata.get("fps")
        if fps is None or float(fps) <= 0:
            raise ValueError("Video metadata must include positive `fps`.")
        self._fps = float(fps)
        self._frame_iter = iio.imiter(self._video_path)
        self._frame_index = 0
        self._exhausted = False
        self._is_open = True

    def read(self) -> FramePacket | None:
        if not self._is_open:
            self.open()
        if self._exhausted:
            return None
        assert self._frame_iter is not None
        assert self._fps is not None
        try:
            raw_frame = next(self._frame_iter)
        except StopIteration:
            self._exhausted = True
            return None

        frame_rgb = self._normalize_frame(raw_frame)
        timestamp_ms = int((self._frame_index / self._fps) * 1000)
        packet = FramePacket(
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
            frame_rgb=frame_rgb,
            source_id=self._source_id,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        self._frame_iter = None
        self._fps = None
        self._is_open = False
        self._exhausted = False

    def _normalize_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(raw_frame)
        if frame.ndim == 2:
            frame = np.stack((frame, frame, frame), axis=-1)
        if frame.ndim != 3:
            raise ValueError("Frame must have 2 or 3 dimensions.")
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.shape[2] != 3:
            raise ValueError("Frame must have 3 channels after normalization.")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        target_width, target_height = self._config.frame_size
        pil_image = Image.fromarray(frame, mode="RGB")
        resized = pil_image.resize(
            (target_width, target_height), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)
