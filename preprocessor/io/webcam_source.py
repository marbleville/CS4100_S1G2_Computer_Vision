"""Live webcam frame source."""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.types import FramePacket

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without OpenCV
    cv2 = None


class WebcamFrameSource:
    """Pull-based frame source backed by the default webcam device."""

    _DEVICE_INDEX = 0

    def __init__(self, config: PreprocessorConfig) -> None:
        self._config = config
        self._capture: object | None = None
        self._frame_index = 0
        self._opened_at: float | None = None
        self._is_open = False
        self._source_id = f"webcam:{self._DEVICE_INDEX}"

    def open(self) -> None:
        if self._is_open:
            return
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for webcam input but is not installed."
            )

        capture = cv2.VideoCapture(self._DEVICE_INDEX)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Failed to open webcam device {self._DEVICE_INDEX}."
            )

        self._capture = capture
        self._frame_index = 0
        self._opened_at = time.monotonic()
        self._is_open = True

    def read(self) -> FramePacket:
        if not self._is_open:
            self.open()

        assert self._capture is not None
        assert self._opened_at is not None

        ok, raw_frame = self._capture.read()
        if not ok or raw_frame is None:
            raise RuntimeError(
                f"Failed to read frame from webcam device {self._DEVICE_INDEX}."
            )

        frame_rgb = self._normalize_frame(raw_frame)
        timestamp_ms = int(round((time.monotonic() - self._opened_at) * 1000))
        packet = FramePacket(
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
            frame_rgb=frame_rgb,
            source_id=self._source_id,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._frame_index = 0
        self._opened_at = None
        self._is_open = False
        if capture is not None:
            capture.release()

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

        frame_rgb = frame[:, :, ::-1].copy()
        target_width, target_height = self._config.frame_size
        pil_image = Image.fromarray(frame_rgb, mode="RGB")
        resized = pil_image.resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.uint8)
