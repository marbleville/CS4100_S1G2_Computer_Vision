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
    """Pull-based frame source backed by an explicit or auto-detected webcam."""

    _AUTO_DETECT_MAX_DEVICE_INDEX = 5

    def __init__(self, config: PreprocessorConfig) -> None:
        self._config = config
        self._capture: object | None = None
        self._prefetched_frame: np.ndarray | None = None
        self._resolved_device_index: int | None = None
        self._frame_index = 0
        self._opened_at: float | None = None
        self._is_open = False

    def open(self) -> None:
        if self._is_open:
            return
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for webcam input but is not installed."
            )

        configured_device = self._config.camera_device
        if configured_device is None:
            capture, raw_frame, device_index = self._auto_detect_device()
        else:
            capture, raw_frame = self._open_explicit_device(configured_device)
            device_index = configured_device

        self._capture = capture
        self._prefetched_frame = raw_frame
        self._resolved_device_index = device_index
        self._frame_index = 0
        self._opened_at = time.monotonic()
        self._is_open = True

    def read(self) -> FramePacket:
        if not self._is_open:
            self.open()

        assert self._capture is not None
        assert self._opened_at is not None
        assert self._resolved_device_index is not None

        raw_frame = self._prefetched_frame
        self._prefetched_frame = None
        if raw_frame is None:
            ok, raw_frame = self._capture.read()
            if not ok or raw_frame is None:
                raise RuntimeError(
                    f"Failed to read frame from webcam device {self._resolved_device_index}."
                )

        frame_rgb = self._normalize_frame(raw_frame)
        timestamp_ms = int(round((time.monotonic() - self._opened_at) * 1000))
        packet = FramePacket(
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
            frame_rgb=frame_rgb,
            source_id=f"webcam:{self._resolved_device_index}",
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._prefetched_frame = None
        self._resolved_device_index = None
        self._frame_index = 0
        self._opened_at = None
        self._is_open = False
        if capture is not None:
            capture.release()

    def _open_explicit_device(self, device_index: int) -> tuple[object, np.ndarray]:
        assert cv2 is not None

        capture = cv2.VideoCapture(device_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open webcam device {device_index}.")

        ok, raw_frame = capture.read()
        if not ok or raw_frame is None:
            capture.release()
            raise RuntimeError(f"Failed to read frame from webcam device {device_index}.")

        return capture, raw_frame

    def _auto_detect_device(self) -> tuple[object, np.ndarray, int]:
        assert cv2 is not None

        for device_index in range(self._AUTO_DETECT_MAX_DEVICE_INDEX + 1):
            capture = cv2.VideoCapture(device_index)
            if not capture.isOpened():
                capture.release()
                continue

            ok, raw_frame = capture.read()
            if ok and raw_frame is not None:
                return capture, raw_frame, device_index

            capture.release()

        raise RuntimeError(
            "No readable webcam device found in indices "
            f"0..{self._AUTO_DETECT_MAX_DEVICE_INDEX}."
        )

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
