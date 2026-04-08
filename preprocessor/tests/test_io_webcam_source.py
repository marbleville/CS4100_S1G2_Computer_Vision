from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.factory import build_frame_source
from preprocessor.io.webcam_source import WebcamFrameSource


def _config(video_path: str | None = None) -> PreprocessorConfig:
    return PreprocessorConfig(
        input_mode="webcam",
        video_path=video_path,
        frame_size=(32, 24),
    )


def test_factory_with_webcam_mode_builds_webcam_source() -> None:
    source = build_frame_source(_config("/tmp/ignored.mp4"))
    assert isinstance(source, WebcamFrameSource)


def test_read_lazily_opens_and_converts_bgr_to_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_bgr = np.zeros((6, 8, 3), dtype=np.uint8)
    frame_bgr[:, :] = [10, 20, 30]
    capture = _FakeCapture([frame_bgr])

    _install_fake_cv2(monkeypatch, [capture])
    _install_monotonic_clock(monkeypatch, [5.0, 5.012])

    source = WebcamFrameSource(_config())
    packet = source.read()

    assert packet.frame_index == 0
    assert packet.timestamp_ms == 12
    assert packet.source_id == "webcam:0"
    assert packet.frame_rgb.shape == (24, 32, 3)
    assert packet.frame_rgb[0, 0].tolist() == [30, 20, 10]
    assert capture.device_index == 0


def test_frame_indexes_and_timestamps_are_monotonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture(
        [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ]
    )

    _install_fake_cv2(monkeypatch, [capture])
    _install_monotonic_clock(monkeypatch, [10.0, 10.010, 10.025])

    source = WebcamFrameSource(_config())
    first = source.read()
    second = source.read()

    assert [first.frame_index, second.frame_index] == [0, 1]
    assert [first.timestamp_ms, second.timestamp_ms] == [10, 25]


def test_open_and_close_are_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    first_capture = _FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)])
    second_capture = _FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)])

    _install_fake_cv2(monkeypatch, [first_capture, second_capture])
    _install_monotonic_clock(monkeypatch, [1.0, 1.010, 2.0, 2.015])

    source = WebcamFrameSource(_config())
    source.open()
    source.open()
    source.close()
    source.close()

    packet = source.read()

    assert first_capture.released is True
    assert second_capture.device_index == 0
    assert packet.frame_index == 0


def test_open_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture([], is_open=False)
    _install_fake_cv2(monkeypatch, [capture])

    source = WebcamFrameSource(_config())
    with pytest.raises(RuntimeError, match="Failed to open webcam device 0"):
        source.open()

    assert capture.released is True


def test_read_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture([])
    _install_fake_cv2(monkeypatch, [capture])
    _install_monotonic_clock(monkeypatch, [3.0])

    source = WebcamFrameSource(_config())
    with pytest.raises(RuntimeError, match="Failed to read frame from webcam device 0"):
        source.read()


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        is_open: bool = True,
    ) -> None:
        self.device_index: int | None = None
        self._frames = list(frames)
        self._is_open = is_open
        self.released = False

    def isOpened(self) -> bool:
        return self._is_open

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


def _install_fake_cv2(
    monkeypatch: pytest.MonkeyPatch,
    captures: list[_FakeCapture],
) -> None:
    import preprocessor.io.webcam_source as webcam_source

    remaining = list(captures)

    def fake_video_capture(device_index: int) -> _FakeCapture:
        capture = remaining.pop(0)
        capture.device_index = device_index
        return capture

    monkeypatch.setattr(
        webcam_source,
        "cv2",
        SimpleNamespace(VideoCapture=fake_video_capture),
    )


def _install_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch,
    values: list[float],
) -> None:
    import preprocessor.io.webcam_source as webcam_source

    timestamps = iter(values)
    monkeypatch.setattr(webcam_source.time, "monotonic", lambda: next(timestamps))
