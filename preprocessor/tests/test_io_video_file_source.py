from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.factory import build_frame_source
from preprocessor.io.video_file_source import VideoFileFrameSource


def _config(video_path: str | None = "/tmp/sample.mp4") -> PreprocessorConfig:
    return PreprocessorConfig(
        input_mode="local_video",
        video_path=video_path,
        frame_size=(32, 24),
    )


def _make_fake_frames(count: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for idx in range(count):
        frame = np.full((10, 12, 3), idx, dtype=np.uint8)
        frames.append(frame)
    return frames


def test_factory_with_video_path_builds_video_file_source() -> None:
    source = build_frame_source(_config("/tmp/example.mp4"))
    assert isinstance(source, VideoFileFrameSource)


def test_factory_without_video_path_raises_value_error() -> None:
    with pytest.raises(ValueError, match="`video_path` is required"):
        build_frame_source(_config(video_path=None))


def test_packet_shape_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _make_fake_frames(1)

    def fake_immeta(path: str) -> dict[str, float]:
        assert path.endswith(".mp4")
        return {"fps": 10.0}

    def fake_imiter(path: str) -> Iterator[np.ndarray]:
        assert path.endswith(".mp4")
        return iter(frames)

    import preprocessor.io.video_file_source as vfs

    monkeypatch.setattr(vfs.iio, "immeta", fake_immeta)
    monkeypatch.setattr(vfs.iio, "imiter", fake_imiter)

    source = VideoFileFrameSource(_config())
    source.open()
    packet = source.read()

    assert packet is not None
    assert packet.frame_index == 0
    assert packet.timestamp_ms == 0
    assert packet.frame_rgb.dtype == np.uint8
    assert packet.frame_rgb.shape == (24, 32, 3)


def test_timestamp_sequence_and_eos(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _make_fake_frames(3)

    def fake_immeta(path: str) -> dict[str, float]:
        return {"fps": 20.0}

    def fake_imiter(path: str) -> Iterator[np.ndarray]:
        return iter(frames)

    import preprocessor.io.video_file_source as vfs

    monkeypatch.setattr(vfs.iio, "immeta", fake_immeta)
    monkeypatch.setattr(vfs.iio, "imiter", fake_imiter)

    source = VideoFileFrameSource(_config())
    packets = []
    while True:
        packet = source.read()
        if packet is None:
            break
        packets.append(packet)

    timestamps = [packet.timestamp_ms for packet in packets]
    assert timestamps == [0, 50, 100]
    assert timestamps == sorted(timestamps)
    assert source.read() is None


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _make_fake_frames(1)

    def fake_immeta(path: str) -> dict[str, float]:
        return {"fps": 5.0}

    def fake_imiter(path: str) -> Iterator[np.ndarray]:
        return iter(frames)

    import preprocessor.io.video_file_source as vfs

    monkeypatch.setattr(vfs.iio, "immeta", fake_immeta)
    monkeypatch.setattr(vfs.iio, "imiter", fake_imiter)

    source = VideoFileFrameSource(_config())
    source.open()
    source.close()
    source.close()
    source.open()
    assert source.read() is not None


def test_deterministic_frame_count_and_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _make_fake_frames(4)

    def fake_immeta(path: str) -> dict[str, float]:
        return {"fps": 8.0}

    def fake_imiter(path: str) -> Iterator[np.ndarray]:
        # Return fresh iterator for each reader instance.
        return iter([frame.copy() for frame in frames])

    import preprocessor.io.video_file_source as vfs

    monkeypatch.setattr(vfs.iio, "immeta", fake_immeta)
    monkeypatch.setattr(vfs.iio, "imiter", fake_imiter)

    first = VideoFileFrameSource(_config())
    second = VideoFileFrameSource(_config())

    def consume(source: VideoFileFrameSource) -> list[int]:
        values: list[int] = []
        while True:
            packet = source.read()
            if packet is None:
                break
            values.append(packet.timestamp_ms)
        return values

    first_ts = consume(first)
    second_ts = consume(second)

    assert len(first_ts) == 4
    assert first_ts == second_ts == [0, 125, 250, 375]


def test_invalid_or_missing_fps_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_imiter(path: str) -> Iterator[np.ndarray]:
        return iter(_make_fake_frames(1))

    import preprocessor.io.video_file_source as vfs

    monkeypatch.setattr(vfs.iio, "imiter", fake_imiter)

    for bad_fps in (None, 0, -1):
        def fake_immeta(path: str, value=bad_fps) -> dict[str, float | None]:
            return {"fps": value}

        monkeypatch.setattr(vfs.iio, "immeta", fake_immeta)
        source = VideoFileFrameSource(_config())
        with pytest.raises(ValueError):
            source.open()
