from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import preprocessor
import pytest

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.types import FramePacket
from preprocessor.types import HandCandidateFrame, HandFrameResult, ResultStatus


def test_preprocessor_config_instantiation() -> None:
    config = PreprocessorConfig(input_mode="webcam")
    assert config.input_mode == "webcam"
    assert config.camera_device is None
    assert config.frame_size == (640, 480)
    assert config.candidate_frame_size_px == 128
    assert config.candidate_buffer_size == 32


def test_preprocessor_config_rejects_negative_camera_device() -> None:
    with pytest.raises(ValueError, match="camera_device"):
        PreprocessorConfig(input_mode="webcam", camera_device=-1)


def test_result_status_enum_coverage() -> None:
    expected = {"ok", "no_hand", "error"}
    actual = {status.value for status in ResultStatus}
    assert actual == expected


def test_hand_result_supports_no_hand_shape() -> None:
    result = HandFrameResult(
        status=ResultStatus.NO_HAND,
        timestamp_ms=123456,
        candidates=[],
    )
    assert result.status == ResultStatus.NO_HAND
    assert result.candidates == []
    assert result.error_message is None


def test_preprocessor_batch_api_uses_source_and_pipeline(
    monkeypatch,
) -> None:
    packet = FramePacket(
        frame_index=0,
        timestamp_ms=25,
        frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        source_id="webcam:0",
    )
    expected = HandFrameResult(
        status=ResultStatus.OK,
        timestamp_ms=25,
        candidates=[],
    )
    fake_source = _FakeSource([packet])
    fake_pipeline = _FakePipeline()
    fake_processor = ModuleType("preprocessor.pipeline.processor")
    fake_processor.PreprocessingPipeline = lambda config: fake_pipeline
    fake_processor.pipeline_result_to_hand_result = lambda result: expected

    import preprocessor.io.factory as factory

    monkeypatch.setattr(factory, "build_frame_source", lambda config: fake_source)
    monkeypatch.setitem(sys.modules, "preprocessor.pipeline.processor", fake_processor)

    processor = preprocessor.Preprocessor(PreprocessorConfig(input_mode="webcam"))
    result = processor.get_current_hand_candidates()

    assert result is expected
    assert fake_source.read_calls == 1
    assert fake_pipeline.processed_packets == [packet]


def test_preprocessor_next_drains_pipeline_queue(monkeypatch) -> None:
    packet = FramePacket(
        frame_index=3,
        timestamp_ms=100,
        frame_rgb=np.zeros((6, 6, 3), dtype=np.uint8),
        source_id="webcam:0",
    )
    expected_candidate = HandCandidateFrame(
        frame_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp_ms=100,
        source_frame_index=3,
        source_id="webcam:0",
        candidate_index=0,
        bbox_xyxy_px=(0, 0, 5, 5),
    )
    fake_source = _FakeSource([packet, None])
    fake_pipeline = _FakePipeline(next_candidates=[expected_candidate])
    fake_processor = ModuleType("preprocessor.pipeline.processor")
    fake_processor.PreprocessingPipeline = lambda config: fake_pipeline
    fake_processor.pipeline_result_to_hand_result = lambda result: result

    import preprocessor.io.factory as factory

    monkeypatch.setattr(factory, "build_frame_source", lambda config: fake_source)
    monkeypatch.setitem(sys.modules, "preprocessor.pipeline.processor", fake_processor)

    processor = preprocessor.Preprocessor(PreprocessorConfig(input_mode="webcam"))

    assert processor.next() is expected_candidate
    assert processor.next() is None
    assert fake_pipeline.processed_packets == [packet]


def test_readme_and_public_exports_stay_aligned() -> None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    documented_names = [
        "PreprocessorConfig",
        "HandCandidateFrame",
        "HandFrameResult",
        "ResultStatus",
        "init_preprocessor",
        "Preprocessor",
    ]

    for name in documented_names:
        assert name in readme_text
        assert hasattr(preprocessor, name)

    assert "camera_device" in readme_text
    assert "first readable camera" in readme_text


class _FakeSource:
    def __init__(self, packets: list[FramePacket | None]) -> None:
        self._packets = list(packets)
        self.read_calls = 0

    def open(self) -> None:
        return None

    def read(self) -> FramePacket | None:
        self.read_calls += 1
        if not self._packets:
            return None
        return self._packets.pop(0)

    def close(self) -> None:
        return None


class _FakePipeline:
    def __init__(self, next_candidates: list[HandCandidateFrame] | None = None) -> None:
        self._queue = list(next_candidates or [])
        self.processed_packets: list[FramePacket] = []

    def process(self, packet: FramePacket) -> object:
        self.processed_packets.append(packet)
        return {"frame_index": packet.frame_index}

    def _pop_next_candidate(self) -> HandCandidateFrame | None:
        if not self._queue:
            return None
        return self._queue.pop(0)
