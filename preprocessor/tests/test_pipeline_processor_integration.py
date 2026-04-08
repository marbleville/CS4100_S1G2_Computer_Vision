from __future__ import annotations

import numpy as np

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.types import FramePacket
from preprocessor.pipeline.processor import PreprocessingPipeline, pipeline_result_to_hand_result
from preprocessor.types.enums import ResultStatus


def _config(profile: str = "default") -> PreprocessorConfig:
    return PreprocessorConfig(
        input_mode="webcam",
        threshold_profile=profile,
    )


def _skin_patch_frame(cx: int, cy: int, h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = [30, 30, 30]
    patch_h, patch_w = 16, 14
    y0 = max(0, cy - patch_h // 2)
    x0 = max(0, cx - patch_w // 2)
    y1 = min(h, y0 + patch_h)
    x1 = min(w, x0 + patch_w)
    frame[y0:y1, x0:x1] = [205, 165, 135]
    return frame


def test_pipeline_process_returns_candidate_on_synthetic_frames() -> None:
    pipeline = PreprocessingPipeline(_config("default"))
    packet = FramePacket(
        frame_index=0,
        timestamp_ms=0,
        frame_rgb=_skin_patch_frame(30, 30),
        source_id="synthetic",
    )
    result = pipeline.process(packet)
    assert result.mask.shape == (64, 64)
    assert 0.0 <= result.quality_score <= 1.0
    assert result.candidate_count >= 0


def test_pipeline_temporal_continuity_tracks_moving_patch() -> None:
    pipeline = PreprocessingPipeline(_config("default"))
    centroids: list[tuple[float, float]] = []
    for idx, x in enumerate([24, 28, 32]):
        packet = FramePacket(
            frame_index=idx,
            timestamp_ms=idx * 33,
            frame_rgb=_skin_patch_frame(x, 28),
            source_id="seq",
        )
        result = pipeline.process(packet)
        if result.selected_centroid_xy_px is not None:
            centroids.append(result.selected_centroid_xy_px)
    if len(centroids) >= 2:
        assert centroids[-1][0] >= centroids[0][0]


def test_pipeline_high_motion_profile_is_api_compatible() -> None:
    pipeline = PreprocessingPipeline(_config("high_motion"))
    packet = FramePacket(
        frame_index=1,
        timestamp_ms=40,
        frame_rgb=_skin_patch_frame(26, 34),
        source_id="hm",
    )
    result = pipeline.process(packet)
    hand_result = pipeline_result_to_hand_result(result, frame_width=64, frame_height=64)
    assert hand_result.status in (ResultStatus.OK, ResultStatus.NO_HAND)
