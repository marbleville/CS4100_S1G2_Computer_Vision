from __future__ import annotations

import numpy as np

from preprocessor.io.types import FramePacket
from preprocessor.pipeline.types import PipelineFrameResult
from preprocessor.visualization import render_pipeline_result


def _sample_packet_and_result() -> tuple[FramePacket, PipelineFrameResult]:
    frame = np.full((10, 12, 3), 120, dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:8, 3:9] = True
    packet = FramePacket(
        frame_index=1,
        timestamp_ms=33,
        frame_rgb=frame,
        source_id="sample",
    )
    result = PipelineFrameResult(
        timestamp_ms=33,
        frame_index=1,
        mask=mask,
        selected_label=1,
        selected_bbox_xyxy_px=(3, 2, 8, 7),
        selected_centroid_xy_px=(5.5, 4.5),
        selected_area_px=int(mask.sum()),
        candidate_count=1,
        quality_score=0.8,
    )
    return packet, result


def test_render_pipeline_result_masks_background_and_draws_bbox() -> None:
    packet, result = _sample_packet_and_result()
    vis = render_pipeline_result(packet, result, bbox_color=(255, 0, 0), bbox_width=1)

    assert vis.shape == packet.frame_rgb.shape
    assert vis.dtype == np.uint8
    assert np.all(vis[0, 0] == np.array([0, 0, 0], dtype=np.uint8))
    assert np.all(vis[2, 3] == np.array([255, 0, 0], dtype=np.uint8))


def test_render_pipeline_result_writes_file(tmp_path) -> None:
    packet, result = _sample_packet_and_result()
    out_path = tmp_path / "vis.png"
    vis = render_pipeline_result(packet, result, output_path=out_path)

    assert out_path.exists()
    assert vis.shape == (10, 12, 3)
