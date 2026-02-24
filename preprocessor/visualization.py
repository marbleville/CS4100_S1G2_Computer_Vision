"""Utilities for visualizing preprocessing outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from preprocessor.io.types import FramePacket
from preprocessor.pipeline.types import PipelineFrameResult


def render_pipeline_result(
    packet: FramePacket,
    result: PipelineFrameResult,
    output_path: str | Path | None = None,
    bbox_color: tuple[int, int, int] = (255, 0, 0),
    bbox_width: int = 2,
) -> np.ndarray:
    """Render a visualization frame from packet + pipeline result.

    Behavior:
    - Background pixels (where mask is False) are set to black.
    - Selected bounding box is drawn if available.
    - If output_path is provided, image is written to disk.
    """
    frame = np.asarray(packet.frame_rgb, dtype=np.uint8).copy()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("FramePacket.frame_rgb must have shape (H, W, 3).")

    mask = np.asarray(result.mask).astype(bool)
    if mask.shape != frame.shape[:2]:
        raise ValueError(
            "PipelineFrameResult.mask shape must match frame height/width.")

    vis = frame.copy()
    vis[~mask] = 0

    for component in result.candidates:
        _draw_bbox(
            image=vis,
            bbox=component.bbox_xyxy,
            color=(0, 0, 255),
            width=max(1, int(bbox_width)),
        )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis, mode="RGB").save(out)

    return vis


def _draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(0, min(w - 1, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h - 1, int(y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    for t in range(width):
        yt = min(h - 1, y0 + t)
        yb = max(0, y1 - t)
        xl = min(w - 1, x0 + t)
        xr = max(0, x1 - t)

        image[yt, x0: x1 + 1] = color
        image[yb, x0: x1 + 1] = color
        image[y0: y1 + 1, xl] = color
        image[y0: y1 + 1, xr] = color
