"""Thresholding utilities."""

from __future__ import annotations

import numpy as np


def global_percentile_threshold(score_map: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    """Threshold using a global percentile over score map."""
    threshold = float(np.percentile(score_map, percentile))
    mask = score_map >= threshold
    return mask.astype(bool), threshold


def local_tile_threshold(
    score_map: np.ndarray,
    global_threshold: float,
    percentile: float = 85.0,
    tiles_x: int = 8,
    tiles_y: int = 6,
    blend: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Apply local tile thresholds and blend with global threshold mask."""
    h, w = score_map.shape
    tile_h = max(1, h // tiles_y)
    tile_w = max(1, w // tiles_x)
    local_mask = np.zeros((h, w), dtype=bool)
    for y0 in range(0, h, tile_h):
        for x0 in range(0, w, tile_w):
            y1 = min(h, y0 + tile_h)
            x1 = min(w, x0 + tile_w)
            tile = score_map[y0:y1, x0:x1]
            tile_threshold = float(np.percentile(tile, percentile))
            effective = (1.0 - blend) * tile_threshold + blend * global_threshold
            local_mask[y0:y1, x0:x1] = tile >= effective
    return local_mask, global_threshold
