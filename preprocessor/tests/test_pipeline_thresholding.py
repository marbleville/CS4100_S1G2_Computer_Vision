from __future__ import annotations

import numpy as np

from preprocessor.pipeline.thresholding import global_percentile_threshold, local_tile_threshold


def test_global_percentile_threshold_is_deterministic() -> None:
    score = np.arange(100, dtype=np.float32).reshape(10, 10) / 100.0
    mask1, t1 = global_percentile_threshold(score, percentile=90.0)
    mask2, t2 = global_percentile_threshold(score, percentile=90.0)
    assert t1 == t2
    assert np.array_equal(mask1, mask2)


def test_local_tile_threshold_returns_binary_mask() -> None:
    score = np.linspace(0.0, 1.0, num=64, dtype=np.float32).reshape(8, 8)
    global_mask, t = global_percentile_threshold(score, percentile=80.0)
    local_mask, _ = local_tile_threshold(score, global_threshold=t, percentile=80.0, tiles_x=4, tiles_y=4)
    assert local_mask.shape == global_mask.shape
    assert local_mask.dtype == bool
