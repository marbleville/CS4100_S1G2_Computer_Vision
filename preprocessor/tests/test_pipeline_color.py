from __future__ import annotations

import numpy as np

from preprocessor.pipeline.color import (
    fused_skin_confidence,
    rgb_to_grayscale,
    rgb_to_hsv,
    rgb_to_ycbcr,
)


def test_color_transforms_shapes_and_ranges() -> None:
    frame = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    gray = rgb_to_grayscale(frame)
    hsv = rgb_to_hsv(frame)
    ycbcr = rgb_to_ycbcr(frame)

    assert gray.shape == (2, 2)
    assert hsv.shape == (2, 2, 3)
    assert ycbcr.shape == (2, 2, 3)
    assert float(gray.min()) >= 0.0 and float(gray.max()) <= 1.0
    assert float(hsv.min()) >= 0.0 and float(hsv.max()) <= 1.0


def test_fused_skin_confidence_stays_bounded() -> None:
    frame = np.full((4, 4, 3), [210, 160, 130], dtype=np.uint8)
    hsv = rgb_to_hsv(frame)
    ycbcr = rgb_to_ycbcr(frame)
    conf = fused_skin_confidence(hsv, ycbcr)
    assert conf.shape == (4, 4)
    assert float(conf.min()) >= 0.0
    assert float(conf.max()) <= 1.0
