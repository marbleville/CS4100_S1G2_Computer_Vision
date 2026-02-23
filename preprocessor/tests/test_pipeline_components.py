from __future__ import annotations

import numpy as np

from preprocessor.pipeline.components import connected_components


def test_connected_components_extracts_stats() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True

    labels, stats = connected_components(mask)
    assert labels.shape == mask.shape
    assert len(stats) == 2

    areas = sorted([s.area for s in stats])
    assert areas == [4, 4]
    bboxes = sorted([s.bbox_xyxy for s in stats])
    assert bboxes[0] == (1, 1, 2, 2)
