from __future__ import annotations

import numpy as np

from preprocessor.pipeline.components import ComponentStats, coalesce_components, connected_components


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


def test_coalesce_components_merges_overlapping_groups() -> None:
    comps = [
        ComponentStats(
            label=1,
            area=10,
            bbox_xyxy=(2, 2, 5, 6),
            centroid_xy=(3.0, 4.0),
            aspect_ratio=0.8,
            fill_ratio=0.5,
            touches_border=False,
        ),
        ComponentStats(
            label=2,
            area=12,
            bbox_xyxy=(5, 4, 8, 8),
            centroid_xy=(6.0, 6.0),
            aspect_ratio=0.8,
            fill_ratio=0.5,
            touches_border=True,
        ),
        ComponentStats(
            label=3,
            area=6,
            bbox_xyxy=(20, 20, 22, 22),
            centroid_xy=(21.0, 21.0),
            aspect_ratio=1.0,
            fill_ratio=0.66,
            touches_border=False,
        ),
    ]

    merged = coalesce_components(comps)
    assert len(merged) == 2

    merged.sort(key=lambda c: c.bbox_xyxy[0])
    first = merged[0]
    assert first.bbox_xyxy == (2, 2, 8, 8)
    assert first.area == 22
    assert first.touches_border is True

    second = merged[1]
    assert second.bbox_xyxy == (20, 20, 22, 22)
    assert second.area == 6
