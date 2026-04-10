from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from preprocessor.pipeline.components import ComponentStats, coalesce_components, connected_components


def test_connected_components_extracts_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_cv2(monkeypatch)
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
    first = min(stats, key=lambda s: s.bbox_xyxy[0])
    assert first.centroid_xy == (1.5, 1.5)
    assert first.aspect_ratio == 1.0
    assert first.fill_ratio == 1.0
    assert first.touches_border is False


def test_connected_components_uses_four_connectivity_and_detects_borders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cv2(monkeypatch)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True

    _, stats = connected_components(mask)

    assert len(stats) == 2
    border_component = min(stats, key=lambda s: s.bbox_xyxy)
    assert border_component.bbox_xyxy == (0, 0, 0, 0)
    assert border_component.touches_border is True


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


def _install_fake_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    import preprocessor.pipeline.components as components_module

    monkeypatch.setattr(
        components_module,
        "cv2",
        SimpleNamespace(
            CV_32S=np.int32,
            connectedComponentsWithStats=_fake_connected_components_with_stats,
        ),
    )


def _fake_connected_components_with_stats(
    binary: np.ndarray,
    connectivity: int,
    ltype: object,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    assert connectivity == 4

    foreground = binary.astype(bool)
    h, w = foreground.shape
    labels = np.zeros((h, w), dtype=np.int32)
    stats: list[list[int]] = [[0, 0, 0, 0, int((~foreground).sum())]]
    centroids: list[list[float]] = [[0.0, 0.0]]
    current_label = 0

    for y in range(h):
        for x in range(w):
            if not foreground[y, x] or labels[y, x] != 0:
                continue

            current_label += 1
            stack = [(y, x)]
            labels[y, x] = current_label
            area = 0
            sum_x = 0.0
            sum_y = 0.0
            min_x = max_x = x
            min_y = max_y = y

            while stack:
                cy, cx = stack.pop()
                area += 1
                sum_x += cx
                sum_y += cy
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and foreground[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current_label
                        stack.append((ny, nx))

            stats.append([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area])
            centroids.append([sum_x / area, sum_y / area])

    return (
        current_label + 1,
        labels,
        np.asarray(stats, dtype=np.int32),
        np.asarray(centroids, dtype=np.float64),
    )
