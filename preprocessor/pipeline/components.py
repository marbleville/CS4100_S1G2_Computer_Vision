"""Connected components and component statistics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ComponentStats:
    """Per-component geometric statistics."""

    label: int
    area: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    aspect_ratio: float
    fill_ratio: float
    touches_border: bool


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[ComponentStats]]:
    """Label connected components and compute stats (4-neighborhood)."""
    binary = mask.astype(bool)
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    stats: list[ComponentStats] = []
    current_label = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            current_label += 1
            q = deque([(y, x)])
            labels[y, x] = current_label

            area = 0
            sum_x = 0.0
            sum_y = 0.0
            min_x, min_y = x, y
            max_x, max_y = x, y
            touches_border = False

            while q:
                cy, cx = q.popleft()
                area += 1
                sum_x += cx
                sum_y += cy
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
                if cy == 0 or cx == 0 or cy == h - 1 or cx == w - 1:
                    touches_border = True
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current_label
                        q.append((ny, nx))

            width = max_x - min_x + 1
            height = max_y - min_y + 1
            bbox_area = float(width * height)
            centroid = (sum_x / area, sum_y / area)
            aspect_ratio = width / max(float(height), 1.0)
            fill_ratio = area / max(bbox_area, 1.0)
            stats.append(
                ComponentStats(
                    label=current_label,
                    area=area,
                    bbox_xyxy=(min_x, min_y, max_x, max_y),
                    centroid_xy=centroid,
                    aspect_ratio=aspect_ratio,
                    fill_ratio=fill_ratio,
                    touches_border=touches_border,
                )
            )

    return labels, stats
