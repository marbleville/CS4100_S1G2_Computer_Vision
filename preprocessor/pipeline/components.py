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


def coalesce_components(components: list[ComponentStats]) -> list[ComponentStats]:
    """Merge overlapping (or edge-touching) component bboxes.

    Returns a new list of aggregated components where each output component
    represents one connected group in the bbox-overlap graph.
    """
    if not components:
        return []

    n = len(components)
    visited = [False] * n
    merged: list[ComponentStats] = []

    for start_idx in range(n):
        if visited[start_idx]:
            continue
        stack = [start_idx]
        group_indices: list[int] = []
        visited[start_idx] = True

        while stack:
            idx = stack.pop()
            group_indices.append(idx)
            a = components[idx]
            for j in range(n):
                if visited[j]:
                    continue
                b = components[j]
                if _bbox_overlaps_or_touches(a.bbox_xyxy, b.bbox_xyxy):
                    visited[j] = True
                    stack.append(j)

        group = [components[i] for i in group_indices]
        merged.append(_merge_group(group))

    return merged


def _bbox_overlaps_or_touches(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x_disjoint = ax1 < bx0 - 1 or bx1 < ax0 - 1
    y_disjoint = ay1 < by0 - 1 or by1 < ay0 - 1
    return not (x_disjoint or y_disjoint)


def _merge_group(group: list[ComponentStats]) -> ComponentStats:
    if len(group) == 1:
        return group[0]

    min_x = min(c.bbox_xyxy[0] for c in group)
    min_y = min(c.bbox_xyxy[1] for c in group)
    max_x = max(c.bbox_xyxy[2] for c in group)
    max_y = max(c.bbox_xyxy[3] for c in group)

    area = sum(c.area for c in group)
    centroid_x = sum(c.centroid_xy[0] * c.area for c in group) / max(area, 1)
    centroid_y = sum(c.centroid_xy[1] * c.area for c in group) / max(area, 1)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    bbox_area = float(width * height)
    aspect_ratio = width / max(float(height), 1.0)
    fill_ratio = area / max(bbox_area, 1.0)
    touches_border = any(c.touches_border for c in group)
    label = min(c.label for c in group)

    return ComponentStats(
        label=label,
        area=area,
        bbox_xyxy=(min_x, min_y, max_x, max_y),
        centroid_xy=(float(centroid_x), float(centroid_y)),
        aspect_ratio=float(aspect_ratio),
        fill_ratio=float(fill_ratio),
        touches_border=touches_border,
    )
