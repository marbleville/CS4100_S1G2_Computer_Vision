"""Component filtering and temporal continuity-based selection."""

from __future__ import annotations

from dataclasses import dataclass

from preprocessor.pipeline.components import ComponentStats


@dataclass(slots=True)
class SelectionConfig:
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.60
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 2.8
    min_fill_ratio: float = 0.12
    border_penalty: float = 0.1
    continuity_weight: float = 0.35


def select_best_component(
    components: list[ComponentStats],
    frame_width: int,
    frame_height: int,
    prev_centroid: tuple[float, float] | None,
    cfg: SelectionConfig,
) -> tuple[ComponentStats | None, dict[str, float | int]]:
    """Select best plausible hand component with continuity preference."""
    frame_area = frame_width * frame_height
    cx_frame = frame_width / 2.0
    cy_frame = frame_height / 2.0

    candidates: list[tuple[float, ComponentStats]] = []
    for comp in components:
        area_ratio = comp.area / max(float(frame_area), 1.0)
        if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
            continue
        if comp.aspect_ratio < cfg.min_aspect_ratio or comp.aspect_ratio > cfg.max_aspect_ratio:
            continue
        if comp.fill_ratio < cfg.min_fill_ratio:
            continue

        # Weighted heuristic score.
        area_score = min(1.0, area_ratio / max(cfg.min_area_ratio * 2.0, 1e-6))
        dx = (comp.centroid_xy[0] - cx_frame) / max(cx_frame, 1.0)
        dy = (comp.centroid_xy[1] - cy_frame) / max(cy_frame, 1.0)
        centrality = max(0.0, 1.0 - (dx * dx + dy * dy) ** 0.5)
        continuity = 0.5
        if prev_centroid is not None:
            px, py = prev_centroid
            ddx = (comp.centroid_xy[0] - px) / max(frame_width, 1.0)
            ddy = (comp.centroid_xy[1] - py) / max(frame_height, 1.0)
            continuity = max(0.0, 1.0 - (ddx * ddx + ddy * ddy) ** 0.5 * 2.0)

        border_score = (1.0 - cfg.border_penalty) if comp.touches_border else 1.0
        score = (
            0.35 * area_score
            + 0.30 * centrality
            + cfg.continuity_weight * continuity
        ) * border_score
        candidates.append((score, comp))

    if not candidates:
        return None, {"candidate_count": 0, "selected_score": 0.0}

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_comp = candidates[0]
    return best_comp, {"candidate_count": len(candidates), "selected_score": float(best_score)}
