from __future__ import annotations

from preprocessor.pipeline.components import ComponentStats
from preprocessor.pipeline.selection import SelectionConfig, select_best_component


def _component(label: int, cx: float, cy: float, area: int = 600) -> ComponentStats:
    return ComponentStats(
        label=label,
        area=area,
        bbox_xyxy=(10, 10, 30, 40),
        centroid_xy=(cx, cy),
        aspect_ratio=0.7,
        fill_ratio=0.5,
        touches_border=False,
    )


def test_selection_prefers_temporal_continuity() -> None:
    comps = [
        _component(1, cx=50.0, cy=50.0),
        _component(2, cx=90.0, cy=90.0),
    ]
    selected, debug = select_best_component(
        components=comps,
        frame_width=120,
        frame_height=120,
        prev_centroid=(52.0, 52.0),
        cfg=SelectionConfig(),
    )
    assert selected is not None
    assert selected.label == 1
    assert int(debug["candidate_count"]) == 2
