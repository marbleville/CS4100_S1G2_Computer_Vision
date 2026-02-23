"""Stateful preprocessing pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.types import FramePacket
from preprocessor.pipeline.background import RunningBackgroundModel
from preprocessor.pipeline.color import (
    fused_skin_confidence,
    rgb_to_grayscale,
    rgb_to_hsv,
    rgb_to_ycbcr,
)
from preprocessor.pipeline.components import ComponentStats, connected_components
from preprocessor.pipeline.filtering import binary_close, binary_open, gaussian_blur
from preprocessor.pipeline.selection import SelectionConfig, select_best_component
from preprocessor.pipeline.thresholding import (
    global_percentile_threshold,
    local_tile_threshold,
)
from preprocessor.pipeline.types import PipelineFrameResult
from preprocessor.types.enums import ResultStatus
from preprocessor.types.results import HandFrameResult


@dataclass(frozen=True, slots=True)
class PipelineProfile:
    percentile: float
    local_percentile: float
    tiles_x: int
    tiles_y: int
    threshold_blend: float
    gaussian_kernel: int
    morphology_kernel: int
    min_area_px: int
    background_enabled: bool
    background_alpha: float
    background_warmup: int


_PROFILE_MAP: dict[str, PipelineProfile] = {
    "default": PipelineProfile(
        percentile=84.0,
        local_percentile=86.0,
        tiles_x=8,
        tiles_y=6,
        threshold_blend=0.35,
        gaussian_kernel=5,
        morphology_kernel=3,
        min_area_px=45,
        background_enabled=False,
        background_alpha=0.08,
        background_warmup=8,
    ),
    "low_light": PipelineProfile(
        percentile=80.0,
        local_percentile=82.0,
        tiles_x=8,
        tiles_y=6,
        threshold_blend=0.45,
        gaussian_kernel=5,
        morphology_kernel=3,
        min_area_px=35,
        background_enabled=True,
        background_alpha=0.06,
        background_warmup=10,
    ),
    "high_motion": PipelineProfile(
        percentile=86.0,
        local_percentile=88.0,
        tiles_x=6,
        tiles_y=4,
        threshold_blend=0.30,
        gaussian_kernel=3,
        morphology_kernel=3,
        min_area_px=45,
        background_enabled=False,
        background_alpha=0.10,
        background_warmup=6,
    ),
}


class PreprocessingPipeline:
    """Stateful frame preprocessing pipeline."""

    def __init__(self, config: PreprocessorConfig) -> None:
        self._config = config
        self._profile = _PROFILE_MAP.get(
            config.threshold_profile, _PROFILE_MAP["default"])
        self._selection_cfg = SelectionConfig()
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_bbox: tuple[int, int, int, int] | None = None
        self._prev_area: int | None = None
        self._bg_model = RunningBackgroundModel(
            alpha=self._profile.background_alpha,
            warmup_frames=self._profile.background_warmup,
        )

    def reset(self) -> None:
        self._prev_centroid = None
        self._prev_bbox = None
        self._prev_area = None
        self._bg_model.reset()

    def process(self, packet: FramePacket) -> PipelineFrameResult:
        frame = _validate_frame(packet.frame_rgb)
        h, w = frame.shape[:2]

        gray = rgb_to_grayscale(frame)
        hsv = rgb_to_hsv(frame)
        ycbcr = rgb_to_ycbcr(frame)

        fg_score = None
        warmup = False
        if self._profile.background_enabled:
            fg_score, warmup = self._bg_model.update_and_score(gray)
            if warmup:
                fg_score = fg_score * 0.25

        score_map = fused_skin_confidence(
            hsv, ycbcr, foreground_score=fg_score, fg_weight=0.20)
        score_map = gaussian_blur(
            score_map, kernel_size=self._profile.gaussian_kernel)

        global_mask, global_threshold = global_percentile_threshold(
            score_map, self._profile.percentile)
        local_mask, _ = local_tile_threshold(
            score_map,
            global_threshold=global_threshold,
            percentile=self._profile.local_percentile,
            tiles_x=self._profile.tiles_x,
            tiles_y=self._profile.tiles_y,
            blend=self._profile.threshold_blend,
        )
        mask = np.logical_and(global_mask, local_mask)
        mask = binary_open(mask, kernel_size=self._profile.morphology_kernel)
        mask = binary_close(mask, kernel_size=self._profile.morphology_kernel)

        labels, components = connected_components(mask)
        components = [c for c in components if c.area >=
                      self._profile.min_area_px]

        selected, selection_debug = select_best_component(
            components=components,
            frame_width=w,
            frame_height=h,
            prev_centroid=self._prev_centroid,
            cfg=self._selection_cfg,
        )
        if selected is not None:
            self._prev_centroid = selected.centroid_xy
            self._prev_bbox = selected.bbox_xyxy
            self._prev_area = selected.area

        quality_score = _quality_score(
            mask, selected, selection_debug.get("selected_score", 0.0))
        debug = {
            "global_threshold": float(global_threshold),
            "candidate_count": int(selection_debug.get("candidate_count", 0)),
            "selected_score": float(selection_debug.get("selected_score", 0.0)),
            "profile": self._config.threshold_profile,
            "warmup": int(warmup),
        }
        return PipelineFrameResult(
            timestamp_ms=packet.timestamp_ms,
            frame_index=packet.frame_index,
            mask=mask.astype(bool),
            selected_label=(selected.label if selected else None),
            selected_bbox_xyxy_px=(selected.bbox_xyxy if selected else None),
            selected_centroid_xy_px=(
                selected.centroid_xy if selected else None),
            selected_area_px=(selected.area if selected else None),
            candidate_count=int(selection_debug.get("candidate_count", 0)),
            quality_score=quality_score,
            debug=debug,
        )


def pipeline_result_to_hand_result(
    result: PipelineFrameResult,
    frame_width: int,
    frame_height: int,
) -> HandFrameResult:
    """Phase 4 handoff seam converting pipeline output to HandFrameResult."""
    if result.selected_bbox_xyxy_px is None or result.selected_centroid_xy_px is None:
        return HandFrameResult(
            status=ResultStatus.NO_HAND,
            timestamp_ms=result.timestamp_ms,
            bbox_xyxy_norm=None,
            centroid_xy_norm=None,
            contour_points_norm=[],
            quality_score=result.quality_score,
        )

    x0, y0, x1, y1 = result.selected_bbox_xyxy_px
    cx, cy = result.selected_centroid_xy_px
    bbox_norm = (
        x0 / max(float(frame_width), 1.0),
        y0 / max(float(frame_height), 1.0),
        x1 / max(float(frame_width), 1.0),
        y1 / max(float(frame_height), 1.0),
    )
    centroid_norm = (
        cx / max(float(frame_width), 1.0),
        cy / max(float(frame_height), 1.0),
    )
    return HandFrameResult(
        status=ResultStatus.OK,
        timestamp_ms=result.timestamp_ms,
        bbox_xyxy_norm=bbox_norm,
        centroid_xy_norm=centroid_norm,
        contour_points_norm=[],
        quality_score=result.quality_score,
    )


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected frame_rgb shape (H, W, 3).")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _quality_score(mask: np.ndarray, selected: ComponentStats | None, selected_score: float) -> float:
    occupancy = float(mask.mean())
    occupancy_term = max(0.0, 1.0 - abs(occupancy - 0.18) / 0.18)
    selected_term = 0.0 if selected is None else min(
        1.0, selected.area / max(mask.size * 0.22, 1.0))
    score = 0.4 * occupancy_term + 0.4 * \
        selected_term + 0.2 * float(selected_score)
    return float(np.clip(score, 0.0, 1.0))
