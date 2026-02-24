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
from preprocessor.pipeline.components import ComponentStats, connected_components, coalesce_components
from preprocessor.pipeline.filtering import binary_close, binary_open, gaussian_blur
from preprocessor.pipeline.thresholding import (
    global_percentile_threshold,
    local_tile_threshold,
)
from preprocessor.pipeline.types import PipelineFrameResult
from preprocessor.types.enums import ResultStatus
from preprocessor.types.results import HandFrameResult


@dataclass(frozen=True, slots=True)
class PipelineProfile:
    """Tunable preprocessing profile.

    Field tuning guide:
    - `percentile`: global threshold percentile on fused score map. Lower keeps more pixels.
    - `local_percentile`: tile-local threshold percentile. Lower increases local sensitivity.
    - `tiles_x`, `tiles_y`: local threshold grid density. Higher adapts better to uneven lighting.
    - `threshold_blend`: mix of local/global thresholds (0=local only, 1=global only).
    - `gaussian_kernel`: blur kernel for score smoothing. Higher suppresses noise but softens edges.
    - `morphology_kernel`: kernel size for open/close cleanup. Higher removes more small artifacts.
    - `min_area_percent`: minimum component area as fraction of frame area.
    - `components_touch_edge`: include components touching frame border when True.
    - `background_enabled`: enables running-average foreground cue.
    - `background_alpha`: background update rate; higher adapts faster, lower is steadier.
    - `background_warmup`: initial frame count where background cue is treated cautiously.
    """

    percentile: float
    local_percentile: float
    tiles_x: int
    tiles_y: int
    threshold_blend: float
    gaussian_kernel: int
    morphology_kernel: int
    min_area_percent: float
    components_touch_edge: bool
    background_enabled: bool
    background_alpha: float
    background_warmup: int


_PROFILE_MAP: dict[str, PipelineProfile] = {
    "default": PipelineProfile(
        # Keep top ~20% fused-confidence pixels globally.
        percentile=80.0,
        # Same percentile locally to adapt to lighting variation per tile.
        local_percentile=80.0,
        tiles_x=8,
        tiles_y=6,
        # 0.5 balances local sensitivity with global stability.
        threshold_blend=0.50,
        # 5x5 smoothing reduces speckle before thresholding.
        gaussian_kernel=5,
        # 3x3 open/close removes tiny islands and fills tiny holes.
        morphology_kernel=3,
        min_area_percent=0.01,
        components_touch_edge=False,
        background_enabled=False,
        # If enabled: slow background update for stability.
        background_alpha=0.08,
        background_warmup=8,
    ),
}


class PreprocessingPipeline:
    """Stateful frame preprocessing pipeline."""

    def __init__(self, config: PreprocessorConfig) -> None:
        self._config = config
        self._profile = _PROFILE_MAP.get(
            config.threshold_profile, _PROFILE_MAP["default"])
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

        _, components = connected_components(mask)
        components = coalesce_components(components)

        h, w = frame.shape[:2]
        frame_area = w * h
        components = self.filter_candidate_components(components, frame_area)

        return PipelineFrameResult(
            timestamp_ms=packet.timestamp_ms,
            frame_index=packet.frame_index,
            mask=mask.astype(bool),
            candidates=components,
        )

    def filter_candidate_components(self, components: list[ComponentStats], frame_area) -> list[ComponentStats]:
        """Filters unwanted compoents based on setting from processor profile"""

        components = [c for c in components if c.area / frame_area >=
                      self._profile.min_area_percent]
        components = [
            c for c in components if self._profile.components_touch_edge or not c.touches_border]

        return components


def pipeline_result_to_hand_result(
    result: PipelineFrameResult,
) -> HandFrameResult:
    """Phase 4 handoff seam converting pipeline output to HandFrameResult."""
    if result.candidates == []:
        return HandFrameResult(
            status=ResultStatus.NO_HAND,
            timestamp_ms=result.timestamp_ms,
            candidates_bbox_px=[]
        )

    candidate_bbox = list(
        map(lambda candidate: candidate.bbox_xyxy, result.candidates))

    return HandFrameResult(
        status=ResultStatus.OK,
        timestamp_ms=result.timestamp_ms,
        candidates_bbox_px=candidate_bbox
    )


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected frame_rgb shape (H, W, 3).")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr
