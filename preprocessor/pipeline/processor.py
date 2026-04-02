"""Stateful preprocessing pipeline orchestration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from preprocessor.config.types import PreprocessorConfig, SkinFusionProfile
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
from preprocessor.types.results import HandCandidateFrame, HandFrameResult


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
        components_touch_edge=True,
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
        self._smoothed_luma: float | None = None
        self._active_light_mode: Literal["normal", "low_light"] = "normal"
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_bbox: tuple[int, int, int, int] | None = None
        self._prev_area: int | None = None
        self._bg_model = RunningBackgroundModel(
            alpha=self._profile.background_alpha,
            warmup_frames=self._profile.background_warmup,
        )
        self._candidate_buffer: deque[HandCandidateFrame] = deque(
            maxlen=config.candidate_buffer_size
        )
        self._candidate_buffer_overwrites = 0

    def reset(self) -> None:
        self._smoothed_luma = None
        self._active_light_mode = "normal"
        self._prev_centroid = None
        self._prev_bbox = None
        self._prev_area = None
        self._bg_model.reset()
        self._candidate_buffer.clear()
        self._candidate_buffer_overwrites = 0

    def process(self, packet: FramePacket) -> PipelineFrameResult:
        frame = _validate_frame(packet.frame_rgb)

        gray = rgb_to_grayscale(frame)
        hsv = rgb_to_hsv(frame)
        ycbcr = rgb_to_ycbcr(frame)
        frame_median_luma = float(np.median(gray))
        active_skin_profile = self._select_skin_profile(frame_median_luma)

        fg_score = None
        warmup = False
        if self._profile.background_enabled:
            fg_score, warmup = self._bg_model.update_and_score(gray)
            if warmup:
                fg_score = fg_score * 0.25

        score_map = fused_skin_confidence(
            hsv,
            ycbcr,
            profile=active_skin_profile,
            foreground_score=fg_score,
        )
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
        components = sorted(
            components, key=lambda candidate: candidate.area, reverse=True)
        candidate_frames = self._extract_candidate_frames(
            packet=packet,
            frame=frame,
            components=components,
        )
        self._enqueue_candidate_frames(candidate_frames)

        return PipelineFrameResult(
            timestamp_ms=packet.timestamp_ms,
            frame_index=packet.frame_index,
            mask=mask.astype(bool),
            candidates=components,
            candidate_frames=candidate_frames,
            debug={
                "candidate_count": len(components),
                "candidate_buffer_depth": len(self._candidate_buffer),
                "candidate_buffer_overwrites": self._candidate_buffer_overwrites,
                "candidate_frame_size_px": self._config.candidate_frame_size_px,
                "active_light_mode": self._active_light_mode,
                "frame_median_luma": frame_median_luma,
                "smoothed_luma": float(self._smoothed_luma or frame_median_luma),
            },
        )

    def filter_candidate_components(
        self,
        components: list[ComponentStats],
        frame_area: int,
    ) -> list[ComponentStats]:
        """Filters unwanted compoents based on setting from processor profile"""

        components = [
            c for c in components
            if c.area / frame_area >= self._profile.min_area_percent
        ]
        components = [
            c
            for c in components
            if self._profile.components_touch_edge or not c.touches_border
        ]

        return components

    def _extract_candidate_frames(
        self,
        packet: FramePacket,
        frame: np.ndarray,
        components: list[ComponentStats],
    ) -> list[HandCandidateFrame]:
        candidate_frames: list[HandCandidateFrame] = []
        for candidate_index, component in enumerate(components):
            candidate_frames.append(
                HandCandidateFrame(
                    frame_rgb=_extract_square_candidate_frame(
                        frame=frame,
                        bbox_xyxy=component.bbox_xyxy,
                        output_size_px=self._config.candidate_frame_size_px,
                    ),
                    timestamp_ms=packet.timestamp_ms,
                    source_frame_index=packet.frame_index,
                    source_id=packet.source_id,
                    candidate_index=candidate_index,
                    bbox_xyxy_px=component.bbox_xyxy,
                )
            )
        return candidate_frames

    def _enqueue_candidate_frames(
        self,
        candidate_frames: list[HandCandidateFrame],
    ) -> None:
        for candidate_frame in candidate_frames:
            if len(self._candidate_buffer) == self._config.candidate_buffer_size:
                self._candidate_buffer_overwrites += 1
            self._candidate_buffer.append(candidate_frame)

    def _pop_next_candidate(self) -> HandCandidateFrame | None:
        if not self._candidate_buffer:
            return None
        return self._candidate_buffer.popleft()

    def _select_skin_profile(self, frame_median_luma: float) -> SkinFusionProfile:
        lighting_switch = self._config.lighting_switch
        previous_smoothed_luma = self._smoothed_luma
        if previous_smoothed_luma is None:
            self._smoothed_luma = frame_median_luma
        else:
            self._smoothed_luma = (
                lighting_switch.ema_alpha * frame_median_luma
                + (1.0 - lighting_switch.ema_alpha) * previous_smoothed_luma
            )

        if lighting_switch.mode == "normal":
            self._active_light_mode = "normal"
            return self._config.normal_skin_profile

        if lighting_switch.mode == "low_light":
            self._active_light_mode = "low_light"
            return self._config.low_light_skin_profile

        smoothed_luma = float(self._smoothed_luma)
        if self._active_light_mode == "low_light":
            if smoothed_luma > lighting_switch.exit_low_light_threshold:
                self._active_light_mode = "normal"
        elif smoothed_luma < lighting_switch.enter_low_light_threshold:
            self._active_light_mode = "low_light"

        if self._active_light_mode == "low_light":
            return self._config.low_light_skin_profile
        return self._config.normal_skin_profile


def pipeline_result_to_hand_result(
    result: PipelineFrameResult,
) -> HandFrameResult:
    """Phase 4 handoff seam converting pipeline output to HandFrameResult."""
    if result.candidate_frames == []:
        return HandFrameResult(
            status=ResultStatus.NO_HAND,
            timestamp_ms=result.timestamp_ms,
            candidates=[],
        )

    return HandFrameResult(
        status=ResultStatus.OK,
        timestamp_ms=result.timestamp_ms,
        candidates=list(result.candidate_frames),
    )


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected frame_rgb shape (H, W, 3).")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _extract_square_candidate_frame(
    frame: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    output_size_px: int,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox_xyxy
    if x1 < x0 or y1 < y0:
        raise ValueError(
            "Candidate bbox must have non-negative width and height.")

    width = x1 - x0 + 1
    height = y1 - y0 + 1
    side = max(width, height)

    pad_x = side - width
    pad_y = side - height

    square_x0 = x0 - (pad_x // 2)
    square_x1 = x1 + (pad_x - (pad_x // 2))
    square_y0 = y0 - (pad_y // 2)
    square_y1 = y1 + (pad_y - (pad_y // 2))

    square_frame = np.zeros((side, side, 3), dtype=np.uint8)
    frame_height, frame_width = frame.shape[:2]

    clip_x0 = max(0, square_x0)
    clip_y0 = max(0, square_y0)
    clip_x1 = min(frame_width - 1, square_x1)
    clip_y1 = min(frame_height - 1, square_y1)

    if clip_x0 <= clip_x1 and clip_y0 <= clip_y1:
        dest_x0 = clip_x0 - square_x0
        dest_y0 = clip_y0 - square_y0
        dest_x1 = dest_x0 + (clip_x1 - clip_x0 + 1)
        dest_y1 = dest_y0 + (clip_y1 - clip_y0 + 1)
        square_frame[dest_y0:dest_y1, dest_x0:dest_x1] = frame[
            clip_y0:clip_y1 + 1,
            clip_x0:clip_x1 + 1,
        ]

    if side == output_size_px:
        return square_frame

    interpolation = _resize_interpolation(
        source_size_px=side,
        target_size_px=output_size_px,
    )
    return cv2.resize(
        square_frame,
        (output_size_px, output_size_px),
        interpolation=interpolation,
    )


def _resize_interpolation(source_size_px: int, target_size_px: int) -> int:
    if target_size_px < source_size_px:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR
