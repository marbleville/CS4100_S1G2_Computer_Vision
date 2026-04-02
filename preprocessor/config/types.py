"""Typed configuration objects for preprocessor initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LightingMode = Literal["normal", "low_light", "auto"]
SKIN_PRIOR_KEYS: tuple[str, str, str, str, str] = (
    "hue",
    "saturation",
    "value",
    "cb",
    "cr",
)
REQUIRED_SKIN_PRIOR_KEYS = frozenset(SKIN_PRIOR_KEYS)


@dataclass(frozen=True, slots=True)
class SkinFusionProfile:
    """Tunable color-fusion profile used to produce the score map."""

    gaussians: dict[str, tuple[float, float]]
    weights: dict[str, float]
    foreground_weight: float

    def __post_init__(self) -> None:
        gaussian_keys = frozenset(self.gaussians)
        weight_keys = frozenset(self.weights)
        if gaussian_keys != REQUIRED_SKIN_PRIOR_KEYS:
            raise ValueError(
                f"`gaussians` keys must be {sorted(REQUIRED_SKIN_PRIOR_KEYS)}."
            )
        if weight_keys != REQUIRED_SKIN_PRIOR_KEYS:
            raise ValueError(
                f"`weights` keys must be {sorted(REQUIRED_SKIN_PRIOR_KEYS)}."
            )

        normalized_gaussians: dict[str, tuple[float, float]] = {}
        for key in SKIN_PRIOR_KEYS:
            try:
                mean, sigma = self.gaussians[key]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"`gaussians[{key}]` must contain exactly two numeric values."
                ) from exc
            mean = float(mean)
            sigma = float(sigma)
            if sigma <= 0.0:
                raise ValueError(f"`gaussians[{key}][1]` must be positive.")
            normalized_gaussians[key] = (mean, sigma)

        normalized_weights: dict[str, float] = {}
        total_weight = 0.0
        for key in SKIN_PRIOR_KEYS:
            weight = float(self.weights[key])
            if weight < 0.0:
                raise ValueError(f"`weights[{key}]` must be non-negative.")
            total_weight += weight
            normalized_weights[key] = weight

        if total_weight <= 0.0:
            raise ValueError("`weights` must sum to a positive value.")

        foreground_weight = float(self.foreground_weight)
        if not 0.0 <= foreground_weight <= 1.0:
            raise ValueError("`foreground_weight` must be in [0, 1].")

        object.__setattr__(self, "gaussians", normalized_gaussians)
        object.__setattr__(self, "weights", normalized_weights)
        object.__setattr__(self, "foreground_weight", foreground_weight)


@dataclass(frozen=True, slots=True)
class LightingSwitchConfig:
    """Controls fixed or automatic selection between lighting profiles."""

    mode: LightingMode = "normal"
    enter_low_light_threshold: float = 0.22
    exit_low_light_threshold: float = 0.28
    ema_alpha: float = 0.25

    def __post_init__(self) -> None:
        if self.mode not in {"normal", "low_light", "auto"}:
            raise ValueError("`mode` must be one of: normal, low_light, auto.")

        enter_low_light_threshold = float(self.enter_low_light_threshold)
        exit_low_light_threshold = float(self.exit_low_light_threshold)
        ema_alpha = float(self.ema_alpha)

        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("`ema_alpha` must be in (0, 1].")
        if exit_low_light_threshold <= enter_low_light_threshold:
            raise ValueError(
                "`exit_low_light_threshold` must be greater than "
                "`enter_low_light_threshold`."
            )

        object.__setattr__(
            self, "enter_low_light_threshold", enter_low_light_threshold
        )
        object.__setattr__(
            self, "exit_low_light_threshold", exit_low_light_threshold
        )
        object.__setattr__(self, "ema_alpha", ema_alpha)


def _default_normal_skin_profile() -> SkinFusionProfile:
    from preprocessor.pipeline.color import build_default_normal_skin_profile

    return build_default_normal_skin_profile()


def _default_low_light_skin_profile() -> SkinFusionProfile:
    from preprocessor.pipeline.color import build_default_low_light_skin_profile

    return build_default_low_light_skin_profile()


@dataclass(frozen=True, slots=True)
class PreprocessorConfig:
    """Configuration for `init_preprocessor`."""

    input_mode: Literal["webcam", "local_video"]
    video_path: str | None = None
    frame_size: tuple[int, int] = (640, 480)
    threshold_profile: str = "default"
    candidate_frame_size_px: int = 128
    candidate_buffer_size: int = 32
    normal_skin_profile: SkinFusionProfile = field(
        default_factory=_default_normal_skin_profile
    )
    low_light_skin_profile: SkinFusionProfile = field(
        default_factory=_default_low_light_skin_profile
    )
    lighting_switch: LightingSwitchConfig = field(
        default_factory=LightingSwitchConfig
    )

    def __post_init__(self) -> None:
        frame_width, frame_height = self.frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("`frame_size` dimensions must be positive.")
        if self.candidate_frame_size_px <= 0:
            raise ValueError("`candidate_frame_size_px` must be positive.")
        if self.candidate_buffer_size <= 0:
            raise ValueError("`candidate_buffer_size` must be positive.")
