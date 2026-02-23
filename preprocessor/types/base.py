"""Optional helper dataclasses for normalized geometry."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PointNorm:
    """Normalized point in image space."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BBoxNorm:
    """Normalized bounding box in xyxy format."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
