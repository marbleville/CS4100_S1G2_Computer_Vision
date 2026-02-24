"""Color-space transforms and score map fusion."""

from __future__ import annotations

import cv2
import numpy as np

NUMERIC_EPSILON = 1e-6

# Gaussian priors for skin-likelihood cues.
# Tuning guidance:
# - `mean` shifts the expected channel center.
# - `sigma` widens/narrows acceptance around that center.
# Channel ranges are normalized to [0, 1].
SKIN_PRIOR_GAUSSIANS: dict[str, tuple[float, float]] = {
    "hue": (0.08, 0.08),
    "saturation": (0.45, 0.22),
    "value": (0.65, 0.25),
    "cb": (0.46, 0.10),
    "cr": (0.62, 0.10),
}

# Weighted contribution of each cue to base skin confidence.
# Tuning guidance:
# - Increase chroma (`cb`/`cr`) weights for stable lighting.
# - Increase `value`/`saturation` in low-light where chroma is noisier.
SKIN_PRIOR_WEIGHTS: dict[str, float] = {
    "hue": 0.22,
    "saturation": 0.12,
    "value": 0.10,
    "cb": 0.28,
    "cr": 0.28,
}

# Fraction of fused confidence assigned to foreground-difference cue.
# Tuning guidance:
# - Raise for static-camera scenes with clutter.
# - Lower for camera motion or unstable backgrounds.
DEFAULT_FOREGROUND_WEIGHT = 0.20


def rgb_to_grayscale(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale float32 in [0, 1]."""
    gray_u8 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return (gray_u8.astype(np.float32) / 255.0).astype(np.float32)


def rgb_to_hsv(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to HSV float32 with channels in [0, 1]."""
    hsv_u8 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hsv = hsv_u8.astype(np.float32)
    # OpenCV HSV ranges: H [0,179], S [0,255], V [0,255]
    hsv[:, :, 0] = hsv[:, :, 0] / 179.0
    hsv[:, :, 1] = hsv[:, :, 1] / 255.0
    hsv[:, :, 2] = hsv[:, :, 2] / 255.0
    return np.clip(hsv, 0.0, 1.0).astype(np.float32)


def rgb_to_ycbcr(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to YCbCr float32 in [0, 1]-scaled channels."""
    # OpenCV returns YCrCb order; reorder to YCbCr to preserve contract.
    ycrcb_u8 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycrcb_u8[:, :, 0] / 255.0
    cr = ycrcb_u8[:, :, 1] / 255.0
    cb = ycrcb_u8[:, :, 2] / 255.0
    return np.stack((y, cb, cr), axis=2).astype(np.float32)


def _gaussian_membership(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, NUMERIC_EPSILON)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2).astype(np.float32)


def fused_skin_confidence(
    hsv: np.ndarray,
    ycbcr: np.ndarray,
    foreground_score: np.ndarray | None = None,
    fg_weight: float = DEFAULT_FOREGROUND_WEIGHT,
) -> np.ndarray:
    """Build a fused hand-likelihood map in [0, 1]."""
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    hue_mean, hue_sigma = SKIN_PRIOR_GAUSSIANS["hue"]
    sat_mean, sat_sigma = SKIN_PRIOR_GAUSSIANS["saturation"]
    val_mean, val_sigma = SKIN_PRIOR_GAUSSIANS["value"]
    cb_mean, cb_sigma = SKIN_PRIOR_GAUSSIANS["cb"]
    cr_mean, cr_sigma = SKIN_PRIOR_GAUSSIANS["cr"]

    hue_score = _gaussian_membership(h, mean=hue_mean, sigma=hue_sigma)
    sat_score = _gaussian_membership(s, mean=sat_mean, sigma=sat_sigma)
    val_score = _gaussian_membership(v, mean=val_mean, sigma=val_sigma)
    cb_score = _gaussian_membership(cb, mean=cb_mean, sigma=cb_sigma)
    cr_score = _gaussian_membership(cr, mean=cr_mean, sigma=cr_sigma)

    base = (
        SKIN_PRIOR_WEIGHTS["hue"] * hue_score
        + SKIN_PRIOR_WEIGHTS["saturation"] * sat_score
        + SKIN_PRIOR_WEIGHTS["value"] * val_score
        + SKIN_PRIOR_WEIGHTS["cb"] * cb_score
        + SKIN_PRIOR_WEIGHTS["cr"] * cr_score
    ).astype(np.float32)

    if foreground_score is None:
        return np.clip(base, 0.0, 1.0)

    fg = foreground_score.astype(np.float32)
    fg = fg - fg.min()
    denom = float(fg.max()) or 1.0
    fg = fg / denom
    fused = (1.0 - fg_weight) * base + fg_weight * fg
    return np.clip(fused, 0.0, 1.0).astype(np.float32)
