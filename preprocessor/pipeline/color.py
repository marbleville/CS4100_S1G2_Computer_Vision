"""Color-space transforms and score map fusion."""

from __future__ import annotations

import numpy as np


def rgb_to_grayscale(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale float32 in [0, 1]."""
    frame = frame_rgb.astype(np.float32) / 255.0
    gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    return gray.astype(np.float32)


def rgb_to_hsv(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to HSV float32 with channels in [0, 1]."""
    rgb = frame_rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    delta = cmax - cmin

    hue = np.zeros_like(cmax, dtype=np.float32)
    nonzero = delta > 0

    idx = nonzero & (cmax == r)
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = nonzero & (cmax == g)
    hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = nonzero & (cmax == b)
    hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    hue = hue / 6.0

    sat = np.zeros_like(cmax, dtype=np.float32)
    nonzero_cmax = cmax > 0
    sat[nonzero_cmax] = delta[nonzero_cmax] / cmax[nonzero_cmax]

    val = cmax.astype(np.float32)
    return np.stack((hue, sat, val), axis=2).astype(np.float32)


def rgb_to_ycbcr(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to YCbCr float32 in [0, 1]-scaled channels."""
    rgb = frame_rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return np.stack((y, cb, cr), axis=2).astype(np.float32)


def _gaussian_membership(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2).astype(np.float32)


def fused_skin_confidence(
    hsv: np.ndarray,
    ycbcr: np.ndarray,
    foreground_score: np.ndarray | None = None,
    fg_weight: float = 0.2,
) -> np.ndarray:
    """Build a fused hand-likelihood map in [0, 1]."""
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    # Broad, conservative priors for skin-like colors.
    hue_score = _gaussian_membership(h, mean=0.08, sigma=0.08)
    sat_score = _gaussian_membership(s, mean=0.45, sigma=0.22)
    val_score = _gaussian_membership(v, mean=0.65, sigma=0.25)
    cb_score = _gaussian_membership(cb, mean=0.46, sigma=0.10)
    cr_score = _gaussian_membership(cr, mean=0.62, sigma=0.10)

    base = (
        0.22 * hue_score
        + 0.12 * sat_score
        + 0.10 * val_score
        + 0.28 * cb_score
        + 0.28 * cr_score
    ).astype(np.float32)

    if foreground_score is None:
        return np.clip(base, 0.0, 1.0)

    fg = foreground_score.astype(np.float32)
    fg = fg - fg.min()
    denom = float(fg.max()) or 1.0
    fg = fg / denom
    fused = (1.0 - fg_weight) * base + fg_weight * fg
    return np.clip(fused, 0.0, 1.0).astype(np.float32)
