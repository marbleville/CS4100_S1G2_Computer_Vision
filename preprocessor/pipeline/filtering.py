"""Numpy-based filtering and binary morphology utilities."""

from __future__ import annotations

import numpy as np


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            patch = padded[y : y + kh, x : x + kw]
            out[y, x] = float(np.sum(patch * kernel))
    return out


def box_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= float(kernel_size * kernel_size)
    return _convolve2d(image.astype(np.float32), kernel).astype(np.float32)


def _gaussian_kernel1d(kernel_size: int, sigma: float | None = None) -> np.ndarray:
    if sigma is None:
        sigma = max(1.0, kernel_size / 3.0)
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    k /= np.sum(k)
    return k.astype(np.float32)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float | None = None) -> np.ndarray:
    k = _gaussian_kernel1d(kernel_size, sigma)
    # Separable blur via row then col pass.
    tmp = _convolve2d(image.astype(np.float32), k.reshape(1, -1))
    out = _convolve2d(tmp, k.reshape(-1, 1))
    return out.astype(np.float32)


def _binary_min_filter(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    r = kernel_size // 2
    padded = np.pad(mask.astype(bool), ((r, r), (r, r)), mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            patch = padded[y : y + kernel_size, x : x + kernel_size]
            out[y, x] = bool(np.all(patch))
    return out


def _binary_max_filter(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    r = kernel_size // 2
    padded = np.pad(mask.astype(bool), ((r, r), (r, r)), mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            patch = padded[y : y + kernel_size, x : x + kernel_size]
            out[y, x] = bool(np.any(patch))
    return out


def binary_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    eroded = _binary_min_filter(mask, kernel_size=kernel_size)
    return _binary_max_filter(eroded, kernel_size=kernel_size)


def binary_close(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    dilated = _binary_max_filter(mask, kernel_size=kernel_size)
    return _binary_min_filter(dilated, kernel_size=kernel_size)
