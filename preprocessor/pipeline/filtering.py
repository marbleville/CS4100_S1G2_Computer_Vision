"""Filtering and binary morphology utilities backed by OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def box_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    image_f = image.astype(np.float32)
    return cv2.blur(image_f, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT).astype(
        np.float32
    )


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float | None = None) -> np.ndarray:
    image_f = image.astype(np.float32)
    sigma_x = 0.0 if sigma is None else float(sigma)
    return cv2.GaussianBlur(
        image_f, (kernel_size, kernel_size), sigmaX=sigma_x, borderType=cv2.BORDER_REFLECT
    ).astype(np.float32)


def binary_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


def binary_close(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)
