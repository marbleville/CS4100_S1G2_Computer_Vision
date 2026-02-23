from __future__ import annotations

import numpy as np

from preprocessor.pipeline.filtering import binary_close, binary_open, box_blur, gaussian_blur


def test_blurs_preserve_shape_and_reduce_variance() -> None:
    rng = np.random.default_rng(42)
    image = rng.random((20, 20), dtype=np.float32)
    box = box_blur(image, kernel_size=3)
    gauss = gaussian_blur(image, kernel_size=5)

    assert box.shape == image.shape
    assert gauss.shape == image.shape
    assert float(box.var()) < float(image.var())
    assert float(gauss.var()) < float(image.var())


def test_binary_open_close_cleanup_noise() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    mask[0, 0] = True  # tiny speckle
    opened = binary_open(mask, kernel_size=3)
    closed = binary_close(opened, kernel_size=3)
    assert not opened[0, 0]
    assert closed[3, 3]
