"""
AugmentationPipeline: Random image augmentations for gesture training data.

Applies random horizontal flip, brightness jitter, contrast jitter, and
rotation to normalized float32 hand crop images. Only use on training
data — val and test sets must remain unmodified.
"""

import numpy as np
from PIL import Image


class AugmentationPipeline:
    """
    Applies random augmentations to a normalized float32 hand crop image.

    Augmentations applied in order: flip → brightness → contrast → rotation.
    All parameters are configurable. Pass seed for reproducible output in tests.

    Only apply to training data. Val and test sets must remain unmodified.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        brightness_range: tuple[float, float] = (0.7, 1.3),
        contrast_range: tuple[float, float] = (0.7, 1.3),
        max_rotation_degrees: float = 15.0,
        seed: int | None = None,
    ):
        """
        Args:
            flip_prob:              Probability of applying a horizontal flip (default 0.5).
            brightness_range:       (min, max) multiplier for brightness jitter (default 0.7-1.3).
            contrast_range:         (min, max) multiplier for contrast jitter (default 0.7-1.3).
            max_rotation_degrees:   Maximum rotation angle in either direction (default 15.0).
            seed:                   Random seed for reproducibility. None for random behavior.
        """
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.max_rotation_degrees = max_rotation_degrees
        # Scoped RandomState so seed does not affect global numpy random state
        self._rng = np.random.RandomState(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a single normalized image.

        Args:
            image: Normalized float32 array, shape (128, 128, 3), values in [0.0, 1.0].

        Returns:
            Augmented float32 array, same shape (128, 128, 3), values clamped to [0.0, 1.0].
        """
        # Always work on a copy — never modify the input in place
        img = image.astype(np.float32).copy()
        img = self._apply_flip(img)
        img = self._apply_brightness(img)
        img = self._apply_contrast(img)
        img = self._apply_rotation(img)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def _apply_flip(self, img: np.ndarray) -> np.ndarray:
        """Randomly flip the image horizontally with probability flip_prob."""
        if self._rng.random() < self.flip_prob:
            return np.fliplr(img).copy()
        return img

    def _apply_brightness(self, img: np.ndarray) -> np.ndarray:
        """
        Multiply all pixel values by a random brightness factor.
        Simulates different lighting conditions.
        """
        factor = self._rng.uniform(*self.brightness_range)
        return img * factor

    def _apply_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Adjust contrast by blending toward the per-channel mean.
        output = mean + factor * (image - mean)
        Simulates different camera exposure settings.
        """
        factor = self._rng.uniform(*self.contrast_range)
        mean = img.mean(axis=(0, 1), keepdims=True)
        return mean + factor * (img - mean)

    def _apply_rotation(self, img: np.ndarray) -> np.ndarray:
        """
        Rotate by a random angle within [-max_rotation_degrees, +max_rotation_degrees].
        Fills corners introduced by rotation with the image mean to avoid black borders.
        Converts to PIL for rotation then back to float32 numpy.
        Skips conversion entirely if max_rotation_degrees is 0 to avoid rounding errors.
        """
        if self.max_rotation_degrees == 0.0:
            return img

        angle = self._rng.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        fill_value = float(img.mean())

        # PIL needs uint8 — scale to [0, 255], rotate, scale back to [0.0, 1.0]
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="RGB")
        fill_rgb = (int(fill_value * 255),) * 3
        rotated = pil_img.rotate(angle, expand=False, fillcolor=fill_rgb)
        return np.asarray(rotated, dtype=np.float32) / 255.0