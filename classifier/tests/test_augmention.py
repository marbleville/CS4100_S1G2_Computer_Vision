"""
Tests for AugmentationPipeline.

Tests are written against observable input/output behavior only.
No knowledge of internal augmentation order or implementation
details is assumed.
"""

import numpy as np
import pytest
from classifier.data.augmentation import AugmentationPipeline

# Helpers

def random_image(seed: int = 0) -> np.ndarray:
    """Return a random normalized float32 image of shape (128, 128, 3)."""
    rng = np.random.RandomState(seed)
    return rng.rand(128, 128, 3).astype(np.float32)

def flat_image(value: float) -> np.ndarray:
    """Return a flat image filled with a single value."""
    return np.full((128, 128, 3), value, dtype=np.float32)

# Output contract tests

def test_output_shape_is_preserved():
    """Output must have the same shape as input."""
    pipeline = AugmentationPipeline(seed=0)
    img = random_image()
    result = pipeline(img)
    assert result.shape == (128, 128, 3), f"Expected (128, 128, 3), got {result.shape}"

def test_output_dtype_is_float32():
    """Output must always be float32."""
    pipeline = AugmentationPipeline(seed=0)
    result = pipeline(random_image())
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

def test_output_values_are_clamped_to_unit_range():
    """Output values must always be in [0.0, 1.0] regardless of augmentation."""
    pipeline = AugmentationPipeline(seed=0)
    result = pipeline(random_image())
    assert result.min() >= 0.0, f"Output contains values below 0.0: {result.min()}"
    assert result.max() <= 1.0, f"Output contains values above 1.0: {result.max()}"

def test_output_values_clamped_with_extreme_brightness():
    """Values must remain in [0.0, 1.0] even with extreme brightness settings."""
    pipeline = AugmentationPipeline(
        brightness_range=(3.0, 3.0),  # force very bright
        contrast_range=(1.0, 1.0),
        max_rotation_degrees=0.0,
        flip_prob=0.0,
        seed=0,
    )
    result = pipeline(flat_image(0.9))
    assert result.max() <= 1.0

def test_input_is_not_modified_in_place():
    """Pipeline must never modify the input array."""
    pipeline = AugmentationPipeline(seed=0)
    img = random_image()
    original = img.copy()
    pipeline(img)
    assert np.array_equal(img, original), "Input array was modified during augmentation"

# Reproducibility tests

def test_seeded_pipeline_is_reproducible():
    """Two pipelines with the same seed must produce identical output."""
    img = random_image()
    p1 = AugmentationPipeline(seed=42)
    p2 = AugmentationPipeline(seed=42)
    assert np.allclose(p1(img), p2(img)), "Seeded pipelines produced different results"

def test_different_seeds_produce_different_results():
    """Two pipelines with different seeds should produce different output."""
    img = random_image()
    p1 = AugmentationPipeline(seed=0)
    p2 = AugmentationPipeline(seed=99)
    assert not np.array_equal(p1(img), p2(img)), (
        "Different seeds produced identical results — seed may not be working"
    )

def test_seed_does_not_affect_global_random_state():
    """Using a seeded pipeline must not change numpy's global random state."""
    np.random.seed(123)
    before = np.random.rand(5)
    np.random.seed(123)
    AugmentationPipeline(seed=42)(random_image())
    after = np.random.rand(5)
    assert np.allclose(before, after), (
        "AugmentationPipeline affected global numpy random state"
    )

# Augmentation behavior tests

def test_no_flip_when_prob_zero():
    """With flip_prob=0, output should never be a horizontal flip of the input."""
    pipeline = AugmentationPipeline(
        flip_prob=0.0,
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        max_rotation_degrees=0.0,
        seed=0,
    )
    img = random_image()
    result = pipeline(img)
    assert not np.allclose(result, np.fliplr(img)), (
        "Image was flipped even with flip_prob=0"
    )

def test_always_flip_when_prob_one():
    """With flip_prob=1 and all other augmentations disabled, output must be flipped."""
    pipeline = AugmentationPipeline(
        flip_prob=1.0,
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        max_rotation_degrees=0.0,
        seed=0,
    )
    img = random_image()
    result = pipeline(img)
    assert np.allclose(result, np.fliplr(img), atol=1e-5), (
        "Image was not flipped with flip_prob=1"
    )

def test_no_change_when_all_augmentations_disabled():
    """With all augmentations at identity settings, output should match input."""
    pipeline = AugmentationPipeline(
        flip_prob=0.0,
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        max_rotation_degrees=0.0,
        seed=0,
    )
    img = random_image()
    result = pipeline(img)
    assert np.allclose(result, img, atol=1e-5), (
        "Output differed from input with all augmentations at identity settings"
    )

def test_repeated_calls_produce_different_results():
    """Without a fixed seed, repeated calls on the same image should differ."""
    pipeline = AugmentationPipeline(seed=None)
    img = random_image()
    results = [pipeline(img) for _ in range(5)]
    # At least one pair should differ
    all_same = all(np.array_equal(results[0], r) for r in results[1:])
    assert not all_same, "All augmented outputs were identical — randomness may be broken"