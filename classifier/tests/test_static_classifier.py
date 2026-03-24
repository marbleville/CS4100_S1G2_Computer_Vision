"""
Tests for StaticClassifier and GestureResult.

Tests are written against observable input/output behavior only.
No knowledge of internal CNN architecture, threshold values,
or normalization logic is assumed.
"""

import numpy as np
import pytest
from classifier.static_classifier import StaticClassifier, GestureResult
from classifier.config import STATIC_GESTURE_CLASSES, CONFIDENCE_THRESHOLD
from preprocessor.types.results import HandDetectionResult

# Helpers

def make_detection(
    hand_detected: bool = True,
    confidence_level: float = 1.0,
    timestamp_ms: int = 1000,
    crop: np.ndarray | None = None,
) -> HandDetectionResult:
    """Build a HandDetectionResult with sensible defaults for testing."""
    if crop is None:
        crop = np.zeros((128, 128, 3), dtype=np.uint8)
    return HandDetectionResult(
        hand_detected=hand_detected,
        confidence_level=confidence_level,
        crop_rgb=crop,
        timestamp_ms=timestamp_ms,
        bbox=None,
    )

# GestureResult contract tests

def test_gesture_result_has_required_fields():
    """GestureResult must expose gesture, confidence, hand_detected, timestamp_ms."""
    result = GestureResult(
        gesture="fist",
        confidence=0.9,
        hand_detected=True,
        timestamp_ms=500,
    )
    assert hasattr(result, "gesture")
    assert hasattr(result, "confidence")
    assert hasattr(result, "hand_detected")
    assert hasattr(result, "timestamp_ms")

def test_gesture_result_accepts_none_gesture():
    """gesture field must accept None to represent no prediction."""
    result = GestureResult(gesture=None, confidence=0.0, hand_detected=False, timestamp_ms=0)
    assert result.gesture is None

def test_gesture_result_stores_values_correctly():
    """GestureResult must store exactly the values passed in."""
    result = GestureResult(gesture="fist", confidence=0.88, hand_detected=True, timestamp_ms=1234)
    assert result.gesture == "fist"
    assert result.confidence == 0.88
    assert result.hand_detected is True
    assert result.timestamp_ms == 1234

# StaticClassifier output contract tests

def test_classify_returns_gesture_result():
    """classify must always return a GestureResult regardless of input."""
    classifier = StaticClassifier()
    result = classifier.classify(make_detection())
    assert isinstance(result, GestureResult)

def test_classify_no_hand_returns_none_gesture():
    """If hand_detected is False, gesture must be None without running inference."""
    classifier = StaticClassifier()
    result = classifier.classify(make_detection(hand_detected=False, confidence_level=1.0))
    assert result.gesture is None

def test_classify_no_hand_passes_through_hand_detected_false():
    """hand_detected=False must be passed through to GestureResult."""
    classifier = StaticClassifier()
    result = classifier.classify(make_detection(hand_detected=False))
    assert result.hand_detected is False

def test_classify_low_confidence_returns_none_gesture():
    """If Module B confidence is below threshold, gesture must be None."""
    classifier = StaticClassifier()
    low_confidence = CONFIDENCE_THRESHOLD - 0.01
    result = classifier.classify(make_detection(confidence_level=low_confidence))
    assert result.gesture is None

def test_classify_no_model_returns_none_gesture():
    """With no model loaded, gesture must be None even for a valid high-confidence crop."""
    classifier = StaticClassifier(model=None)
    result = classifier.classify(make_detection(hand_detected=True, confidence_level=1.0))
    assert result.gesture is None

def test_classify_confidence_is_float_in_valid_range():
    """Confidence in GestureResult must be a float between 0.0 and 1.0."""
    classifier = StaticClassifier()
    result = classifier.classify(make_detection())
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

def test_classify_gesture_is_none_or_valid_class():
    """If gesture is not None it must be one of the known static gesture classes."""
    classifier = StaticClassifier()
    result = classifier.classify(make_detection())
    assert result.gesture is None or result.gesture in STATIC_GESTURE_CLASSES

def test_classify_does_not_crash_on_different_crop_values():
    """Classifier must not raise for any valid uint8 crop content."""
    classifier = StaticClassifier()
    for fill_value in [0, 128, 255]:
        crop = np.full((128, 128, 3), fill_value, dtype=np.uint8)
        result = classifier.classify(make_detection(crop=crop))
        assert isinstance(result, GestureResult)

def test_classify_does_not_modify_input_crop():
    """classify must not modify the crop array passed in HandDetectionResult."""
    classifier = StaticClassifier()
    crop = np.ones((128, 128, 3), dtype=np.uint8) * 100
    original = crop.copy()
    classifier.classify(make_detection(crop=crop))
    assert np.array_equal(crop, original), "crop_rgb was modified during classify"