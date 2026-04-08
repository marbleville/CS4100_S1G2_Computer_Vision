"""
StaticClassifier: Takes a HandDetectionResult from Module B and returns a GestureResult.
Runs the crop through the CNN, applies confidence thresholding, and returns a gesture label.
"""

import numpy as np
import torch
import torch.nn.functional as F
from preprocessor.types.results import HandDetectionResult
from classifier.data.adapter import candidate_to_detection
from classifier.config import (
    CONFIDENCE_THRESHOLD,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    STATIC_GESTURE_CLASSES,
)
from classifier.models.cnn import GestureCNN


class GestureResult:
    """
    Output of the static gesture classifier, passed to Module E for command mapping.

    Attributes:
        gesture:      Predicted gesture label (e.g. "thumbs_up"), or None if confidence
                      is below threshold or no hand was detected.
        confidence:   Softmax probability of the predicted class, in [0.0, 1.0].
        hand_detected: Passed through from the input HandDetectionResult.
        timestamp_ms: Passed through from the input HandDetectionResult.
    """
    def __init__(
        self,
        gesture: str | None,
        confidence: float,
        hand_detected: bool,
        timestamp_ms: int,
    ):
        self.gesture = gesture
        self.confidence = confidence
        self.hand_detected = hand_detected
        self.timestamp_ms = timestamp_ms

    def __repr__(self):
        return (
            f"GestureResult(gesture={self.gesture!r}, confidence={self.confidence:.2f}, "
            f"hand_detected={self.hand_detected}, timestamp_ms={self.timestamp_ms})"
        )


class StaticClassifier:
    """
    Classifies static hand gestures from a cropped hand image.

    Takes a HandDetectionResult from Module B, runs the crop through the CNN,
    applies confidence thresholding, and returns a GestureResult for Module E.
    """

    def __init__(self, model_path: str = "artifacts/models/cnn_best.pt"):
        """
        Args:
            model_path: Path to trained CNN checkpoint. Loads model into
                        eval mode ready for inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureCNN(num_classes=len(STATIC_GESTURE_CLASSES))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, crop_rgb: np.ndarray) -> np.ndarray:
        """
        Normalize the raw uint8 crop for CNN inference.

        Converts pixel values from [0, 255] to [0.0, 1.0] and applies
        per-channel mean/std normalization.

        Args:
            crop_rgb: Raw hand crop, shape (128, 128, 3), dtype uint8.

        Returns:
            Normalized float32 array of shape (128, 128, 3).
        """
        img = crop_rgb.astype(np.float32) / 255.0
        mean = np.array(NORMALIZATION_MEAN, dtype=np.float32)
        std = np.array(NORMALIZATION_STD, dtype=np.float32)
        img = (img - mean) / std
        return img.astype(np.float32)

    def _apply_threshold(
        self, probs: np.ndarray
    ) -> tuple[str | None, float]:
        """
        Apply confidence threshold to softmax probabilities.

        Args:
            probs: Softmax output array of shape (num_classes,).

        Returns:
            (gesture_label, confidence) if max prob >= CONFIDENCE_THRESHOLD,
            (None, max_confidence) otherwise.
        """
        max_idx = int(np.argmax(probs))
        max_conf = float(probs[max_idx])
        if max_conf >= CONFIDENCE_THRESHOLD:
            return STATIC_GESTURE_CLASSES[max_idx], max_conf
        return None, max_conf

    def classify(self, detection: HandDetectionResult) -> GestureResult:
        """
        Classify the gesture in a hand crop.

        Skips inference and returns no gesture if:
        - hand_detected is False
        - confidence_level from Module B is below threshold
        - no model is loaded yet

        Args:
            detection: HandDetectionResult from Module B containing the hand crop.

        Returns:
            GestureResult with gesture label and confidence for Module E.
        """
        # Guard: no hand detected or low quality crop from Module B
        if not detection.hand_detected or detection.confidence_level < CONFIDENCE_THRESHOLD:
            return GestureResult(
                gesture=None,
                confidence=detection.confidence_level,
                hand_detected=detection.hand_detected,
                timestamp_ms=detection.timestamp_ms,
            )

        # Preprocess crop and run CNN inference
        processed = self._preprocess(detection.crop_rgb)

        # Convert to tensor (1, 3, 128, 128)
        tensor = torch.from_numpy(
            processed.transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        gesture, confidence = self._apply_threshold(probs)

        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            hand_detected=detection.hand_detected,
            timestamp_ms=detection.timestamp_ms,
        )